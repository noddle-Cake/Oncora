"""Compound → target training (spec 05, Req 1–4; FR-5–FR-7, FR-22).

    uv run python -m oncora.models.target.train [--jobs N] [--max-targets K]

One LightGBM binary classifier per target, trained on that target's labeled compounds
(positives, measured and sampled negatives) from the shared modeling view. Per-target models
let each target keep its own negatives and calibration. Technique notes:

- Gradient-boosted trees fit an ensemble of shallow decision trees, each correcting the residual
  errors of the previous ones. They work well on sparse binary fingerprints. **Early stopping**
  on the validation compounds picks the number of trees, so trees are not added past the point
  where held-out loss stops improving.
- The split is **global**: a test compound is held out from every target's training data, so
  per-compound top-k is honest.
- Comparison model: L2-regularized logistic regression on the ECFP bits, a linear model with
  one weight per substructure.

The shipped artifact is the evaluated model (fit on train; val used for early stopping and
calibration). Test compounds therefore remain unseen by the model that serves predictions.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from oncora.chem.similarity import TrainingIndex
from oncora.config import (
    MLFLOW_URI,
    MODELS,
    ROOT,
    REPORTS_EVAL,
    SEED,
    TARGET_EARLY_STOPPING,
    TARGET_LGBM,
    TOPK_MAX_COMPOUNDS,
)
from oncora.data_prep.modeling import FEATURES, target_training_table
from oncora.data_prep.provenance import read_provenance
from oncora.evaluation.baselines import PopularityRanker, PriorBaseline
from oncora.evaluation.harness import evaluate_target, topk_sample
from oncora.evaluation.protocol import Partition, make_partition
from oncora.evaluation.report import EvalReport, EvalResult
from oncora.features.compound import FeatureStore
from oncora.models.target.calibration import apply_calibrator, fit_calibrator, ranking_score

_W: dict = {}  # per-process worker state


@dataclass
class TargetTask:
    target: str
    train_keys: list[str]
    train_y: np.ndarray
    val_keys: list[str]
    val_y: np.ndarray
    test_keys: list[str]


def _init_worker(store, sample_keys: list[str], run_dir: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["store"] = FeatureStore.load(Path(store)) if isinstance(store, (str, Path)) else store
    _W["sample"] = _W["store"].matrix(sample_keys) if sample_keys else None
    _W["run_dir"] = Path(run_dir)


def _train_one(task: TargetTask) -> dict:
    store: FeatureStore = _W["store"]
    n_bits = store.cfg.n_bits
    out = {"target": task.target, "n_train": len(task.train_y), "n_train_pos": int(task.train_y.sum()),
           "n_val": len(task.val_y), "n_val_pos": int(task.val_y.sum()), "n_test": len(task.test_keys)}
    if len(np.unique(task.train_y)) < 2:
        return out | {"skipped": "single class in train"}

    x_tr = store.matrix(task.train_keys)
    clf = lgb.LGBMClassifier(**TARGET_LGBM, random_state=SEED, n_jobs=1)
    if len(task.val_y):
        x_val = store.matrix(task.val_keys)
        clf.fit(x_tr, task.train_y, eval_X=(x_val,), eval_y=(task.val_y,), eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(TARGET_EARLY_STOPPING, verbose=False)])
        cal = fit_calibrator(clf.predict_proba(x_val)[:, 1], task.val_y)
    else:
        clf.set_params(n_estimators=200).fit(x_tr, task.train_y)
        cal = {"method": "identity"}
    best = int(clf.best_iteration_ or clf.n_estimators)
    clf.booster_.save_model(str(_W["run_dir"] / "lgbm" / f"{task.target}.txt"), num_iteration=best)

    x_te = store.matrix(task.test_keys) if task.test_keys else np.zeros((0, store.cfg.n_features), np.float32)
    raw_te = clf.predict_proba(x_te, num_iteration=best)[:, 1] if len(x_te) else np.zeros(0)
    lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=300).fit(x_tr[:, :n_bits], task.train_y)
    lr_te = lr.predict_proba(x_te[:, :n_bits])[:, 1] if len(x_te) else np.zeros(0)
    sample = _W.get("sample")
    sample_scores = None
    if sample is not None:
        raw_s = clf.predict_proba(sample, num_iteration=best)[:, 1]
        sample_scores = ranking_score(apply_calibrator(cal, raw_s), raw_s).astype(np.float64)
    return out | {
        "best_iteration": best, "calibration": cal,
        "raw_test": raw_te.astype(np.float32), "cal_test": apply_calibrator(cal, raw_te).astype(np.float32),
        "lr_test": lr_te.astype(np.float32), "sample_scores": sample_scores,
    }


def build_tasks(df: pl.DataFrame, part: Partition) -> tuple[list[TargetTask], dict[str, pl.DataFrame]]:
    df = df.with_columns(pl.Series("_split", np.select([part.train, part.val, part.test], [0, 1, 2], -1)))
    tasks, tests = [], {}
    for (target,), g in sorted(df.partition_by("target_chembl_id", as_dict=True).items()):
        tr, va, te = (g.filter(pl.col("_split") == s) for s in (0, 1, 2))
        tests[target] = te
        tasks.append(TargetTask(target, tr["inchi_key"].to_list(), tr["label"].to_numpy(),
                                va["inchi_key"].to_list(), va["label"].to_numpy(), te["inchi_key"].to_list()))
    tasks.sort(key=lambda t: -len(t.train_y))   # largest first for load balancing
    return tasks, tests


def run_tasks(tasks, store, sample_keys, run_dir: Path, n_jobs: int) -> list[dict]:
    (run_dir / "lgbm").mkdir(parents=True, exist_ok=True)
    if n_jobs <= 1:
        _init_worker(store, sample_keys, str(run_dir))
        return [_train_one(t) for t in tasks]
    with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(store, sample_keys, str(run_dir))) as ex:
        return list(ex.map(_train_one, tasks, chunksize=2))


def evaluate(results, tests, fit_df, targets, sample_keys, positives, *, with_ci: bool) -> tuple[EvalReport, pl.DataFrame]:
    done = [r for r in results if "raw_test" in r]
    test = pl.concat([tests[r["target"]] for r in done])
    raw = np.concatenate([r["raw_test"] for r in done])
    cal = np.concatenate([r["cal_test"] for r in done])
    lr = np.concatenate([r["lr_test"] for r in done])
    t_index = {t: i for i, t in enumerate(targets)}
    matrix = np.zeros((len(sample_keys), len(targets)), dtype=np.float64)
    for r in done:
        if r["sample_scores"] is not None:
            matrix[:, t_index[r["target"]]] = r["sample_scores"]

    m_model, ci_model = evaluate_target(test, raw, score_matrix=matrix, positives=positives, with_ci=with_ci)
    m_model["pooled_pr_auc"] = float(average_precision_score(test["label"].to_numpy(), cal))
    y = test["label"].to_numpy()
    m_model["brier_calibrated"] = float(brier_score_loss(y, cal))
    m_model["brier_raw"] = float(brier_score_loss(y, raw))
    m_lr, ci_lr = evaluate_target(test, lr, with_ci=with_ci)
    test_targets = test.select("target_chembl_id")
    prior = PriorBaseline().fit(fit_df).predict(test_targets)
    pop = np.asarray(PopularityRanker().fit(fit_df).score_matrix(len(sample_keys), targets))
    m_prior, _ = evaluate_target(test, prior, score_matrix=pop, positives=positives)

    per_target = pl.DataFrame([{k: r.get(k) for k in ("target", "n_train", "n_train_pos", "n_val", "n_val_pos",
                                                       "n_test", "best_iteration", "skipped")} |
                               {"calibration": (r.get("calibration") or {}).get("method")} for r in results])
    report = EvalReport(
        "Compound → target model", "target",
        [EvalResult("model", "compound", m_model, ci_model), EvalResult("logreg_ecfp", "compound", m_lr, ci_lr),
         EvalResult("prior_popularity", "compound", m_prior)],
        primary=["pr_auc_macro", "pr_auc_weighted"],
        comparisons=["logreg_ecfp"],
    )
    return report, per_target


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def train(n_jobs: int | None = None, max_targets: int | None = None, with_ci: bool = True,
          table: pl.DataFrame | None = None, store=None, models_dir: Path = MODELS / "target",
          report_dir: Path | None = REPORTS_EVAL, metadata: pl.DataFrame | None = None,
          log_mlflow: bool = True) -> Path:
    t0 = time.time()
    df = table if table is not None else target_training_table()
    if max_targets:
        keep = df["target_chembl_id"].unique().sort().head(max_targets)
        df = df.filter(pl.col("target_chembl_id").is_in(keep.implode()))
    part = make_partition(df, "compound")
    targets = sorted(df["target_chembl_id"].unique().to_list())
    test_df = df.filter(part.test)
    sample_keys, positives = topk_sample(test_df, targets, TOPK_MAX_COMPOUNDS)
    tasks, tests = build_tasks(df, part)

    run_dir = models_dir / _run_id()
    store = store if store is not None else FEATURES
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    results = run_tasks(tasks, store, sample_keys, run_dir, n_jobs)
    fit_df = df.filter(part.fit)
    report, per_target = evaluate(results, tests, fit_df, targets, sample_keys, positives, with_ci=with_ci)

    fs = store if isinstance(store, FeatureStore) else FeatureStore.load(Path(store))
    fit_keys = fit_df["inchi_key"].unique().sort().to_list()
    TrainingIndex(fit_keys, fs.packed_rows(fit_keys)).save(run_dir / "train_index.npz")
    (run_dir / "calibrators.json").write_text(json.dumps({r["target"]: r["calibration"] for r in results if "calibration" in r}))
    meta = metadata if metadata is not None else target_metadata()
    per_target.rename({"target": "target_chembl_id"}).join(meta, on="target_chembl_id", how="left").write_parquet(run_dir / "targets.parquet")

    skipped = per_target.filter(pl.col("skipped").is_not_null()).height
    report.notes += [
        f"{df.height:,} collapsed pairs, {len(targets):,} targets ({skipped} skipped: single class in train).",
        "Per-target ROC/PR-AUC use raw scores (calibration is monotone but isotonic steps create ties); "
        "pooled PR-AUC and top-k use calibrated probabilities, which are comparable across targets.",
        f"Brier score on test: raw {report.results[0].metrics['brier_raw']:.4f} → calibrated "
        f"{report.results[0].metrics['brier_calibrated']:.4f}.",
        f"Top-k over {len(sample_keys):,} sampled test compounds with ≥ 1 known positive, against all targets.",
        "CIs: 95% percentile bootstrap over test compounds (50 resamples). In a resample, a small "
        "target whose test set becomes single-class drops out of the macro average, which nudges the "
        "bootstrap distribution slightly upward; read the point estimate as the reference.",
        "Calibration: identity below 10 validation compounds per class, regularized Platt (pulled "
        "toward identity) below 100, isotonic above. Rankings break isotonic ties by raw score.",
        "Sampled negatives are assumed inactive (see target_pairs provenance); measured precision is "
        "therefore a lower bound if some sampled negatives are untested actives.",
    ]
    report.manifests = {"compound": part.manifest.to_dict()}
    card = model_card(run_dir, df, part, report, fs, per_target, time.time() - t0)
    (run_dir / "model_card.json").write_text(json.dumps(card, indent=1, default=float))
    report.save(run_dir, "eval")
    if report_dir is not None:
        report.save(report_dir, "target_model")
    if log_mlflow:
        _log_mlflow(card, report, run_dir)
    return run_dir


def target_metadata() -> pl.DataFrame:
    tp = pl.read_parquet(FEATURES.parent / "target_pairs.parquet",
                         columns=["target_chembl_id", "target_name", "target_gene_symbol", "uniprot_accession"])
    return tp.unique("target_chembl_id").sort("target_chembl_id")


def model_card(run_dir, df, part, report, store, per_target, seconds) -> dict:
    tp_path = FEATURES.parent / "target_pairs.parquet"
    tp_prov = read_provenance(tp_path) if tp_path.exists() else {}
    return {
        "model": "compound_to_target", "run_id": run_dir.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "per-target LightGBM binary classifier + per-target calibration",
        "hyperparameters": TARGET_LGBM | {"early_stopping_rounds": TARGET_EARLY_STOPPING, "seed": SEED},
        "features": {"compound": store.cfg.to_dict()},
        "training_data": {"table": "data/processed/target_pairs.parquet (collapsed per standardized parent)",
                          "rows": df.height, "targets": df["target_chembl_id"].n_unique(),
                          "label_policy": tp_prov.get("config", {}), "sources": tp_prov.get("sources", [])},
        "split": part.manifest.to_dict() | {"group_key": "standardized connectivity block (D-01)",
                                            "rows": {"train": int(part.train.sum()), "val": int(part.val.sum()),
                                                     "test": int(part.test.sum())}},
        "calibration_methods": dict(per_target.group_by("calibration").len().iter_rows()),
        "metrics": {r.name: r.metrics | {"ci": r.ci} for r in report.results},
        "accepted": report.accepted(),
        "shipped_model": "evaluated model (fit on train; val for early stopping/calibration)",
        "training_seconds": round(seconds),
        "intended_use": "Computational research predictions, not validated clinical conclusions.",
    }


def _log_mlflow(card: dict, report: EvalReport, run_dir: Path) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("compound_to_target")
    with mlflow.start_run(run_name=card["run_id"]):
        mlflow.log_params({f"lgbm_{k}": v for k, v in card["hyperparameters"].items()})
        mlflow.log_metrics({k: v for k, v in report.results[0].metrics.items() if isinstance(v, float) and np.isfinite(v)})
        mlflow.log_artifact(str(run_dir / "model_card.json"))
        mlflow.log_artifact(str(run_dir / "eval.md"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--max-targets", type=int, default=None, help="train on the first K targets (smoke runs)")
    ap.add_argument("--no-ci", action="store_true")
    args = ap.parse_args()
    run_dir = train(args.jobs, args.max_targets, with_ci=not args.no_ci,
                    report_dir=None if args.max_targets else REPORTS_EVAL)
    print(f"Saved {run_dir}")
    print((run_dir / "eval.md").read_text().split("## Notes")[0])


if __name__ == "__main__":
    main()
