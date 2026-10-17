"""Compound × cell line → sensitivity training (spec 06, Req 1, 2, 5; FR-9, FR-10, FR-12).

    uv run python -m oncora.models.sensitivity.train

Models, all fit on the training partition with early stopping on validation:
- **regressor:** LightGBM on rank-transformed DSS3 (D-04). Its prediction ranks cell lines and
  compounds;
- **classifier:** LightGBM for "line is in the compound's top 10%" (D-05), whose probability is
  the selectivity score;
- **compound-only ablation:** the same learner on compound features alone, fit to each
  compound's mean response. It can only predict potency, and comparing against it shows what
  the cell-line features add.

Both split strategies (compound and scaffold) are evaluated. The compound-split model is saved as
the artifact. The mixed-scale sensitivity analysis (D-03) re-runs the compound split with the 20
mixed-scale compounds included at ×0.01 scale.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from oncora.chem.similarity import TrainingIndex
from oncora.config import (
    DATA_INTERIM,
    GLOBAL_SENSITIVE_QUANTILE,
    MLFLOW_URI,
    MODELS,
    REPORTS_EVAL,
    ROOT,
    SEED,
    SENS_EARLY_STOPPING,
    SENS_LGBM,
    SENSITIVE_TOP_FRAC,
)
from oncora.data_prep.modeling import FEATURES, sensitivity_training_table
from oncora.evaluation.baselines import CellLineMeanBaseline, GlobalMeanBaseline
from oncora.evaluation.harness import evaluate_sensitivity
from oncora.evaluation.labels import RankTransform, global_threshold, within_compound_top
from oncora.evaluation.protocol import STRATEGIES, make_partition
from oncora.evaluation.report import EvalReport, EvalResult
from oncora.features.cellline import fit_cellline_features, transform
from oncora.features.compound import FeatureStore
from oncora.models.sensitivity.design import EntityMatrix, assert_structure_only, build_design_matrix

PRIMARY = ["spearman_per_cell_line", "spearman_per_compound", "pr_auc_within_top10"]


def _early(n):
    return [lgb.early_stopping(n, verbose=False)]


def fit_strategy(df: pl.DataFrame, strategy: str, store: FeatureStore, *, n_jobs: int, with_ci: bool,
                 ablation: bool = True, lgbm: dict = SENS_LGBM) -> dict:
    t0 = time.time()
    part = make_partition(df, strategy)
    train, val, test, fit = (df.filter(m) for m in (part.train, part.val, part.test, part.fit))

    comp_keys = df["inchi_key"].unique().sort().to_list()
    compounds = EntityMatrix(comp_keys, store.matrix(comp_keys), store.cfg.feature_names)
    cell_cfg = fit_cellline_features(train["model_id"].unique().sort().to_list())
    lines = df["model_id"].unique().sort().to_list()
    cells = EntityMatrix(lines, transform(lines, cell_cfg), cell_cfg.feature_names)
    assert_structure_only(compounds.names + cells.names, store.cfg, cell_cfg)

    rt = RankTransform().fit(train["dss3"].to_numpy())
    thr = global_threshold(train["dss3"].to_numpy(), GLOBAL_SENSITIVE_QUANTILE)
    y_tr, y_val = rt.transform(train["dss3"].to_numpy()), rt.transform(val["dss3"].to_numpy())
    lab_tr = within_compound_top(train, SENSITIVE_TOP_FRAC, compound="group").to_numpy().astype(int)
    lab_val = within_compound_top(val, SENSITIVE_TOP_FRAC, compound="group").to_numpy().astype(int)

    x_tr, x_val = build_design_matrix(train, compounds, cells), build_design_matrix(val, compounds, cells)
    reg = lgb.LGBMRegressor(**lgbm, random_state=SEED, n_jobs=n_jobs)
    reg.fit(x_tr, y_tr, eval_X=(x_val,), eval_y=(y_val,), callbacks=_early(SENS_EARLY_STOPPING))
    clf = lgb.LGBMClassifier(**lgbm, random_state=SEED, n_jobs=n_jobs)
    clf.fit(x_tr, lab_tr, eval_X=(x_val,), eval_y=(lab_val,), eval_metric="binary_logloss",
            callbacks=_early(SENS_EARLY_STOPPING))
    del x_tr
    x_te = build_design_matrix(test, compounds, cells)
    pred_te, prob_te = reg.predict(x_te), clf.predict_proba(x_te)[:, 1]
    del x_te

    results = []
    m, ci = evaluate_sensitivity(test, pred_te, rt, thr, SENSITIVE_TOP_FRAC, with_ci=with_ci,
                                 label_scores={"within_top10": prob_te})
    results.append(EvalResult("model", strategy, m, ci))

    abl = None
    if ablation:
        def means(frame):
            return frame.with_columns(pl.Series("y", rt.transform(frame["dss3"].to_numpy()))) \
                        .group_by("inchi_key").agg(pl.col("y").mean()).sort("inchi_key")
        cm_tr, cm_val = means(train), means(val)
        abl = lgb.LGBMRegressor(**(lgbm | {"num_leaves": 15, "min_child_samples": 5}), random_state=SEED, n_jobs=n_jobs)
        abl.fit(compounds.values[compounds.index(cm_tr["inchi_key"])], cm_tr["y"].to_numpy(),
                eval_X=(compounds.values[compounds.index(cm_val["inchi_key"])],), eval_y=(cm_val["y"].to_numpy(),),
                callbacks=_early(SENS_EARLY_STOPPING))
        abl_te = abl.predict(compounds.values[compounds.index(test["inchi_key"])])
        m, ci = evaluate_sensitivity(test, abl_te, rt, thr, SENSITIVE_TOP_FRAC, with_ci=with_ci)
        results.append(EvalResult("compound_only", strategy, m, ci))

    y_fit = rt.transform(fit["dss3"].to_numpy())
    for name, b in (("cell_line_mean", CellLineMeanBaseline()), ("global_mean", GlobalMeanBaseline())):
        m, _ = evaluate_sensitivity(test, b.fit(fit, y_fit).predict(test), rt, thr, SENSITIVE_TOP_FRAC)
        results.append(EvalResult(name, strategy, m))

    return {"strategy": strategy, "results": results, "partition": part, "reg": reg, "clf": clf, "ablation": abl,
            "rt": rt, "cell_cfg": cell_cfg, "threshold": thr, "fit_keys": fit["inchi_key"].unique().sort().to_list(),
            "rows": {"train": train.height, "val": val.height, "test": test.height},
            "best_iterations": {"regressor": int(reg.best_iteration_ or 0), "classifier": int(clf.best_iteration_ or 0)},
            "seconds": round(time.time() - t0)}


def selectivity_verdict(report: EvalReport, split: str = "compound") -> tuple[bool, str]:
    """Does the model rank cell lines for a *specific* compound better than the line-average ranking?

    Two selectivity metrics, which can disagree:
    - per-compound Spearman ρ: ordering the whole list of lines for one compound;
    - within-top-10% PR-AUC: picking out that compound's most responsive lines.
    The second is the product question; the first is stricter. A model beats the baseline on a
    metric only if its 95% CI lower bound exceeds the baseline's point estimate.
    """
    model, base = report._get("model", split), report._get("cell_line_mean", split)
    abl = report._get("compound_only", split)

    def beats(key):
        lo = model.ci.get(key, (float("nan"),))[0]
        return bool(np.isfinite(lo) and lo > base.metrics[key]), lo

    rho_beats, rho_lo = beats("spearman_per_compound")
    pr_beats, pr_lo = beats("pr_auc_within_top10")
    learned = rho_beats or pr_beats
    parts = [
        f"Selectivity ({split} split): ranking a compound's top-decile lines, PR-AUC "
        f"{model.metrics['pr_auc_within_top10']:.3f} (95% CI lower {pr_lo:.3f}) vs cell-line-mean "
        f"{base.metrics['pr_auc_within_top10']:.3f}" + (" — **better**" if pr_beats else " — not better") + ". "
        f"Ordering the full list of lines, per-compound ρ {model.metrics['spearman_per_compound']:.3f} "
        f"(CI lower {rho_lo:.3f}) vs {base.metrics['spearman_per_compound']:.3f}"
        + (" — **better**." if rho_beats else " — not better.")
    ]
    if abl is not None:
        parts.append(
            f"The compound-only ablation scores {abl.metrics['pr_auc_within_top10']:.3f} on that PR-AUC, at the "
            f"{abl.metrics['prevalence_within_top10']:.3f} prevalence floor, because it cannot vary across lines"
            + (": the model's margin therefore comes from the cell-line expression features."
               if pr_beats else "."))
    parts.append(
        "**The model identifies which lines respond most to a given compound better than the line-average "
        "ranking does, but does not order the full list better.** Read the top of the list as the usable "
        "signal, not the tail." if (pr_beats and not rho_beats) else
        ("**Cell-line rankings carry compound-specific signal.**" if learned else
         "**Cell-line rankings mostly reflect compound potency plus each line's general sensitivity, not "
         "learned selectivity.**"))
    return learned, " ".join(parts)


def save_artifact(run_dir: Path, fitted: dict, store: FeatureStore) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fitted["reg"].booster_.save_model(str(run_dir / "regressor.txt"), num_iteration=fitted["reg"].best_iteration_)
    fitted["clf"].booster_.save_model(str(run_dir / "classifier.txt"), num_iteration=fitted["clf"].best_iteration_)
    np.save(run_dir / "rank_transform.npy", fitted["rt"].sorted_train)
    (run_dir / "cellline_config.json").write_text(json.dumps(fitted["cell_cfg"].to_dict()))
    # Every CCLE line (not only the trained ones) can be scored at inference (Req 3.1).
    all_lines = pl.read_parquet(DATA_INTERIM / "ccle_expression.parquet", columns=["model_id"])["model_id"].sort().to_list()
    mat = transform(all_lines, fitted["cell_cfg"])
    pl.DataFrame({"model_id": all_lines}).hstack(
        pl.DataFrame(mat, schema=fitted["cell_cfg"].feature_names)).write_parquet(run_dir / "cellline_matrix.parquet")
    TrainingIndex(fitted["fit_keys"], store.packed_rows(fitted["fit_keys"])).save(run_dir / "train_index.npz")


def train(n_jobs: int | None = None, with_ci: bool = True, sensitivity_analysis: bool = True,
          report_dir: Path | None = REPORTS_EVAL, models_dir: Path = MODELS / "sensitivity",
          log_mlflow: bool = True) -> Path:
    t0 = time.time()
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    store = FeatureStore.load(FEATURES)
    df = sensitivity_training_table()
    fitted = {s: fit_strategy(df, s, store, n_jobs=n_jobs, with_ci=with_ci) for s in STRATEGIES}
    results = [r for f in fitted.values() for r in f["results"]]
    report = EvalReport("Compound × cell line → sensitivity model", "sensitivity", results, PRIMARY,
                        manifests={s: f["partition"].manifest.to_dict() for s, f in fitted.items()},
                        comparisons=["compound_only"])
    learned, verdict = selectivity_verdict(report)
    report.notes += [
        verdict,
        f"Headline set (D-03): {df.height:,} rows, {df['group'].n_unique()} compounds, {df['model_id'].n_unique()} cell lines.",
        "Regression target: DSS3 percentile under the training distribution (D-04). The within_top10 "
        "AUCs use the dedicated classifier's probability; all other metrics use the regressor.",
        "per-cell-line ρ ranks compounds within a line (potency); per-compound ρ ranks lines within a compound (selectivity).",
        "compound_only cannot vary across lines, so its per-compound ρ is undefined: it isolates potency.",
        "CIs: 95% percentile bootstrap over test compounds (200 resamples).",
    ]
    mixed = None
    if sensitivity_analysis:
        mdf = sensitivity_training_table(include_mixed_scale=True)
        mixed = fit_strategy(mdf, "compound", store, n_jobs=n_jobs, with_ci=False, ablation=False)
        mm = mixed["results"][0].metrics
        report.notes.append(
            f"Mixed-scale sensitivity analysis (D-03; 20 compounds added with DSS3 ≥ 1 × 0.01, {mdf.height:,} rows, "
            f"compound split, no CI): per-cell-line ρ {mm['spearman_per_cell_line']:.3f}, per-compound ρ "
            f"{mm['spearman_per_compound']:.3f}, within_top10 PR-AUC {mm['pr_auc_within_top10']:.3f}.")

    run_dir = models_dir / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    main_fit = fitted["compound"]
    save_artifact(run_dir, main_fit, store)
    card = {
        "model": "compound_cellline_to_sensitivity", "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "LightGBM regressor (rank-transformed DSS3) + LightGBM classifier (within-compound top 10%)",
        "hyperparameters": SENS_LGBM | {"early_stopping_rounds": SENS_EARLY_STOPPING, "seed": SEED},
        "best_iterations": main_fit["best_iterations"],
        "features": {"compound": store.cfg.to_dict(),
                     "cell_line": {"strategy": main_fit["cell_cfg"].strategy, "n_genes": len(main_fit["cell_cfg"].genes)}},
        "training_data": {"table": "data/processed/sensitivity.parquet (headline: ~mixed_scale)", "rows": df.height,
                          "compounds": df["group"].n_unique(), "cell_lines": df["model_id"].n_unique(),
                          "mixed_scale_handling": "excluded (D-03); rescaled analysis in notes"},
        "split": main_fit["partition"].manifest.to_dict() | {"rows": main_fit["rows"]},
        "labels": {"transform": "rank under training DSS3 (D-04)",
                   "sensitive": f"top {SENSITIVE_TOP_FRAC:.0%} of a compound's lines and DSS3 > 0 (D-05)",
                   "global_threshold_p90": main_fit["threshold"]},
        "metrics": {r.name + "/" + r.split: r.metrics | {"ci": r.ci} for r in results},
        "accepted": report.accepted(),
        "learned_selectivity": learned,
        "shipped_model": "compound-split model (fit on train; val for early stopping)",
        "training_seconds": round(time.time() - t0),
        "intended_use": "Computational research predictions, not validated clinical conclusions.",
    }
    if mixed:
        card["mixed_scale_analysis"] = mixed["results"][0].metrics
    (run_dir / "model_card.json").write_text(json.dumps(card, indent=1, default=float))
    report.save(run_dir, "eval")
    if report_dir is not None:
        report.save(report_dir, "sensitivity_model")
    if log_mlflow:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow
        (ROOT / "mlruns").mkdir(exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("compound_cellline_to_sensitivity")
        with mlflow.start_run(run_name=run_dir.name):
            mlflow.log_params({f"lgbm_{k}": v for k, v in card["hyperparameters"].items()})
            mlflow.log_metrics({k: v for k, v in results[0].metrics.items() if isinstance(v, float) and np.isfinite(v)})
            mlflow.log_artifact(str(run_dir / "model_card.json"))
            mlflow.log_artifact(str(run_dir / "eval.md"))
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-ci", action="store_true")
    ap.add_argument("--no-sensitivity-analysis", action="store_true")
    args = ap.parse_args()
    run_dir = train(args.jobs, with_ci=not args.no_ci, sensitivity_analysis=not args.no_sensitivity_analysis)
    print(f"Saved {run_dir}")
    print((run_dir / "eval.md").read_text().split("## Split manifests")[0])


if __name__ == "__main__":
    main()
