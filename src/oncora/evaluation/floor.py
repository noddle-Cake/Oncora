"""Baseline floor: score the trivial baselines under the shared protocol (spec 04, task 4).

    uv run python -m oncora.evaluation.floor

These are the numbers every model must beat. Baselines are fit on train ∪ val (all non-test
rows), which is slightly generous to them: a conservative choice.
"""

import numpy as np

from oncora.config import GLOBAL_SENSITIVE_QUANTILE, REPORTS_EVAL, SENSITIVE_TOP_FRAC, TOPK_MAX_COMPOUNDS
from oncora.data_prep.modeling import sensitivity_training_table, target_training_table
from oncora.evaluation.baselines import CellLineMeanBaseline, GlobalMeanBaseline, PopularityRanker, PriorBaseline
from oncora.evaluation.harness import evaluate_sensitivity, evaluate_target, topk_sample
from oncora.evaluation.labels import RankTransform, global_threshold
from oncora.evaluation.protocol import STRATEGIES, make_partition
from oncora.evaluation.report import EvalReport, EvalResult


def sensitivity_floor() -> EvalReport:
    df = sensitivity_training_table()
    results, manifests = [], {}
    for strategy in STRATEGIES:
        part = make_partition(df, strategy)
        fit, test = df.filter(part.fit), df.filter(part.test)
        rt = RankTransform().fit(fit["dss3"].to_numpy())
        y_fit = rt.transform(fit["dss3"].to_numpy())
        thr = global_threshold(fit["dss3"].to_numpy(), GLOBAL_SENSITIVE_QUANTILE)
        for name, b in (("global_mean", GlobalMeanBaseline()), ("cell_line_mean", CellLineMeanBaseline())):
            m, _ = evaluate_sensitivity(test, b.fit(fit, y_fit).predict(test), rt, thr, SENSITIVE_TOP_FRAC)
            results.append(EvalResult(name, strategy, m))
        manifests[strategy] = part.manifest.to_dict()
    return EvalReport(
        "Sensitivity baselines (floor)", "sensitivity", results,
        primary=["spearman_per_cell_line", "spearman_per_compound", "pr_auc_within_top10"],
        manifests=manifests,
        notes=[f"Headline set (D-03): {df.height:,} rows, {df['group'].n_unique()} compounds.",
               "global_mean predicts a constant, so every ρ is undefined (—).",
               "cell_line_mean is constant within a cell line across compounds, so per-cell-line ρ is "
               "undefined: it carries no compound information. Its per-compound ρ is the selectivity "
               "bar a model must clear."],
    )


def target_floor() -> EvalReport:
    df = target_training_table()
    part = make_partition(df, "compound")
    fit, test = df.filter(part.fit), df.filter(part.test)
    targets = sorted(df["target_chembl_id"].unique().to_list())
    prior = PriorBaseline().fit(fit)
    keys, pos = topk_sample(test, targets, TOPK_MAX_COMPOUNDS)
    pop = PopularityRanker().fit(fit).score_matrix(len(keys), targets)
    m, _ = evaluate_target(test, prior.predict(test), score_matrix=np.asarray(pop), positives=pos)
    return EvalReport(
        "Target baselines (floor)", "target", [EvalResult("prior_popularity", "compound", m)],
        primary=["pr_auc_macro", "pr_auc_weighted"],
        manifests={"compound": part.manifest.to_dict()},
        notes=[f"{df.height:,} collapsed pairs, {len(targets):,} targets; test {test.height:,} rows.",
               "The prior is constant within a target, so per-target ROC-AUC = 0.5 and per-target "
               "PR-AUC = that target's test prevalence (prevalence_macro).",
               f"Top-k uses popularity ranking over {len(keys):,} sampled test compounds with ≥ 1 known positive."],
    )


def main() -> None:
    for stem, report in (("baselines_sensitivity", sensitivity_floor()), ("baselines_target", target_floor())):
        report.save(REPORTS_EVAL, stem)
        print(f"Wrote {REPORTS_EVAL / stem}.md")


if __name__ == "__main__":
    main()
