"""Shared scoring of test-partition predictions, so baselines and models are judged identically."""

import numpy as np
import polars as pl

from oncora.config import BOOTSTRAP_N, SEED
from oncora.evaluation.labels import RankTransform, within_compound_top
from oncora.evaluation.metrics import (
    bootstrap_ci,
    group_index,
    sensitivity_metrics,
    target_metrics,
    top_k_hit_rate,
)

SENSITIVITY_CI_METRICS = ("spearman_overall", "spearman_per_cell_line", "spearman_per_compound",
                          "pr_auc_within_top10", "roc_auc_within_top10")
TARGET_CI_METRICS = ("pr_auc_macro", "roc_auc_macro", "pr_auc_weighted")


def sensitivity_labels(test: pl.DataFrame, global_thr: float, top_frac: float) -> dict[str, np.ndarray]:
    return {
        "within_top10": within_compound_top(test, top_frac, compound="group").to_numpy(),
        "global_p90": (test["dss3"] >= global_thr).to_numpy(),
    }


def evaluate_sensitivity(test: pl.DataFrame, pred_t: np.ndarray, rt: RankTransform, global_thr: float,
                         top_frac: float, *, with_ci: bool = False, n_boot: int = BOOTSTRAP_N):
    y_t = rt.transform(test["dss3"].to_numpy())
    comp, line = test["group"].to_numpy(), test["model_id"].to_numpy()
    labels = sensitivity_labels(test, global_thr, top_frac)
    metrics = sensitivity_metrics(y_t, pred_t, comp, line, labels)
    ci = {}
    if with_ci:
        def boot(idx, key):
            return sensitivity_metrics(y_t[idx], pred_t[idx], comp[idx], line[idx],
                                       {k: v[idx] for k, v in labels.items()})[key]
        index = group_index(comp)
        ci = {k: bootstrap_ci(lambda idx, k=k: boot(idx, k), index, n=n_boot, seed=SEED) for k in SENSITIVITY_CI_METRICS}
    return metrics, ci


def evaluate_target(test: pl.DataFrame, scores: np.ndarray, *, score_matrix: np.ndarray | None = None,
                    positives: list[set[int]] | None = None, with_ci: bool = False, n_boot: int = 50):
    y, targets = test["label"].to_numpy(), test["target_chembl_id"].to_numpy()
    metrics = target_metrics(y, scores, targets)
    if score_matrix is not None:
        metrics |= top_k_hit_rate(score_matrix, positives)
    ci = {}
    if with_ci:
        index = group_index(test["group"].to_numpy())
        for k in TARGET_CI_METRICS:
            ci[k] = bootstrap_ci(lambda idx, k=k: target_metrics(y[idx], scores[idx], targets[idx])[k],
                                 index, n=n_boot, seed=SEED)
    return metrics, ci


def topk_sample(test: pl.DataFrame, targets: list[str], max_compounds: int, seed: int = SEED):
    """Sample test compounds that have >= 1 known positive; return (inchi_keys, positive index sets)."""
    t_index = {t: i for i, t in enumerate(targets)}
    pos = (test.filter(pl.col("label") == 1).group_by("group")
           .agg(pl.col("inchi_key").first(), pl.col("target_chembl_id")).sort("group"))
    if pos.height > max_compounds:
        pos = pos.sample(max_compounds, seed=seed).sort("group")
    keys = pos["inchi_key"].to_list()
    sets = [{t_index[t] for t in ts if t in t_index} for ts in pos["target_chembl_id"].to_list()]
    return keys, sets
