"""Metrics chosen for imbalance and skew (spec 04, Req 5; AC-4, AC-5).

Target prediction is heavily imbalanced, so PR-AUC is primary: it measures precision among the
top predictions, which is what a chemist acts on. ROC-AUC is reported but looks flattering when
negatives dominate. Top-k asks whether a known target appears among the first k suggestions.

Sensitivity is ranking-shaped and skewed, so Spearman ρ (rank correlation) is the headline.
- overall ρ: all test rows pooled;
- per-cell-line ρ: rank compounds within a line, then average. This measures potency, which is
  learned from structure;
- per-compound ρ: rank lines within a compound, then average. This measures selectivity.

Confidence intervals resample *compounds* (the independent unit), never rows.
"""

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe(fn, y, s):
    return float(fn(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def per_group_spearman(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, min_size: int = 5) -> np.ndarray:
    df = pl.DataFrame({"g": groups, "y": y, "p": pred})
    out = (
        df.group_by("g").agg(pl.len().alias("n"), pl.corr("y", "p", method="spearman").alias("rho"))
        .filter((pl.col("n") >= min_size) & pl.col("rho").is_not_nan() & pl.col("rho").is_not_null())
        .sort("g")
    )
    return out["rho"].to_numpy()


def per_target_auc(y: np.ndarray, score: np.ndarray, targets: np.ndarray) -> pl.DataFrame:
    df = pl.DataFrame({"t": targets, "y": y, "s": score})
    rows = []
    for (t,), g in df.group_by(["t"], maintain_order=False):
        yy, ss = g["y"].to_numpy(), g["s"].to_numpy()
        rows.append({"target": t, "n": len(yy), "n_pos": int(yy.sum()),
                     "pr_auc": _safe(average_precision_score, yy, ss),
                     "roc_auc": _safe(roc_auc_score, yy, ss),
                     "prevalence": float(yy.mean())})
    return pl.DataFrame(rows).sort("target")


def target_metrics(y, score, targets) -> dict:
    per_t = per_target_auc(np.asarray(y), np.asarray(score), np.asarray(targets)).drop_nans()
    w = per_t["n_pos"].to_numpy()
    return {
        "n_targets_evaluated": per_t.height,
        "pr_auc_macro": float(per_t["pr_auc"].mean()),
        "pr_auc_weighted": float(np.average(per_t["pr_auc"], weights=w)),
        "roc_auc_macro": float(per_t["roc_auc"].mean()),
        "roc_auc_weighted": float(np.average(per_t["roc_auc"], weights=w)),
        "prevalence_macro": float(per_t["prevalence"].mean()),
        "pooled_pr_auc": _safe(average_precision_score, np.asarray(y), np.asarray(score)),
    }


def top_k_hit_rate(score_matrix: np.ndarray, positives: list[set[int]], ks=(1, 5, 10)) -> dict:
    """Share of compounds whose top-k predicted targets include at least one known positive."""
    keep = [i for i, p in enumerate(positives) if p]
    if not keep:
        return {f"top{k}": float("nan") for k in ks}
    order = np.argsort(-score_matrix[keep], axis=1, kind="stable")
    out = {}
    for k in ks:
        top = order[:, :k]
        out[f"top{k}"] = float(np.mean([bool(set(top[j]) & positives[i]) for j, i in enumerate(keep)]))
    return out


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def sensitivity_metrics(y_t: np.ndarray, pred_t: np.ndarray, compounds: np.ndarray, lines: np.ndarray,
                        labels: dict[str, np.ndarray] | None = None, min_group: int = 5,
                        label_scores: dict[str, np.ndarray] | None = None) -> dict:
    """y_t / pred_t are on the transformed scale; labels maps label name -> boolean array.
    Per-group ρ is undefined for a constant prediction within a group; such groups are skipped."""
    out = {
        "spearman_overall": float(spearmanr(y_t, pred_t).statistic) if np.ptp(pred_t) > 0 else float("nan"),
        "spearman_per_cell_line": _mean(per_group_spearman(y_t, pred_t, lines, min_group)),
        "spearman_per_compound": _mean(per_group_spearman(y_t, pred_t, compounds, min_group)),
        "rmse_transformed": float(np.sqrt(np.mean((y_t - pred_t) ** 2))),
    }
    for name, lab in (labels or {}).items():
        score = (label_scores or {}).get(name, pred_t)   # e.g. a dedicated classifier's probability
        out[f"roc_auc_{name}"] = _safe(roc_auc_score, lab, score)
        out[f"pr_auc_{name}"] = _safe(average_precision_score, lab, score)
        out[f"prevalence_{name}"] = float(np.mean(lab))
    return out


def bootstrap_ci(metric_fn, n_groups_index: dict, *, n: int = 200, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of metric_fn(row_indices) over bootstrap resamples of groups.

    n_groups_index maps each group to the array of its row indices.
    """
    rng = np.random.default_rng(seed)
    groups = list(n_groups_index.values())
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        stats.append(metric_fn(np.concatenate([groups[i] for i in pick])))
    stats = np.asarray(stats, dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def group_index(groups: np.ndarray) -> dict:
    df = pl.DataFrame({"g": groups}).with_row_index("i")
    return {g: np.asarray(idx) for g, idx in df.group_by("g", maintain_order=True).agg(pl.col("i")).iter_rows()}
