"""DSS3 sensitivity target EDA (spec 01, Req 4.1–4.3; FR-12).

Questions answered:
1. How skewed is DSS3, and which transform tames it?
2. Is DSS3 on one scale? (No: see `mixed_scale_compounds`.)
3. How much variance is due to *which compound* versus *which cell line*? This decides what the
   model must learn from structure and which baselines are meaningful.
4. Where should a sensitive/resistant threshold go?

Technique notes:
- Variance decomposition: fit row = mean + compound effect + cell-line effect by alternating
  group means (a two-way additive ANOVA without interaction) and report R² for each part. The
  remainder is interaction plus noise, which is the cell-line-specific response.
- log1p compresses the long right tail, and ranks remove scale entirely. Compare how each
  changes the decomposition.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from oncora.config import DATA_INTERIM, REPORTS
from oncora.eda.common import save_fig, trainable_model_ids

# Values at or above this mark the second population (see report): a different scale, not biology.
SCALE_BREAK = 1.0


def load_trainable() -> pl.DataFrame:
    s = pl.read_parquet(DATA_INTERIM / "sensitivity.parquet")
    names = pl.read_parquet(DATA_INTERIM / "compound_bridge.parquet", columns=["pert_id", "cmap_names"])
    return (
        s.filter(pl.col("model_id").is_in(trainable_model_ids()))
        .join(names, on="pert_id", how="left")
        .with_columns(pl.col("cmap_names").list.join(", ").alias("name"))
        .drop("cmap_names")
    )


def mixed_scale_compounds(df: pl.DataFrame, cutoff: float = SCALE_BREAK) -> pl.DataFrame:
    """Compounds with any DSS3 >= cutoff, with how many of their rows sit above it."""
    return (
        df.group_by("pert_id", "name")
        .agg(
            pl.len().alias("n_lines"),
            (pl.col("dss3") >= cutoff).mean().alias("frac_high"),
            pl.col("dss3").median().alias("median"),
            pl.col("dss3").max().alias("max"),
        )
        .filter(pl.col("frac_high") > 0)
        .sort("frac_high", descending=True)
    )


def variance_decomposition(df: pl.DataFrame, col: str, n_iter: int = 25) -> dict[str, float]:
    d = df.select("pert_id", "model_id", col).to_pandas()
    t = d[col]
    mu = t.mean()
    tot = ((t - mu) ** 2).sum()

    def r2(pred) -> float:
        return float(1 - ((t - pred) ** 2).sum() / tot)

    line_effect = pd.Series(0.0, index=d["model_id"].unique())
    for _ in range(n_iter):
        cmp_effect = (t - mu - d["model_id"].map(line_effect)).groupby(d["pert_id"]).mean()
        line_effect = (t - mu - d["pert_id"].map(cmp_effect)).groupby(d["model_id"]).mean()
    return {
        "compound": r2(t.groupby(d["pert_id"]).transform("mean")),
        "cell_line": r2(t.groupby(d["model_id"]).transform("mean")),
        "additive": r2(mu + d["pert_id"].map(cmp_effect) + d["model_id"].map(line_effect)),
    }


def _quantiles(y: np.ndarray, qs=(0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999)) -> dict[float, float]:
    return {q: float(np.quantile(y, q)) for q in qs}


def fig_log_histogram(y: np.ndarray) -> str:
    pos = y[y > 1e-6]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.hist(np.log10(pos), bins=np.arange(-6, 2.25, 0.125), color="#4C72B0")
    ax.axvline(np.log10(SCALE_BREAK), color="#C44E52", ls="--", lw=1)
    ax.text(np.log10(SCALE_BREAK) + 0.05, ax.get_ylim()[1] * 0.85, "DSS3 = 1", color="#C44E52")
    ax.set_yscale("log")
    ax.set_xlabel("log10(DSS3), rows with DSS3 > 1e-6")
    ax.set_ylabel("rows (log scale)")
    ax.set_title(f"DSS3 is bimodal; {np.mean(y == 0):.1%} of rows are exactly 0 (not shown)")
    return save_fig(fig, "dss3_log_histogram.png")


def fig_transforms(clean: pl.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    for ax, (label, vals) in zip(
        axes,
        [
            ("raw DSS3", clean["dss3"]),
            ("log1p(DSS3)", clean["dss3"].log1p()),
            ("rank (percentile)", clean["dss3"].rank() / clean.height),
        ],
    ):
        ax.hist(vals.to_numpy(), bins=60, color="#55A868")
        ax.set_title(f"{label}\nskew {vals.skew():.2f}")
        ax.set_yticks([])
    fig.suptitle("Target transforms on the single-scale set", y=1.05)
    return save_fig(fig, "dss3_transforms.png")


def fig_decomposition(rows: list[tuple[str, dict[str, float]]]) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    parts = ["compound", "cell_line", "additive"]
    width = 0.8 / len(rows)
    x = np.arange(len(parts))
    for i, (label, dec) in enumerate(rows):
        bars = ax.bar(x + i * width, [dec[p] for p in parts], width, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x + width * (len(rows) - 1) / 2, ["compound only", "cell line only", "compound + cell line"])
    ax.set_ylabel("R² of group-mean model")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Which compound matters far more than which cell line")
    return save_fig(fig, "dss3_variance_decomposition.png")


def build_report() -> str:
    df = load_trainable()
    y = df["dss3"].to_numpy()
    mixed = mixed_scale_compounds(df)
    clean = df.filter(~pl.col("pert_id").is_in(mixed["pert_id"].implode()))
    yc = clean["dss3"].to_numpy()
    n_mixed_rows_high = int((y >= SCALE_BREAK).sum())

    decs = []
    for label, data in [("all compounds", df), ("single-scale compounds", clean)]:
        data = data.with_columns(
            pl.col("dss3").log1p().alias("log1p"), pl.col("dss3").rank().alias("rank")
        )
        decs.append((f"{label}, log1p", variance_decomposition(data, "log1p")))
        decs.append((f"{label}, rank", variance_decomposition(data, "rank")))

    per_cmp = clean.group_by("pert_id").agg(pl.col("dss3").mean().alias("m"))
    big = df.group_by("pert_id").len().filter(pl.col("len") > 700)["pert_id"]
    q_all, q_clean = _quantiles(y), _quantiles(yc)

    figs = {
        "hist": fig_log_histogram(y),
        "transforms": fig_transforms(clean),
        "decomp": fig_decomposition(decs),
    }

    def qrow(q):
        return f"| p{q * 100:g} | {q_all[q]:.4g} | {q_clean[q]:.4g} |"

    lines = [
        "# DSS3 target EDA",
        "",
        "Generated by `uv run python -m oncora.eda.target`. Do not edit by hand. Scope: the "
        f"{df.height:,} DSS3 rows on the {df['model_id'].n_unique():,} cell lines that also have "
        "CCLE expression.",
        "",
        "## 1. DSS3 is not on a single scale",
        "",
        f"![log histogram]({figs['hist']})",
        "",
        f"- {np.mean(y == 0):.1%} of rows are exactly 0. The rest form a main mode around 0.1 and "
        f"stop sharply at 1. A separate population runs up to {y.max():.0f}.",
        f"- **Every one of the {n_mixed_rows_high:,} rows with DSS3 ≥ 1 belongs to just "
        f"{mixed.height} of {df['pert_id'].n_unique():,} compounds.** Excluding them, the maximum "
        f"is {yc.max():.3f}.",
        "- Those compounds are mostly the reference drugs of the original CCLE drug screen "
        "(PD-0325901, panobinostat, paclitaxel, topotecan, lapatinib, crizotinib, selumetinib, "
        "saracatinib, tanespimycin, NVP-AEW541, NVP-TAE684, RAF-265, PHA-665752 …). Most are "
        f"measured on {mixed['n_lines'].min()}–{mixed['n_lines'].max()} cell lines, against a "
        f"typical ~460. Overall, {big.len()} compounds exceed 700 lines, consistent with several "
        "studies merged per compound.",
        "- Within one of these compounds, values both above and below 1 occur (topotecan: about "
        "half each). The likeliest explanation is that PharmacoDB merged studies reporting DSS3 on "
        "different scales (≈0–1 vs ≈0–100). **For these compounds, a value's scale cannot be "
        "recovered row by row.** A value of 0.5 could be 0.5 on the 0–1 scale or 0.005 on the "
        "0–100 scale.",
        "",
        "**Consequence:** a global threshold at 1 would mostly label *came from the CCLE screen*, "
        "not *sensitive*, and the raw values of these compounds would dominate any regression "
        "loss.",
        "",
        "| pert_id | name | lines | rows ≥ 1 | median | max |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['pert_id']} | {r['name']} | {r['n_lines']} | {r['frac_high']:.0%} | {r['median']:.3g} | {r['max']:.3g} |"
        for r in mixed.iter_rows(named=True)
    ]
    lines += [
        "",
        "## 2. Distribution and transforms",
        "",
        "| Quantile | all rows | single-scale compounds |",
        "|---|---|---|",
        *[qrow(q) for q in q_all],
        "",
        f"![transforms]({figs['transforms']})",
        "",
        f"On the single-scale set, raw DSS3 has skew {clean['dss3'].skew():.2f}, log1p has "
        f"{clean['dss3'].log1p().skew():.2f}, and ranks are uniform by construction. log1p barely helps once "
        "values are confined to [0, 1], because log1p is nearly linear there. Most of the skew "
        "comes from the zero spike plus the long tail toward 1.",
        "",
        "## 3. Variance decomposition: compound vs. cell line",
        "",
        f"![decomposition]({figs['decomp']})",
        "",
        "| Data, transform | compound only | cell line only | compound + cell line (additive) |",
        "|---|---|---|---|",
        *[
            f"| {label} | {d['compound']:.3f} | {d['cell_line']:.3f} | {d['additive']:.3f} |"
            for label, d in decs
        ],
        "",
        "- **Compound identity explains most of the variance; cell-line identity explains about "
        "4–5%.** This holds with and without the mixed-scale compounds and under both transforms.",
        "- The additive model leaves about 24–30% for compound × cell-line interaction plus noise. "
        "That remainder is the selectivity signal: which lines respond more than the compound's average.",
        f"- Per-compound mean DSS3 (single-scale) ranges {per_cmp['m'].min():.3f}–{per_cmp['m'].max():.3f}. "
        "Predicting this overall potency from structure is most of the achievable signal for an unseen compound.",
        "",
        "Implications for specs 04 and 06:",
        "",
        "1. The **cell-line-mean baseline is weak** (R² ≈ 0.04), not strong as spec 06 assumed. "
        "Beating it is necessary but far from sufficient.",
        "2. **Per-cell-line ρ** (ranking compounds within a line) mostly measures predicted "
        "potency, the dominant, structure-driven component. **Per-compound ρ** (ranking lines for "
        "one compound) measures selectivity, the smaller and harder component. Report both and "
        "interpret them separately.",
        "3. Add a **compound-features-only ablation**. If the full model does not beat it on "
        "per-compound ρ, the cell-line features add nothing, and the cell-line rankings shown to "
        "users would amount to the compound's potency plus the line's average.",
        "",
        "## 4. Sensitive/resistant threshold: candidates",
        "",
        "There is no natural break inside the single-scale data (the only break, at 1, is the scale artifact).",
        "",
        "| Candidate | Definition | Positive rate | Measures |",
        "|---|---|---|---|",
        f"| A. Global p90 | DSS3 ≥ {q_clean[0.9]:.3f} (single-scale set) | 10% | potent compound × responsive line; dominated by potent compounds |",
        f"| B. Global p95 | DSS3 ≥ {q_clean[0.95]:.3f} | 5% | same, stricter |",
        "| C. Within-compound top decile | line is in the top 10% of that compound's lines | 10% of each compound | selectivity: which lines respond most to *this* compound, which matches the product question |",
        "",
        "**Recommendation (decide at the task 6 checkpoint):**",
        "",
        "- Exclude the mixed-scale compounds from headline training and evaluation, list them in "
        "the model card, and report one sensitivity analysis that includes them after rescaling "
        "values ≥ 1 by 1/100.",
        "- Regression target: rank-transformed DSS3 (robust to the zero spike and any residual "
        "scale differences). Headline: Spearman ρ.",
        "- Classification label: **C (within-compound top decile)** as the product-facing metric, "
        "with A reported for comparison.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPORTS / "eda_target.md")
    args = ap.parse_args()
    args.out.write_text(build_report())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
