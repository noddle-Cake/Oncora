"""Compound–target label construction (spec 02, Req 3; DR-4, DR-6, FR-7).

Technique: binary labels with constructed negatives. ChEMBL records mostly what chemists
*tested*, and they test promising compounds, so actives dominate and "never measured" is not
"inactive". Labels come from three tiers, recorded per row in `negative_source`:

- **positive:** measured median pChEMBL ≥ threshold;
- **measured negative:** measured median below the threshold, or a censored ">" bound below it;
- **sampled negative:** where a target has too few measured negatives, compounds that are active
  on *other* targets but were never measured on this one. Being active elsewhere means they are
  real, drug-like, assay-friendly molecules, not trivially different decoys. The assumption that
  they are inactive here ("absence ≠ inactive" in general) is recorded in provenance and must
  appear in the model card.

Exclusion and collision checks use the connectivity block (D-01), so a stereoisomer of a
measured compound is never sampled as a negative for the same target.
"""

import argparse
import zlib
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from oncora.config import (
    ACTIVE_PCHEMBL,
    DATA_INTERIM,
    DATA_PROCESSED,
    DEMO_HOLDOUT,
    MIN_POSITIVES_PER_TARGET,
    NEG_RATIO,
    REPORTS,
    SEED,
)
from oncora.data_prep.provenance import DataIntegrityError, assert_table, write_parquet

ACTIVITIES = DATA_INTERIM / "chembl_activities.parquet"
CHEMBL_DIR = DATA_INTERIM / "chembl"
OUT = DATA_PROCESSED / "target_pairs.parquet"
DEMO_OUT = DATA_PROCESSED / "demo_holdout_targets.parquet"

ASSUMPTION = (
    "Sampled negatives are compounds active on some other target but never measured on this one; "
    "they are assumed inactive. Some will be unmeasured true actives (label noise biased toward "
    "under-estimating precision)."
)


def target_annotations(chembl_dir: Path = CHEMBL_DIR) -> pl.DataFrame:
    """target_chembl_id → gene symbol and UniProt accession (single-protein targets have one component)."""
    p = {t: f"'{chembl_dir / f'{t}.parquet'}'" for t in
         ("target_dictionary", "target_components", "component_sequences", "component_synonyms")}
    return duckdb.sql(f"""
        SELECT td.chembl_id AS target_chembl_id,
               MIN(syn.component_synonym) AS target_gene_symbol,
               MIN(seq.accession) AS uniprot_accession
        FROM {p["target_dictionary"]} td
        JOIN {p["target_components"]} tc USING (tid)
        JOIN {p["component_sequences"]} seq USING (component_id)
        LEFT JOIN {p["component_synonyms"]} syn
               ON syn.component_id = tc.component_id AND syn.syn_type = 'GENE_SYMBOL'
        WHERE td.target_type = 'SINGLE PROTEIN'
        GROUP BY td.chembl_id
    """).pl()


def label_measured(act: pl.DataFrame, active: float) -> pl.DataFrame:
    return act.with_columns(
        pl.when(pl.col("pchembl_median") >= active).then(1).otherwise(0).cast(pl.Int8).alias("label"),
    ).with_columns(
        pl.when(pl.col("label") == 1).then(None).otherwise(pl.lit("measured")).alias("negative_source"),
    )


def _target_seed(target: str, seed: int) -> int:
    return (zlib.crc32(target.encode()) ^ seed) & 0xFFFFFFFF


def sample_negatives(measured: pl.DataFrame, neg_ratio: int, seed: int) -> pl.DataFrame:
    """Top up each target to neg_ratio × positives with compounds active elsewhere."""
    # Pool: one representative row per connectivity block that is active on any kept target.
    pool = (
        measured.filter(pl.col("label") == 1)
        .sort("connectivity", "inchi_key")
        .group_by("connectivity", maintain_order=True)
        .agg(pl.col("inchi_key").first(), pl.col("canonical_smiles").first())
    )
    pool_conn = pool["connectivity"].to_numpy()
    index_of = {c: i for i, c in enumerate(pool_conn)}

    per_target = (
        measured.group_by("target_chembl_id")
        .agg(
            (pl.col("label") == 1).sum().alias("n_pos"),
            (pl.col("label") == 0).sum().alias("n_neg"),
            pl.col("connectivity").unique().alias("measured_conn"),
        )
        .sort("target_chembl_id")
    )
    picks_t, picks_i = [], []
    for row in per_target.iter_rows(named=True):
        need = neg_ratio * row["n_pos"] - row["n_neg"]
        if need <= 0:
            continue
        excluded = {index_of[c] for c in row["measured_conn"] if c in index_of}
        available = len(pool_conn) - len(excluded)
        need = min(need, available)
        rng = np.random.default_rng(_target_seed(row["target_chembl_id"], seed))
        # Oversample by the excluded count so that, after dropping exclusions, enough remain.
        draw = rng.choice(len(pool_conn), size=min(len(pool_conn), need + len(excluded)), replace=False)
        chosen = [i for i in draw if i not in excluded][:need]
        picks_t.extend([row["target_chembl_id"]] * len(chosen))
        picks_i.extend(chosen)

    idx = pl.Series("pool_idx", picks_i, dtype=pl.UInt32)
    return (
        pool.with_row_index("pool_idx").join(
            pl.DataFrame({"target_chembl_id": picks_t, "pool_idx": idx}), on="pool_idx"
        )
        .drop("pool_idx")
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("pchembl_median"),
            pl.lit(None, dtype=pl.Float64).alias("censored_bound"),
            pl.lit(0, dtype=pl.Int8).alias("label"),
            pl.lit("sampled").alias("negative_source"),
        )
    )


def build_target_pairs(
    act: pl.DataFrame,
    annotations: pl.DataFrame,
    *,
    active: float = ACTIVE_PCHEMBL,
    min_pos: int = MIN_POSITIVES_PER_TARGET,
    neg_ratio: int = NEG_RATIO,
    seed: int = SEED,
    holdout: dict[str, str] = DEMO_HOLDOUT,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    is_demo = pl.col("connectivity").is_in(list(holdout))
    demo = label_measured(act.filter(is_demo), active)
    measured = label_measured(act.filter(~is_demo), active)

    n_pos = measured.group_by("target_chembl_id").agg((pl.col("label") == 1).sum().alias("n_pos"))
    keep = n_pos.filter(pl.col("n_pos") >= min_pos)["target_chembl_id"]
    measured = measured.filter(pl.col("target_chembl_id").is_in(keep.implode()))

    sampled = sample_negatives(measured, neg_ratio, seed)
    names = measured.select("target_chembl_id", "target_name").unique("target_chembl_id")
    sampled = sampled.join(names, on="target_chembl_id")

    cols = ["inchi_key", "connectivity", "canonical_smiles", "target_chembl_id", "target_name",
            "pchembl_median", "censored_bound", "label", "negative_source"]
    pairs = (
        pl.concat([measured.select(cols), sampled.select(cols)])
        .join(annotations, on="target_chembl_id", how="left")
        .sort("target_chembl_id", "label", "inchi_key", descending=[False, True, False])
    )

    collisions = sampled.join(measured, on=["connectivity", "target_chembl_id"], how="semi")
    if collisions.height:
        raise DataIntegrityError(f"{collisions.height} sampled negatives collide with measured pairs")
    assert_table(pairs, name="target_pairs", required_cols=cols, min_rows=10_000,
                 max_null_rate={"inchi_key": 0.0, "canonical_smiles": 0.0, "label": 0.0},
                 unique_key=["inchi_key", "target_chembl_id"])

    stats = {
        "targets_total": n_pos.height,
        "targets_kept": keep.len(),
        "targets_dropped_min_pos": n_pos.height - keep.len(),
        "demo_pairs_removed": demo.height,
    }
    return pairs, demo, stats


def render_report(pairs: pl.DataFrame, demo: pl.DataFrame, stats: dict, *, active, min_pos, neg_ratio) -> str:
    per_t = pairs.group_by("target_chembl_id").agg(
        (pl.col("label") == 1).sum().alias("pos"),
        (pl.col("negative_source") == "measured").sum().alias("neg_measured"),
        (pl.col("negative_source") == "sampled").sum().alias("neg_sampled"),
    )
    counts = dict(pairs.group_by("negative_source").len().iter_rows())
    n_pos = counts.get(None, 0)
    q = per_t["pos"].quantile
    frac_sampled = per_t.select(
        (pl.col("neg_sampled") / (pl.col("neg_sampled") + pl.col("neg_measured"))).alias("f")
    )["f"]
    top = (
        pairs.filter(pl.col("label") == 1)
        .group_by("target_chembl_id", "target_name", "target_gene_symbol").len()
        .sort(["len", "target_chembl_id"], descending=[True, False]).head(10)
    )
    demo_names = ", ".join(DEMO_HOLDOUT.values())
    demo_top = demo.filter(pl.col("label") == 1).sort(["pchembl_median", "target_chembl_id"], descending=[True, False]).head(8)
    lines = [
        "# Compound–target modeling table",
        "",
        "Generated by `uv run python -m oncora.data_prep.target_pairs`. Do not edit by hand.",
        "",
        f"Policy: active ⇔ median pChEMBL ≥ {active:g}; targets need ≥ {min_pos} positives; "
        f"negatives topped up to {neg_ratio}× positives by sampling.",
        "",
        "| | Count |",
        "|---|---|",
        f"| targets with any measured pair | {stats['targets_total']:,} |",
        f"| targets kept (≥ {min_pos} positives) | **{stats['targets_kept']:,}** |",
        f"| rows | {pairs.height:,} |",
        f"| positives | {n_pos:,} |",
        f"| measured negatives | {counts.get('measured', 0):,} |",
        f"| sampled negatives | {counts.get('sampled', 0):,} |",
        f"| distinct compounds (connectivity) | {pairs['connectivity'].n_unique():,} |",
        "",
        f"Positives per target: median {q(0.5):,.0f}, p90 {q(0.9):,.0f}, max {per_t['pos'].max():,}. "
        f"Share of negatives that are sampled, per target: median {frac_sampled.median():.0%}.",
        "",
        f"**Assumption (model card):** {ASSUMPTION}",
        "",
        "## Targets with the most positives",
        "",
        "| Target | Name | Gene | Positives |",
        "|---|---|---|---|",
        *[f"| {r['target_chembl_id']} | {r['target_name']} | {r['target_gene_symbol'] or ''} | {r['len']:,} |"
          for r in top.iter_rows(named=True)],
        "",
        f"## Demo holdout ({demo_names})",
        "",
        f"{stats['demo_pairs_removed']} ChEMBL pairs removed before any target filtering and saved "
        f"to `{DEMO_OUT.relative_to(DATA_PROCESSED.parent.parent)}` for the spec 09 comparison. Top measured actives:",
        "",
        "| Target | pChEMBL |",
        "|---|---|",
        *[f"| {r['target_name']} | {r['pchembl_median']:.2f} |" for r in demo_top.iter_rows(named=True)],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, default=REPORTS / "target_pairs.md")
    args = ap.parse_args()
    act = pl.read_parquet(ACTIVITIES)
    pairs, demo, stats = build_target_pairs(act, target_annotations())
    config = {"active_pchembl": ACTIVE_PCHEMBL, "min_positives_per_target": MIN_POSITIVES_PER_TARGET,
              "neg_ratio": NEG_RATIO, "seed": SEED, "demo_holdout": DEMO_HOLDOUT,
              "negative_assumption": ASSUMPTION}
    src = [ACTIVITIES] + [CHEMBL_DIR / f"{t}.parquet" for t in
                          ("target_dictionary", "target_components", "component_sequences", "component_synonyms")]
    write_parquet(pairs, OUT, sources=src, created_by=__spec__.name, config=config,
                  filters=["demo holdout removed", f"targets with >= {MIN_POSITIVES_PER_TARGET} positives"])
    write_parquet(demo, DEMO_OUT, sources=[ACTIVITIES], created_by=__spec__.name, config=config)
    args.report.write_text(render_report(pairs, demo, stats, active=ACTIVE_PCHEMBL,
                                         min_pos=MIN_POSITIVES_PER_TARGET, neg_ratio=NEG_RATIO))
    print(f"Wrote {OUT.name}: {pairs.height:,} rows, {stats['targets_kept']:,} targets")


if __name__ == "__main__":
    main()
