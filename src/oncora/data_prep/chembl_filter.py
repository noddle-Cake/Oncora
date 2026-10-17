"""ChEMBL activity filtering and potency standardization (spec 02, Req 2; DR-5, FR-6).

Technique: ChEMBL pools millions of measurements from thousands of papers with different assay
types, units, confidence levels and censoring. Using them naively mixes cell-based phenotypes
with binding constants and treats "> 10 µM" as if it were a number. This stage keeps
comparable measurements and puts them on one scale:

- **Scope:** binding (B) and functional (F) assays on a single human protein, mapped with high
  confidence (assay confidence_score ≥ 8: a direct or homologous single-protein assignment).
- **Scale:** pChEMBL = −log10(molar potency), which unifies IC50/EC50/Ki/Kd/… on one axis.
- **Censoring:** "> X" says the true potency is weaker than X. If even X is below the activity
  threshold, the compound is reliably inactive; otherwise the row carries no label. "< X" is
  ambiguous for an inactive call and is dropped.
- **Dedup:** many pairs are measured repeatedly. The median pChEMBL per (compound, target) is
  robust to an outlier assay.
"""

import argparse
from pathlib import Path

import duckdb
import polars as pl

from oncora.config import ACTIVE_PCHEMBL, CHEMBL_MIN_CONFIDENCE, DATA_INTERIM, REPORTS
from oncora.data_prep.provenance import assert_table, write_parquet

CHEMBL_DIR = DATA_INTERIM / "chembl"
OUT = DATA_INTERIM / "chembl_activities.parquet"

# Activity types for which ChEMBL computes pChEMBL. Censored rows of these types (in nM) get an
# implied pChEMBL bound on the same scale.
POTENCY_TYPES = ("IC50", "XC50", "EC50", "AC50", "Ki", "Kd", "Potency", "ED50")
GREATER = (">", ">=", ">>")
LESS = ("<", "<=", "<<")
# A data_validity_comment flags a suspect value, except this one, which confirms it.
VALID_COMMENT = "Manually validated"


def _p(table: str) -> str:
    return f"'{CHEMBL_DIR / f'{table}.parquet'}'"


def _scoped_sql(min_confidence: int) -> str:
    """Activities joined to assay, target and structure, restricted to the modeling scope."""
    return f"""
        SELECT a.activity_id, a.assay_id, a.doc_id, a.molregno,
               a.standard_type, a.standard_relation, a.standard_value, a.standard_units,
               a.pchembl_value::DOUBLE AS pchembl_value,
               a.data_validity_comment, a.potential_duplicate,
               s.assay_type, s.confidence_score,
               td.chembl_id AS target_chembl_id, td.pref_name AS target_name,
               cs.standard_inchi_key AS inchi_key, cs.canonical_smiles,
               md.chembl_id AS mol_chembl_id
        FROM {_p("activities")} a
        JOIN {_p("assays")} s USING (assay_id)
        JOIN {_p("target_dictionary")} td ON td.tid = s.tid
        JOIN {_p("compound_structures")} cs USING (molregno)
        JOIN {_p("molecule_dictionary")} md USING (molregno)
        WHERE s.assay_type IN ('B', 'F')
          AND s.confidence_score >= {min_confidence}
          AND td.target_type = 'SINGLE PROTEIN'
          AND td.organism = 'Homo sapiens'
    """


def funnel(con, min_confidence: int) -> list[tuple[str, int]]:
    """Row counts after each filter, for the report."""
    act, ass = _p("activities"), _p("assays")
    steps = [
        ("all activities", f"SELECT COUNT(*) FROM {act}"),
        ("binding/functional assays", f"SELECT COUNT(*) FROM {act} JOIN {ass} USING (assay_id) WHERE assay_type IN ('B','F')"),
        (f"… confidence ≥ {min_confidence}", f"SELECT COUNT(*) FROM {act} JOIN {ass} USING (assay_id) WHERE assay_type IN ('B','F') AND confidence_score >= {min_confidence}"),
        ("… single human protein + structure", "SELECT COUNT(*) FROM scoped"),
        ("… no validity flag, not a duplicate", "SELECT COUNT(*) FROM clean"),
        ("… usable potency (measured or censored-inactive)", "SELECT COUNT(*) FROM usable"),
    ]
    return [(label, con.sql(q).fetchone()[0]) for label, q in steps]


def filter_activities(min_confidence: int = CHEMBL_MIN_CONFIDENCE,
                      active: float = ACTIVE_PCHEMBL) -> tuple[pl.DataFrame, list[tuple[str, int]], dict]:
    con = duckdb.connect()
    con.sql(f"CREATE TEMP TABLE scoped AS {_scoped_sql(min_confidence)} AND cs.standard_inchi_key IS NOT NULL")
    con.sql(f"""
        CREATE TEMP TABLE clean AS SELECT * FROM scoped
        WHERE (data_validity_comment IS NULL OR data_validity_comment = '{VALID_COMMENT}')
          AND potential_duplicate = 0
    """)
    types = ", ".join(f"'{t}'" for t in POTENCY_TYPES)
    greater = ", ".join(f"'{r}'" for r in GREATER)
    # implied pChEMBL for a censored nM value: -log10(value * 1e-9) = 9 - log10(value)
    con.sql(f"""
        CREATE TEMP TABLE usable AS
        SELECT *, 'measured' AS kind, pchembl_value AS pchembl
        FROM clean WHERE standard_relation = '=' AND pchembl_value IS NOT NULL
        UNION ALL
        SELECT *, 'censored_inactive' AS kind, 9 - log10(standard_value) AS pchembl
        FROM clean
        WHERE standard_relation IN ({greater}) AND standard_type IN ({types})
          AND standard_units = 'nM' AND standard_value > 0
          AND 9 - log10(standard_value) < {active}
    """)
    relation_counts = dict(con.sql(
        "SELECT COALESCE(standard_relation, 'NULL'), COUNT(*) FROM clean GROUP BY 1"
    ).fetchall())
    steps = funnel(con, min_confidence)

    # One row per (InChIKey, target). Measured values win over censored ones; censored-only
    # pairs keep the strongest (largest) bound, which is still below the active threshold.
    pairs = con.sql("""
        SELECT inchi_key,
               LEFT(inchi_key, 14) AS connectivity,
               ANY_VALUE(canonical_smiles) AS canonical_smiles,
               ANY_VALUE(mol_chembl_id) AS mol_chembl_id,
               target_chembl_id,
               ANY_VALUE(target_name) AS target_name,
               MEDIAN(pchembl) FILTER (WHERE kind = 'measured') AS pchembl_median,
               MAX(pchembl) FILTER (WHERE kind = 'censored_inactive') AS censored_bound,
               COUNT(*) FILTER (WHERE kind = 'measured') AS n_measured,
               COUNT(*) FILTER (WHERE kind = 'censored_inactive') AS n_censored,
               COUNT(DISTINCT doc_id) AS n_documents
        FROM usable
        GROUP BY inchi_key, target_chembl_id
        ORDER BY target_chembl_id, inchi_key
    """).pl()
    stats = {
        "usable_rows": steps[-1][1],
        "relations_after_quality_filter": relation_counts,
        "measured_rows": con.sql("SELECT COUNT(*) FROM usable WHERE kind='measured'").fetchone()[0],
        "censored_inactive_rows": con.sql("SELECT COUNT(*) FROM usable WHERE kind='censored_inactive'").fetchone()[0],
    }
    assert_table(pairs, name="chembl_activities",
                 required_cols=["inchi_key", "target_chembl_id", "pchembl_median"],
                 min_rows=100_000, max_null_rate={"inchi_key": 0.0, "canonical_smiles": 0.0},
                 unique_key=["inchi_key", "target_chembl_id"])
    return pairs, steps, stats


def render_report(pairs: pl.DataFrame, steps, stats, min_confidence: int, active: float) -> str:
    measured = pairs.filter(pl.col("pchembl_median").is_not_null())
    censored_only = pairs.filter(pl.col("pchembl_median").is_null())
    n_active = (measured["pchembl_median"] >= active).sum()
    repeat = measured.filter(pl.col("n_measured") > 1)
    rel = stats["relations_after_quality_filter"]
    lines = [
        "# ChEMBL activity filtering",
        "",
        "Generated by `uv run python -m oncora.data_prep.chembl_filter`. Do not edit by hand.",
        "",
        f"Scope: binding/functional assays, confidence ≥ {min_confidence}, single human protein; "
        f"active threshold pChEMBL ≥ {active:g} (≤ {10 ** (9 - active):,.0f} nM).",
        "",
        "## Filter funnel (activity rows)",
        "",
        "| Step | Rows |",
        "|---|---|",
        *[f"| {label} | {n:,} |" for label, n in steps],
        "",
        f"Of the usable rows, {stats['measured_rows']:,} are measured (`=` with a pChEMBL) and "
        f"{stats['censored_inactive_rows']:,} are censored inactives (`>`/`>=`/`>>` with a bound "
        f"below {active:g}).",
        "",
        "Relations after the quality filter, before the potency rules: "
        + ", ".join(f"`{k}` {v:,}" for k, v in sorted(rel.items(), key=lambda kv: -kv[1]))
        + ". `<`-type rows are dropped (ambiguous for an inactive call), and so are `~` and "
        "unlabeled rows. `=` rows without a pChEMBL (percent inhibition, non-molar units, "
        "activity types outside the pChEMBL set) are also dropped.",
        "",
        "## After dedup: one row per (InChIKey, target)",
        "",
        "| | Count |",
        "|---|---|",
        f"| compound–target pairs | {pairs.height:,} |",
        f"| … with a measured median pChEMBL | {measured.height:,} |",
        f"| … … active (≥ {active:g}) | {n_active:,} ({n_active / measured.height:.1%} of measured) |",
        f"| … censored-inactive only | {censored_only.height:,} |",
        f"| distinct compounds (InChIKey) | {pairs['inchi_key'].n_unique():,} |",
        f"| distinct connectivity blocks | {pairs['connectivity'].n_unique():,} |",
        f"| distinct targets | {pairs['target_chembl_id'].n_unique():,} |",
        f"| pairs measured more than once | {repeat.height:,} |",
        "",
        "Dedup rule: the median of measured pChEMBL values. Censored rows are ignored when a "
        "measured value exists. Censored-only pairs keep the largest bound.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, default=REPORTS / "chembl_filter.md")
    args = ap.parse_args()
    pairs, steps, stats = filter_activities()
    sources = [CHEMBL_DIR / f"{t}.parquet" for t in
               ("activities", "assays", "target_dictionary", "compound_structures", "molecule_dictionary")]
    write_parquet(
        pairs, OUT, sources=sources, created_by=__spec__.name,
        filters=["assay_type in (B, F)", f"confidence_score >= {CHEMBL_MIN_CONFIDENCE}",
                 "SINGLE PROTEIN, Homo sapiens", f"data_validity_comment null or '{VALID_COMMENT}'",
                 "potential_duplicate = 0", "'=' with pchembl, or censored > bound < active",
                 "median pchembl per (inchi_key, target)"],
        config={"min_confidence": CHEMBL_MIN_CONFIDENCE, "active_pchembl": ACTIVE_PCHEMBL,
                "potency_types": list(POTENCY_TYPES)},
    )
    args.report.write_text(render_report(pairs, steps, stats, CHEMBL_MIN_CONFIDENCE, ACTIVE_PCHEMBL))
    print(f"Wrote {OUT.name}: {pairs.height:,} pairs; report {args.report}")


if __name__ == "__main__":
    main()
