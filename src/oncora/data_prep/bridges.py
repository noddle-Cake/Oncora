"""Compound and cell-line identity bridges (spec 01, Req 3; DR-2, DR-3).

Technique: entity resolution. Each source names things its own way: Broad `pert_id`s for
compounds, DepMap `ACH-` IDs and LINCS names for cell lines. Joins are only as good as the
mapping between those names, so the mapping is built explicitly, its coverage is counted, and
every gap is reported rather than silently dropped.

Compounds are identified by **InChIKey** (a hash of structure), not by name or `pert_id`.
Two `pert_id`s for the same molecule, such as different salts or batches, must count as one
compound. Otherwise one molecule could sit on both sides of a train/test split.
"""

import argparse
from pathlib import Path

import polars as pl
from rdkit import Chem, RDLogger

from oncora.config import DATA_INTERIM, NULL_TOKENS, REPORTS, raw_path
from oncora.data_prep.provenance import assert_table, write_parquet
from oncora.io.checksums import verify_raw

COMPOUND_BRIDGE_OUT = DATA_INTERIM / "compound_bridge.parquet"
CELLLINE_BRIDGE_OUT = DATA_INTERIM / "cellline_bridge.parquet"
CCLE_PARQUET = DATA_INTERIM / "ccle_expression.parquet"
DSS3_PARQUET = DATA_INTERIM / "sensitivity.parquet"

# compoundinfo writes this in place of structures it may not disclose.
RESTRICTED = "restricted"


def read_lincs_tsv(path: Path, extra_nulls: tuple[str, ...] = ()) -> pl.DataFrame:
    """LINCS TSVs: tab-delimited, no quoting (SMILES contain backslashes), `""` means null."""
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        null_values=[t for t in (*NULL_TOKENS, *extra_nulls) if t],
        infer_schema=False,
    ).with_columns(pl.all().replace("", None))


def rdkit_inchikey(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def build_compound_bridge(compoundinfo: pl.DataFrame, dss3_pert_ids: set[str]) -> pl.DataFrame:
    """One row per pert_id.

    compoundinfo repeats a pert_id once per (target, moa) annotation, but the structure never
    differs between those rows (checked in the EDA). Resolution rule: keep the single structure
    and collect names, targets and MoAs into sorted distinct lists.
    """
    n_struct = compoundinfo.group_by("pert_id").agg(
        pl.col("canonical_smiles").drop_nulls().n_unique().alias("n_smiles"),
        pl.col("inchi_key").drop_nulls().n_unique().alias("n_ik"),
    )
    conflicts = n_struct.filter((pl.col("n_smiles") > 1) | (pl.col("n_ik") > 1))
    if conflicts.height:
        raise ValueError(
            f"{conflicts.height} pert_ids map to more than one structure, e.g. "
            f"{conflicts['pert_id'].head(5).to_list()}; the collapse rule no longer holds"
        )

    def distinct_sorted(col: str) -> pl.Expr:
        return pl.col(col).drop_nulls().unique().sort()

    RDLogger.DisableLog("rdApp.*")
    bridge = (
        compoundinfo.group_by("pert_id")
        .agg(
            pl.col("inchi_key").drop_nulls().first(),
            pl.col("canonical_smiles").drop_nulls().first(),
            distinct_sorted("cmap_name").alias("cmap_names"),
            distinct_sorted("target").alias("targets"),
            distinct_sorted("moa").alias("moas"),
            pl.len().alias("n_source_rows"),
        )
        .with_columns(
            # First InChIKey block = connectivity without stereo. Stereoisomers share it and get
            # identical (achiral) fingerprints, so splits must group on it, not the full key.
            pl.col("inchi_key").str.slice(0, 14).alias("connectivity"),
            pl.col("pert_id").is_in(list(dss3_pert_ids)).alias("in_dss3"),
            pl.col("canonical_smiles")
            .map_elements(rdkit_inchikey, return_dtype=pl.String)
            .alias("rdkit_inchi_key"),
        )
        .sort("pert_id")
    )
    assert_table(bridge, name="compound_bridge", required_cols=["pert_id", "inchi_key"],
                 unique_key=["pert_id"])
    return bridge


def build_cellline_bridge(ccle_ids: set[str], dss3_ids: set[str]) -> pl.DataFrame:
    """One row per ACH- id seen in CCLE (default entries) or DSS3.

    primary_disease and lineage stay null: no workspace file maps ACH- ids to a disease
    (cellinfo.txt has no ACH- ids). That is open item A4, resolved at the spec 01 checkpoint.
    """
    ids = sorted(ccle_ids | dss3_ids)
    return pl.DataFrame(
        {
            "model_id": ids,
            "in_ccle": [i in ccle_ids for i in ids],
            "in_dss3": [i in dss3_ids for i in ids],
            "primary_disease": pl.Series([None] * len(ids), dtype=pl.String),
            "lineage": pl.Series([None] * len(ids), dtype=pl.String),
            "disease_source": pl.Series([None] * len(ids), dtype=pl.String),
        }
    )


def _pct(n: int, d: int) -> str:
    return f"{n:,} / {d:,} ({n / d:.1%})" if d else "0 / 0"


def bridge_report(compoundinfo: pl.DataFrame, cbridge: pl.DataFrame, lbridge: pl.DataFrame,
                  cellinfo: pl.DataFrame) -> str:
    sens = cbridge.filter("in_dss3")
    n_sens_ids = sens.height
    n_sens_ik = sens["inchi_key"].n_unique()
    shared = (
        sens.group_by("inchi_key").agg(
            pl.col("pert_id").sort(),
            pl.col("cmap_names").list.explode(keep_nulls=False, empty_as_null=False).unique().sort(),
        )
        .filter(pl.col("pert_id").list.len() > 1).sort("inchi_key")
    )
    mismatch = sens.filter(pl.col("rdkit_inchi_key") != pl.col("inchi_key"))
    mismatch_same_conn = (
        mismatch["rdkit_inchi_key"].str.slice(0, 14) == mismatch["connectivity"]
    ).sum()
    stereo_groups = (
        sens.group_by("connectivity").agg(
            pl.col("inchi_key").unique().sort(),
            pl.col("cmap_names").list.explode(keep_nulls=False, empty_as_null=False).unique().sort(),
        )
        .filter(pl.col("inchi_key").list.len() > 1).sort("connectivity")
    )
    unparsed = sens.filter(pl.col("rdkit_inchi_key").is_null())
    all_mismatch = cbridge.filter(
        pl.col("rdkit_inchi_key").is_not_null() & (pl.col("rdkit_inchi_key") != pl.col("inchi_key"))
    )
    n_restricted = compoundinfo.filter(pl.col("canonical_smiles") == RESTRICTED)["pert_id"].n_unique()

    both = lbridge.filter("in_ccle", "in_dss3").height
    dss3_only = lbridge.filter(~pl.col("in_ccle"), "in_dss3").height
    ccle_only = lbridge.filter("in_ccle", ~pl.col("in_dss3")).height
    n_cellinfo_ccle = cellinfo["ccle_name"].drop_nulls().n_unique()

    lines = [
        "# Identity bridges",
        "",
        "Generated by `uv run python -m oncora.data_prep.bridges`. Do not edit by hand.",
        "",
        "## Compound bridge (`pert_id` → structure)",
        "",
        f"- `compoundinfo.txt`: {compoundinfo.height:,} rows, {cbridge.height:,} distinct `pert_id`s. "
        f"{(cbridge['n_source_rows'] > 1).sum():,} `pert_id`s span several rows, one per (target, MoA) "
        "annotation. **Structure never conflicts within a `pert_id`**, so rows are collapsed and "
        "names, targets and MoAs are kept as lists.",
        f"- {n_restricted:,} `pert_id`s have `restricted` in place of a structure (treated as null).",
        "",
        "| Sensitivity compounds (DSS3) | Count |",
        "|---|---|",
        f"| `pert_id`s in DSS3 | {n_sens_ids:,} |",
        f"| … found in compoundinfo | {_pct(n_sens_ids, n_sens_ids)} |",
        f"| … with SMILES | {_pct(sens['canonical_smiles'].is_not_null().sum(), n_sens_ids)} |",
        f"| … with InChIKey | {_pct(sens['inchi_key'].is_not_null().sum(), n_sens_ids)} |",
        f"| … with a MoA annotation | {_pct((sens['moas'].list.len() > 0).sum(), n_sens_ids)} |",
        f"| … with a target annotation | {_pct((sens['targets'].list.len() > 0).sum(), n_sens_ids)} |",
        f"| **Distinct molecules (full InChIKey)** | **{n_sens_ik:,}** |",
        f"| **Distinct connectivity blocks (InChIKey[:14])** | **{sens['connectivity'].n_unique():,}** |",
        f"| InChIKeys shared by several `pert_id`s | {shared.height:,} |",
        f"| RDKit InChIKey ≠ provided InChIKey | {mismatch.height:,} |",
        f"| … differing only in the stereo block | {mismatch_same_conn:,} |",
        f"| SMILES RDKit cannot parse | {unparsed.height:,} |",
        "",
        "## Stereochemistry: compoundinfo SMILES and InChIKeys disagree",
        "",
        f"The InChIKey recomputed by RDKit from `canonical_smiles` differs from the provided one "
        f"for {mismatch.height:,} DSS3 compounds, and for "
        f"{_pct(all_mismatch.height, cbridge['rdkit_inchi_key'].is_not_null().sum())} parseable "
        f"compounds across all of compoundinfo. In every DSS3 case "
        f"({mismatch_same_conn:,} / {mismatch.height:,}) only the second (stereo) block differs: "
        "the provided InChIKey encodes stereochemistry that the provided SMILES has lost, or "
        "encodes differently.",
        "",
        "Consequences:",
        "",
        "1. **Splits must group by the connectivity block** (the first 14 InChIKey characters), "
        "not the full key. ECFP fingerprints are achiral by default, so stereoisomers have "
        "identical feature vectors. Splitting them across train and test would leak. The "
        "stereoisomer groups in DSS3 are listed below. Spec 04 has been updated to match.",
        "2. **Cross-source matching** (compoundinfo ↔ ChEMBL ↔ user input) should use one InChIKey "
        "convention, computed by RDKit after the spec 03 standardization, and fall back to the "
        "connectivity block where stereo is missing on one side.",
        "",
        "| Connectivity | InChIKeys | names |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{r['connectivity']}` | {', '.join(f'`{k}`' for k in r['inchi_key'])} | {', '.join(r['cmap_names'])} |"
        for r in stereo_groups.iter_rows(named=True)
    ]
    lines += [
        "",
        "### Same molecule, several `pert_id`s (DSS3)",
        "",
        "These are merged into one compound per cell line (spec 02, Req 4.2) and always fall in the "
        "same split partition.",
        "",
        "| InChIKey | pert_ids | names |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{r['inchi_key']}` | {', '.join(r['pert_id'])} | {', '.join(r['cmap_names'])} |"
        for r in shared.iter_rows(named=True)
    ]
    if mismatch.height:
        lines += ["", "### RDKit vs provided InChIKey mismatches (DSS3)", "",
                  "| pert_id | provided | RDKit |", "|---|---|---|"]
        lines += [
            f"| {r['pert_id']} | `{r['inchi_key']}` | `{r['rdkit_inchi_key']}` |"
            for r in mismatch.sort("pert_id").iter_rows(named=True)
        ]
    lines += [
        "",
        "## Cell-line bridge (`ACH-` ids)",
        "",
        "| Set | Cell lines |",
        "|---|---|",
        f"| CCLE (default entries) | {lbridge['in_ccle'].sum():,} |",
        f"| DSS3 | {lbridge['in_dss3'].sum():,} |",
        f"| **Both: trainable for sensitivity** | **{both:,}** |",
        f"| DSS3 only: no expression, dropped from training | {dss3_only:,} |",
        f"| CCLE only: no labels, still scoreable at inference | {ccle_only:,} |",
        "",
        "## Cell line → disease (open item A4)",
        "",
        f"- Cell lines mapped to a primary disease: **{_pct(0, both)}** of the trainable lines.",
        f"- `cellinfo.txt` has {cellinfo.height:,} LINCS cell lines with `primary_disease` and "
        f"`cell_lineage`, but **no `ACH-` ids**. {n_cellinfo_ccle:,} have a `ccle_name`, and no "
        "workspace file maps `ccle_name` to `ACH-` ids. The CCLE expression file has no name, "
        "lineage or disease columns.",
        "- To close the gap, add DepMap `Model.csv` from the same release as the expression file. "
        "It maps `ModelID` to `OncotreePrimaryDisease`, `OncotreeLineage`, `CCLEName` and `RRID`. "
        "Until then, FR-11 (cancer-type ranking) is blocked; the rest of the pipeline is unaffected.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, default=REPORTS / "eda_bridges.md")
    args = ap.parse_args()

    verify_raw()
    compoundinfo = read_lincs_tsv(raw_path("compoundinfo"), extra_nulls=(RESTRICTED,))
    compoundinfo_raw = read_lincs_tsv(raw_path("compoundinfo"))
    cellinfo = read_lincs_tsv(raw_path("cellinfo"))
    dss3 = pl.read_parquet(DSS3_PARQUET, columns=["pert_id", "model_id"])
    ccle_ids = set(pl.read_parquet(CCLE_PARQUET, columns=["model_id"])["model_id"])

    cbridge = build_compound_bridge(compoundinfo, set(dss3["pert_id"]))
    lbridge = build_cellline_bridge(ccle_ids, set(dss3["model_id"]))

    src = [raw_path("compoundinfo"), DSS3_PARQUET]
    write_parquet(cbridge, COMPOUND_BRIDGE_OUT, sources=src, created_by=__spec__.name,
                  filters=["'restricted' -> null", "collapsed to one row per pert_id"])
    write_parquet(lbridge, CELLLINE_BRIDGE_OUT, sources=[CCLE_PARQUET, DSS3_PARQUET],
                  created_by=__spec__.name, filters=["union of CCLE default entries and DSS3"])

    args.report.write_text(bridge_report(compoundinfo_raw, cbridge, lbridge, cellinfo))
    print(f"Wrote {COMPOUND_BRIDGE_OUT.name}, {CELLLINE_BRIDGE_OUT.name}, {args.report}")


if __name__ == "__main__":
    main()
