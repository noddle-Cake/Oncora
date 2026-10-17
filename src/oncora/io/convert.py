"""Convert the large raw CSVs to interim Parquet once (spec 01, task 3.3).

Parsing the 305 MB, 19k-column CCLE CSV takes seconds with Polars, but every later stage would
pay that cost and repeat the default-entry filtering. Parquet is columnar and typed, so a stage
can load only the genes it needs.
"""

import re

import polars as pl

from oncora.config import DATA_INTERIM, raw_path
from oncora.data_prep.provenance import assert_table, write_parquet
from oncora.io.checksums import verify_raw
from oncora.io.profile import CCLE_META_COLS

_GENE_COL = re.compile(r"^(?P<symbol>.+) \((?P<entrez>\d+)\)$")

CCLE_OUT = DATA_INTERIM / "ccle_expression.parquet"
CCLE_GENES_OUT = DATA_INTERIM / "ccle_genes.parquet"
DSS3_OUT = DATA_INTERIM / "sensitivity.parquet"


def ccle_to_parquet() -> pl.DataFrame:
    """One row per cell line: CCLE has several sequencing profiles for some models, and
    IsDefaultEntryForModel marks DepMap's chosen one."""
    src = raw_path("ccle")
    df = pl.read_csv(src)
    gene_cols = df.columns[CCLE_META_COLS:]
    out = (
        df.filter(pl.col("IsDefaultEntryForModel") == "Yes")
        .select(
            pl.col("ModelID").alias("model_id"),
            pl.col("SequencingID").alias("sequencing_id"),
            pl.col("ModelConditionID").alias("model_condition_id"),
            *gene_cols,
        )
        .sort("model_id")
    )
    assert_table(out, name="ccle_expression", required_cols=["model_id"], min_rows=1000,
                 unique_key=["model_id"])

    genes = pl.DataFrame(
        [
            {"gene_col": c, "gene_symbol": m["symbol"], "entrez_id": int(m["entrez"])}
            for c in gene_cols
            if (m := _GENE_COL.match(c))
        ]
    )
    if genes.height != len(gene_cols):
        raise ValueError(f"{len(gene_cols) - genes.height} CCLE gene columns are not SYMBOL (ENTREZID)")

    filters = ["IsDefaultEntryForModel == 'Yes'", "dropped unnamed index column"]
    write_parquet(out, CCLE_OUT, sources=[src], created_by=__spec__.name, filters=filters)
    write_parquet(genes, CCLE_GENES_OUT, sources=[src], created_by=__spec__.name)
    return out


def dss3_to_parquet() -> pl.DataFrame:
    src = raw_path("dss3")
    out = (
        pl.read_csv(src)
        .rename({"cell_line": "model_id", "sensitivity_score": "dss3"})
        .sort("pert_id", "model_id")
    )
    assert_table(out, name="sensitivity", required_cols=["pert_id", "model_id", "dss3"],
                 min_rows=100_000, max_null_rate={"dss3": 0.0}, unique_key=["pert_id", "model_id"])
    write_parquet(out, DSS3_OUT, sources=[src], created_by=__spec__.name,
                  filters=["renamed cell_line->model_id, sensitivity_score->dss3"])
    return out


def main() -> None:
    verify_raw()
    ccle = ccle_to_parquet()
    print(f"Wrote {CCLE_OUT.name}: {ccle.height:,} cell lines × {ccle.width - 3:,} genes")
    dss3 = dss3_to_parquet()
    print(f"Wrote {DSS3_OUT.name}: {dss3.height:,} rows")


if __name__ == "__main__":
    main()
