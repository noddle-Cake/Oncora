"""Modeling-ready views of the processed tables, shared by baselines and models (specs 04–06).

Both functions attach the split keys from compound_features.parquet:
- `group`: connectivity block of the standardized parent (D-01);
- `scaffold`: Murcko scaffold of the standardized parent (spec 04 scaffold split).

Keeping the row selection here guarantees that baselines and models score exactly the same rows.
"""

import polars as pl

from oncora.config import ACTIVE_PCHEMBL, DATA_PROCESSED

FEATURES = DATA_PROCESSED / "compound_features.parquet"


def _split_keys() -> pl.DataFrame:
    return pl.read_parquet(FEATURES, columns=["inchi_key", "ok", "std_inchi_key", "scaffold"]).filter("ok").select(
        "inchi_key", pl.col("std_inchi_key").str.slice(0, 14).alias("group"), "scaffold"
    )


def collapse_target_pairs(df: pl.DataFrame, active: float = ACTIVE_PCHEMBL) -> pl.DataFrame:
    """One row per (group, target) (spec 05, Req 1.6).

    Rows sharing a standardized parent have identical features, so duplicates add no information
    and can disagree. Measured rows (positives and measured negatives) take precedence over
    sampled negatives. The label is recomputed from the median measured pChEMBL; censored-only
    groups stay inactive.
    """
    df = df.with_columns((pl.col("negative_source") == "sampled").fill_null(False).alias("_sampled"))
    has_measured = (~pl.col("_sampled")).any().over("group", "target_chembl_id")
    kept = df.filter(~(pl.col("_sampled") & has_measured))
    return (
        kept.group_by("group", "target_chembl_id")
        .agg(
            pl.col("inchi_key").min(),
            pl.col("scaffold").first(),
            pl.col("pchembl_median").median().alias("pchembl"),
            pl.col("_sampled").all().alias("sampled"),
            pl.len().alias("n_merged"),
        )
        .with_columns(
            pl.when(pl.col("pchembl") >= active).then(1).otherwise(0).cast(pl.Int8).alias("label"),
            pl.when(pl.col("sampled")).then(pl.lit("sampled"))
            .when(pl.col("pchembl") >= active).then(None)
            .otherwise(pl.lit("measured")).alias("negative_source"),
        )
        .drop("sampled")
        .sort("target_chembl_id", "group")
    )


def target_training_table() -> pl.DataFrame:
    tp = pl.read_parquet(DATA_PROCESSED / "target_pairs.parquet",
                         columns=["inchi_key", "target_chembl_id", "pchembl_median", "label", "negative_source"])
    return collapse_target_pairs(tp.join(_split_keys(), on="inchi_key", how="inner"))


def sensitivity_training_table(include_mixed_scale: bool = False) -> pl.DataFrame:
    """Headline set by default (D-03). With include_mixed_scale, `dss3` is replaced by the
    ×0.01-rescaled values for the sensitivity analysis."""
    s = pl.read_parquet(DATA_PROCESSED / "sensitivity.parquet")
    if include_mixed_scale:
        s = s.with_columns(pl.col("dss3_rescaled").alias("dss3"))
    else:
        s = s.filter(~pl.col("mixed_scale"))
    return s.join(_split_keys(), on="inchi_key", how="inner").sort("inchi_key", "model_id")
