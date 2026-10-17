"""Cell line → cancer type aggregation (spec 06, Req 4; FR-11).

Per-line predictions are rolled up to the disease level: a disease scores the mean predicted
percentile of its top-k lines (default 3), alongside the fraction of its lines predicted
sensitive. Diseases with too few lines are flagged low-support rather than dropped, and lines
without a disease are counted, never guessed.

The disease map comes from DepMap `Model.csv` (decision D-07). Until that file is added,
`load_disease_map` raises `DiseaseMapMissing` and callers return cell-line rankings only.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from oncora.config import CANCER_MIN_LINES, CANCER_TOP_K, DATA_RAW

MODEL_CSV = DATA_RAW / "depmap" / "Model.csv"


class DiseaseMapMissing(FileNotFoundError):
    pass


@dataclass
class CancerTypePrediction:
    primary_disease: str
    lineage: str | None
    score: float
    frac_sensitive: float
    n_cell_lines: int
    low_support: bool
    rank: int


def load_disease_map(path: Path = MODEL_CSV) -> pl.DataFrame:
    """model_id, cell_line_name, primary_disease, lineage from DepMap Model.csv."""
    if not path.exists():
        raise DiseaseMapMissing(
            f"{path} not found: add DepMap Model.csv (same release as the expression data) to enable "
            "cancer-type ranking (decision D-07)."
        )
    m = pl.read_csv(path, infer_schema=False)
    pick = {"ModelID": "model_id", "CellLineName": "cell_line_name",
            "OncotreePrimaryDisease": "primary_disease", "OncotreeLineage": "lineage"}
    missing = [c for c in ("ModelID", "OncotreePrimaryDisease") if c not in m.columns]
    if missing:
        raise ValueError(f"Model.csv lacks expected columns {missing}")
    return m.select([pl.col(c).alias(a) for c, a in pick.items() if c in m.columns])


def aggregate(preds: pl.DataFrame, disease_map: pl.DataFrame, *, k: int = CANCER_TOP_K,
              min_lines: int = CANCER_MIN_LINES, sensitive_threshold: float = 0.5) -> tuple[list[CancerTypePrediction], int]:
    """preds: model_id, predicted_percentile, p_sensitive. Returns (ranked diseases, n unmapped lines)."""
    joined = preds.join(disease_map, on="model_id", how="left")
    unmapped = joined.filter(pl.col("primary_disease").is_null()).height
    lineage = pl.col("lineage").drop_nulls().mode().first() if "lineage" in joined.columns else pl.lit(None)
    agg = (
        joined.filter(pl.col("primary_disease").is_not_null())
        .group_by("primary_disease")
        .agg(
            pl.col("predicted_percentile").sort(descending=True).head(k).mean().alias("score"),
            (pl.col("p_sensitive") >= sensitive_threshold).mean().alias("frac_sensitive"),
            pl.len().alias("n"),
            lineage.alias("lineage"),
        )
        .sort(["score", "primary_disease"], descending=[True, False])
    )
    out = [CancerTypePrediction(r["primary_disease"], r["lineage"], float(r["score"]), float(r["frac_sensitive"]),
                                int(r["n"]), r["n"] < min_lines, i)
           for i, r in enumerate(agg.iter_rows(named=True), start=1)]
    return out, unmapped
