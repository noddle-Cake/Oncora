"""Sensitivity target transforms and labels (decisions D-04, D-05).

- `RankTransform`: DSS3 → percentile under the *training* distribution (midpoint ranks for ties,
  so the 20% exact zeros share one value). Fitted on training rows and applied to test rows,
  RMSE on the transform involves no test information. Spearman ρ is unaffected by any monotone
  transform.
- `within_compound_top`: a line is "sensitive" to a compound if it is among that compound's top
  `frac` of lines by DSS3 *and* DSS3 > 0. The > 0 guard stops a compound that is mostly zeros
  from labeling every tied line as sensitive. This label measures selectivity, the product
  question, rather than potency.
- `global_threshold_label`: DSS3 ≥ a training-set quantile (the comparison label).
"""

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class RankTransform:
    sorted_train: np.ndarray | None = None

    def fit(self, y_train: np.ndarray) -> "RankTransform":
        self.sorted_train = np.sort(np.asarray(y_train, dtype=np.float64))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        s = self.sorted_train
        lo = np.searchsorted(s, y, side="left")
        hi = np.searchsorted(s, y, side="right")
        return ((lo + hi) / 2 / len(s)).astype(np.float64)


def within_compound_top(df: pl.DataFrame, frac: float, *, compound: str = "inchi_key",
                        value: str = "dss3") -> pl.Series:
    n = pl.len().over(compound)
    k = (n * frac).ceil()
    rank_desc = pl.col(value).rank(method="min", descending=True).over(compound)
    return df.select(((pl.col(value) > 0) & (rank_desc <= k)).alias("sensitive"))["sensitive"]


def global_threshold(y_train: np.ndarray, quantile: float) -> float:
    return float(np.quantile(y_train, quantile))
