"""Trivial baselines (spec 04, Req 4; AC-3). Every model must beat these on the same split.

All are fit on training rows only and share a fit/predict interface with the models.
"""

import numpy as np
import polars as pl


class PriorBaseline:
    """Target: score every (compound, target) pair with the target's training positive rate.
    Within one target the score is constant, so per-target AUC is 0.5 by construction. It
    calibrates the floor and anchors top-k (popular targets rank first)."""

    def fit(self, train: pl.DataFrame) -> "PriorBaseline":
        self.prior = dict(train.group_by("target_chembl_id").agg(pl.col("label").mean()).iter_rows())
        self.default = float(train["label"].mean())
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return df["target_chembl_id"].replace_strict(self.prior, default=self.default, return_dtype=pl.Float64).to_numpy()


class PopularityRanker:
    """Target top-k: rank targets by training positive count, identically for every compound."""

    def fit(self, train: pl.DataFrame) -> "PopularityRanker":
        counts = train.filter(pl.col("label") == 1).group_by("target_chembl_id").len()
        self.score = dict(counts.iter_rows())
        return self

    def score_matrix(self, n_compounds: int, targets: list[str]) -> np.ndarray:
        row = np.array([self.score.get(t, 0) for t in targets], dtype=np.float64)
        return np.broadcast_to(row, (n_compounds, len(targets)))


class GlobalMeanBaseline:
    def fit(self, train: pl.DataFrame, y: np.ndarray) -> "GlobalMeanBaseline":
        self.mean = float(np.mean(y))
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.full(df.height, self.mean)


class CellLineMeanBaseline:
    """Sensitivity: each cell line's mean (transformed) response on training rows; unseen lines get
    the global mean. It gives every compound the same cell-line ranking: the bar a model must
    clear to claim it has learned selectivity."""

    def fit(self, train: pl.DataFrame, y: np.ndarray) -> "CellLineMeanBaseline":
        means = pl.DataFrame({"model_id": train["model_id"], "y": y}).group_by("model_id").agg(pl.col("y").mean())
        self.means = dict(means.iter_rows())
        self.default = float(np.mean(y))
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return df["model_id"].replace_strict(self.means, default=self.default, return_dtype=pl.Float64).to_numpy()
