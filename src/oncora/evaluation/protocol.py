"""The evaluation protocol every model and baseline shares (spec 04).

One seeded group holdout per strategy: ~20% of compounds (or scaffolds) are test. A validation
partition (~10% of the training groups, also group-disjoint) is carved out of train for early
stopping and calibration, so test rows are never used for a modeling decision.

Using one fixed holdout instead of k-fold CV is a compute trade-off. The target model is ~1,360
separate classifiers and the sensitivity design matrix is ~470k × 3k. Uncertainty is reported
with compound-level bootstrap CIs on the test partition instead.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from oncora.config import SEED, TEST_FRAC, VAL_FRAC
from oncora.evaluation.splits import SplitManifest, assert_disjoint, holdout, scaffold_groups

STRATEGIES = ("compound", "scaffold")


@dataclass
class Partition:
    strategy: str
    train: np.ndarray   # boolean masks over the table's rows
    val: np.ndarray
    test: np.ndarray
    manifest: SplitManifest

    @property
    def fit(self) -> np.ndarray:
        """train ∪ val, for a final refit once hyperparameters are fixed."""
        return self.train | self.val


def split_groups(df: pl.DataFrame, strategy: str) -> np.ndarray:
    if strategy == "compound":
        return df["group"].to_numpy()
    if strategy == "scaffold":
        return scaffold_groups(df["scaffold"], df["group"]).to_numpy()
    raise ValueError(f"unknown strategy {strategy!r}")


def make_partition(df: pl.DataFrame, strategy: str, seed: int = SEED,
                   test_frac: float = TEST_FRAC, val_frac: float = VAL_FRAC) -> Partition:
    groups = split_groups(df, strategy)
    rest, test, manifest = holdout(groups, test_frac=test_frac, seed=seed, strategy=strategy)
    train = rest.copy()
    _, val_sub, _ = holdout(groups[rest], test_frac=val_frac, seed=seed + 1, strategy=strategy)
    val = np.zeros(len(df), dtype=bool)
    val[np.flatnonzero(rest)[val_sub]] = True
    train &= ~val

    labels = np.select([train, val, test], [0, 1, 2], default=-1)
    assert_disjoint(labels, groups, f"{strategy} group")
    assert_disjoint(labels, df["group"].to_numpy(), "compound")   # scaffold splits nest compounds
    return Partition(strategy, train, val, test, manifest)
