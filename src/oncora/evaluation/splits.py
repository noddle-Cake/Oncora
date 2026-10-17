"""Leakage-safe splitters (spec 04, Req 1–3; NFR-1, AC-2).

The central rule: a compound, meaning every row that shares its standardized connectivity block
(salt forms and stereoisomers included, D-01), lives in exactly one partition. A random row
split would put the same molecule on both sides, and the model would score well by remembering
it. That is the headline failure this module exists to prevent.

Two strategies share one assignment routine:
- **compound:** group = standardized connectivity block. Measures generalization to new compounds.
- **scaffold:** group = Bemis–Murcko scaffold, stereo removed. Acyclic compounds fall back to their
  connectivity block. A harder test: new chemotypes, not just new analogs of seen ones.

Groups are assigned to folds largest-first, each to the currently smallest fold (by rows), after
a seeded shuffle. Folds come out balanced in size even when one scaffold is huge.
"""

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import polars as pl
from rdkit import Chem, rdBase

Strategy = Literal["compound", "scaffold"]


class LeakageError(AssertionError):
    pass


@dataclass(frozen=True)
class SplitManifest:
    strategy: str
    seed: int
    n_folds: int
    group_counts: list[int]
    row_counts: list[int]
    partition_hashes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def strip_stereo(smiles: str) -> str:
    if not smiles:
        return smiles
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else smiles


def scaffold_groups(scaffolds: pl.Series, connectivity: pl.Series) -> pl.Series:
    """Stereo-free scaffold per row; acyclic rows (empty scaffold) group by connectivity."""
    uniq = scaffolds.unique().drop_nulls().to_list()
    mapping = {s: strip_stereo(s) for s in uniq}
    flat = scaffolds.replace_strict(mapping, default=None, return_dtype=pl.String)
    return (
        pl.DataFrame({"s": flat, "c": connectivity})
        .select(pl.when(pl.col("s").is_null() | (pl.col("s") == "")).then("acyclic:" + pl.col("c"))
                .otherwise("scaffold:" + pl.col("s")).alias("g"))["g"]
    )


def _assign(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    uniq, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    order = order[np.argsort(-sizes[order], kind="stable")]  # largest first; shuffle breaks ties
    fold_of_group = np.empty(len(uniq), dtype=np.int16)
    load = np.zeros(n_folds, dtype=np.int64)
    for g in order:
        f = int(np.argmin(load))
        fold_of_group[g] = f
        load[f] += sizes[g]
    return fold_of_group[inverse]


def _manifest(strategy: str, seed: int, n_folds: int, groups: np.ndarray, folds: np.ndarray) -> SplitManifest:
    gc, rc, hashes = [], [], []
    for f in range(n_folds):
        g = np.unique(groups[folds == f])
        gc.append(int(len(g)))
        rc.append(int((folds == f).sum()))
        hashes.append(hashlib.sha256("\n".join(g.astype(str)).encode()).hexdigest()[:16])
    return SplitManifest(strategy, seed, n_folds, gc, rc, hashes)


def group_folds(groups, *, n_folds: int, seed: int, strategy: str = "compound") -> tuple[np.ndarray, SplitManifest]:
    """Fold id (0..n_folds-1) per row such that every group sits in exactly one fold."""
    groups = np.asarray(groups)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(np.unique(groups)) < n_folds:
        raise ValueError(f"only {len(np.unique(groups))} groups for {n_folds} folds")
    folds = _assign(groups, n_folds, seed)
    return folds, _manifest(strategy, seed, n_folds, groups, folds)


def holdout(groups, *, test_frac: float, seed: int, strategy: str = "compound") -> tuple[np.ndarray, np.ndarray, SplitManifest]:
    """Boolean train/test masks; the test partition is ~test_frac of rows, group-disjoint."""
    n_folds = max(2, round(1 / test_frac))
    folds, manifest = group_folds(groups, n_folds=n_folds, seed=seed, strategy=strategy)
    test = folds == 0
    return ~test, test, manifest


def assert_disjoint(folds: np.ndarray, keys, name: str = "key") -> None:
    """Raise LeakageError if any key value appears in more than one fold."""
    df = pl.DataFrame({"fold": np.asarray(folds), "key": np.asarray(keys)})
    leaky = df.group_by("key").agg(pl.col("fold").n_unique().alias("n")).filter(pl.col("n") > 1)
    if leaky.height:
        raise LeakageError(
            f"{leaky.height} {name} values occur in more than one fold, e.g. {leaky['key'].head(3).to_list()}"
        )
