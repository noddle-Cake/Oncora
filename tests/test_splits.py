"""Anti-leakage suite (spec 04, Req 3). These tests guard the headline metrics."""

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import KFold

from oncora.evaluation.splits import (
    LeakageError,
    assert_disjoint,
    group_folds,
    holdout,
    scaffold_groups,
    strip_stereo,
)


def _synthetic(seed: int, n_groups: int = 300):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 40, n_groups)              # compounds measured on 1..40 rows
    groups = np.repeat([f"G{i:04d}" for i in range(n_groups)], sizes)
    return groups


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_folds", [2, 5, 10])
def test_group_folds_never_leak(seed, n_folds):
    groups = _synthetic(seed)
    folds, manifest = group_folds(groups, n_folds=n_folds, seed=seed)
    assert_disjoint(folds, groups, "compound")
    assert set(folds) == set(range(n_folds))
    assert sum(manifest.row_counts) == len(groups)
    # greedy assignment keeps folds balanced within one large group
    assert max(manifest.row_counts) - min(manifest.row_counts) <= 40


def test_random_row_split_is_detected_as_leaky():
    """Negative control: the check must catch the naive split it is meant to prevent."""
    groups = _synthetic(0)
    folds = np.empty(len(groups), dtype=int)
    for f, (_, test) in enumerate(KFold(5, shuffle=True, random_state=0).split(groups)):
        folds[test] = f
    with pytest.raises(LeakageError):
        assert_disjoint(folds, groups, "compound")


def test_full_inchikey_split_leaks_stereoisomers():
    """Negative control: dexamethasone / betamethasone share a connectivity block."""
    full = np.array(["UREBDLICKHMUKA-CXSFZGCWSA-N", "UREBDLICKHMUKA-DVTGEIKXSA-N", "AAAAAAAAAAAAAA-UHFFFAOYSA-N"])
    folds_by_full_key = np.array([0, 1, 1])        # each full key in its own place: looks fine...
    assert_disjoint(folds_by_full_key, full, "inchi_key")
    with pytest.raises(LeakageError):              # ...but the connectivity blocks leak
        assert_disjoint(folds_by_full_key, np.array([k[:14] for k in full]), "connectivity")


def test_same_seed_same_split():
    groups = _synthetic(1)
    a, ma = group_folds(groups, n_folds=5, seed=7)
    b, mb = group_folds(groups, n_folds=5, seed=7)
    c, _ = group_folds(groups, n_folds=5, seed=8)
    assert (a == b).all() and ma == mb
    assert not (a == c).all()


def test_holdout_fraction_and_disjointness():
    groups = _synthetic(2)
    train, test, m = holdout(groups, test_frac=0.2, seed=0)
    assert 0.15 < test.mean() < 0.25
    assert not set(groups[train]) & set(groups[test])


def test_scaffold_groups_strip_stereo_and_handle_acyclic():
    scaf = pl.Series(["C1CC[C@H](C)CC1", "C1CC[C@@H](C)CC1", "", "c1ccccc1"])
    conn = pl.Series(["A" * 14, "B" * 14, "C" * 14, "D" * 14])
    g = scaffold_groups(scaf, conn).to_list()
    assert g[0] == g[1]                             # stereo-only difference -> same scaffold group
    assert g[2] == "acyclic:" + "C" * 14            # acyclic -> its own compound group
    assert strip_stereo("C[C@H](N)O") == "CC(N)O"


def test_scaffold_split_disjoint_on_both_keys():
    rng = np.random.default_rng(3)
    conn = np.array([f"C{i:03d}" for i in rng.integers(0, 200, 2000)])
    scaffold_of = {c: f"S{int(c[1:]) % 37}" for c in np.unique(conn)}
    scaf = np.array([scaffold_of[c] for c in conn])
    folds, _ = group_folds(scaf, n_folds=5, seed=0, strategy="scaffold")
    assert_disjoint(folds, scaf, "scaffold")
    assert_disjoint(folds, conn, "connectivity")    # scaffold groups nest compounds


def test_too_few_groups_raises():
    with pytest.raises(ValueError):
        group_folds(np.array(["a", "a", "b"]), n_folds=5, seed=0)
