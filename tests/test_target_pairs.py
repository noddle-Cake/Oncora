import polars as pl
import pytest

from oncora.data_prep.target_pairs import build_target_pairs


def _ik(i: int, stereo: str = "UHFFFAOYSA") -> str:
    """Fake InChIKey whose first 14 chars (connectivity) depend only on i."""
    return f"C{i:013d}-{stereo}-N"


def _act(rows):
    return pl.DataFrame(
        [
            {"inchi_key": ik, "connectivity": ik[:14], "canonical_smiles": f"C{n}",
             "target_chembl_id": t, "target_name": t.lower(), "pchembl_median": p, "censored_bound": cb}
            for n, (ik, t, p, cb) in enumerate(rows)
        ],
        schema_overrides={"pchembl_median": pl.Float64, "censored_bound": pl.Float64},
    )


@pytest.fixture
def act():
    rows = []
    # T1: 4 actives (compounds 0-3), 1 measured inactive (4), 1 censored-only (5)
    rows += [(_ik(i), "T1", 7.0, None) for i in range(4)]
    rows += [(_ik(4), "T1", 5.0, None), (_ik(5), "T1", None, 4.5)]
    # T2: 4 actives (compounds 10-13), plenty of pool overlap with T1 compounds
    rows += [(_ik(i), "T2", 6.5, None) for i in range(10, 14)]
    # T3: only 1 active -> dropped by min_pos
    rows += [(_ik(20), "T3", 8.0, None)]
    # stereoisomer of compound 10 measured on T1 (inactive): compound 10 must not be sampled for T1
    rows += [(_ik(10, "ABCDEFGHIJ"), "T1", 4.0, None)]
    # demo compound
    rows += [(_ik(99), "T1", 9.0, None), (_ik(99), "T2", 5.0, None)]
    return _act(rows)


ANNOT = pl.DataFrame({"target_chembl_id": ["T1", "T2", "T3"], "target_gene_symbol": ["G1", "G2", "G3"],
                      "uniprot_accession": ["P1", "P2", "P3"]})


def _build(act, seed=42):
    return build_target_pairs(act, ANNOT, active=6.0, min_pos=2, neg_ratio=3, seed=seed,
                              holdout={_ik(99)[:14]: "demo"})


def test_labels_and_sources(act, monkeypatch):
    monkeypatch.setattr("oncora.data_prep.target_pairs.assert_table", lambda *a, **k: None)
    pairs, demo, stats = _build(act)
    t1 = pairs.filter(pl.col("target_chembl_id") == "T1")
    assert (t1["label"] == 1).sum() == 4
    assert set(t1.filter(pl.col("negative_source") == "measured")["connectivity"]) == {
        _ik(4)[:14], _ik(5)[:14], _ik(10)[:14]}
    assert t1["target_gene_symbol"].unique().to_list() == ["G1"]
    assert stats["targets_kept"] == 2 and stats["targets_dropped_min_pos"] == 1


def test_sampled_negatives_respect_exclusions(act, monkeypatch):
    monkeypatch.setattr("oncora.data_prep.target_pairs.assert_table", lambda *a, **k: None)
    pairs, _, _ = _build(act)
    sampled_t1 = pairs.filter((pl.col("target_chembl_id") == "T1") & (pl.col("negative_source") == "sampled"))
    # need = 3*4 - 3 = 9, but pool minus exclusions is only compounds 11-13 (10 is a measured stereoisomer)
    assert set(sampled_t1["connectivity"]) == {_ik(i)[:14] for i in (11, 12, 13)}
    assert sampled_t1["pchembl_median"].null_count() == sampled_t1.height


def test_demo_holdout_removed(act, monkeypatch):
    monkeypatch.setattr("oncora.data_prep.target_pairs.assert_table", lambda *a, **k: None)
    pairs, demo, stats = _build(act)
    assert _ik(99)[:14] not in set(pairs["connectivity"])
    assert demo.height == 2 and stats["demo_pairs_removed"] == 2


def test_sampling_is_deterministic(act, monkeypatch):
    monkeypatch.setattr("oncora.data_prep.target_pairs.assert_table", lambda *a, **k: None)
    a, _, _ = _build(act, seed=1)
    b, _, _ = _build(act, seed=1)
    assert a.equals(b)
