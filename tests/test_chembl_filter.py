import polars as pl
import pytest

from oncora.data_prep import chembl_filter

IK_A = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"
IK_B = "BBBBBBBBBBBBBB-UHFFFAOYSA-N"


def _activity(aid, molregno, relation, value, pchembl, *, assay=1, comment=None, dup=0,
              units="nM", stype="IC50"):
    return {"activity_id": aid, "assay_id": assay, "doc_id": aid, "molregno": molregno,
            "standard_type": stype, "standard_relation": relation, "standard_value": value,
            "standard_units": units, "pchembl_value": pchembl,
            "data_validity_comment": comment, "potential_duplicate": dup}


@pytest.fixture
def chembl_dir(tmp_path, monkeypatch):
    acts = [
        # compound A on target T1: three measured values -> median 7.0 (active)
        _activity(1, 1, "=", 10.0, 8.0),
        _activity(2, 1, "=", 100.0, 7.0),
        _activity(3, 1, "=", 1000.0, 6.0),
        # censored row for the same pair is ignored because measured values exist
        _activity(4, 1, ">", 30000.0, None),
        # compound B on T1: '>' 10 µM -> implied 5.0 < 6 -> censored inactive
        _activity(5, 2, ">", 10000.0, None),
        # compound B: '>' 100 nM -> implied 7.0 >= 6 -> no label (dropped)
        _activity(6, 2, ">", 100.0, None, assay=2),
        # '<' is ambiguous -> dropped
        _activity(7, 2, "<", 10.0, None, assay=2),
        # flagged values dropped; 'Manually validated' kept
        _activity(8, 2, "=", 1.0, 9.0, assay=2, comment="Outside typical range"),
        _activity(9, 2, "=", 50.0, 7.3, assay=2, comment="Manually validated"),
        _activity(10, 2, "=", 50.0, 7.3, assay=2, dup=1),
        # wrong assay type / low confidence / non-human target -> dropped by scope
        _activity(11, 1, "=", 5.0, 8.3, assay=3),
        _activity(12, 1, "=", 5.0, 8.3, assay=4),
        _activity(13, 1, "=", 5.0, 8.3, assay=5),
    ]
    tables = {
        "activities": pl.DataFrame(acts, schema_overrides={"pchembl_value": pl.Float64}),
        "assays": pl.DataFrame({"assay_id": [1, 2, 3, 4, 5], "assay_type": ["B", "F", "A", "B", "B"],
                                "confidence_score": [9, 8, 9, 5, 9], "tid": [10, 20, 10, 10, 30]}),
        "target_dictionary": pl.DataFrame({
            "tid": [10, 20, 30], "chembl_id": ["T1", "T2", "T3"], "pref_name": ["t1", "t2", "t3"],
            "target_type": ["SINGLE PROTEIN"] * 3, "organism": ["Homo sapiens", "Homo sapiens", "Mus musculus"]}),
        "compound_structures": pl.DataFrame({"molregno": [1, 2], "standard_inchi_key": [IK_A, IK_B],
                                             "canonical_smiles": ["CCO", "CCN"]}),
        "molecule_dictionary": pl.DataFrame({"molregno": [1, 2], "chembl_id": ["CHEMBL_A", "CHEMBL_B"]}),
    }
    for name, df in tables.items():
        df.write_parquet(tmp_path / f"{name}.parquet")
    monkeypatch.setattr(chembl_filter, "CHEMBL_DIR", tmp_path)
    monkeypatch.setattr(chembl_filter, "assert_table", lambda *a, **k: None)
    return tmp_path


def _pairs(chembl_dir):
    pairs, steps, stats = chembl_filter.filter_activities(min_confidence=8, active=6.0)
    return {(r["inchi_key"], r["target_chembl_id"]): r for r in pairs.iter_rows(named=True)}, steps


def test_measured_median_wins_over_censored(chembl_dir):
    pairs, _ = _pairs(chembl_dir)
    a = pairs[(IK_A, "T1")]
    assert a["pchembl_median"] == pytest.approx(7.0)
    assert a["n_measured"] == 3 and a["n_censored"] == 1
    assert a["connectivity"] == IK_A[:14]


def test_censored_inactive_kept_only_below_threshold(chembl_dir):
    pairs, _ = _pairs(chembl_dir)
    b = pairs[(IK_B, "T1")]
    assert b["pchembl_median"] is None
    assert b["censored_bound"] == pytest.approx(5.0)


def test_quality_and_relation_filters(chembl_dir):
    pairs, _ = _pairs(chembl_dir)
    b2 = pairs[(IK_B, "T2")]
    # only the 'Manually validated' row survives: flagged, duplicate, '<' and high-bound '>' dropped
    assert b2["n_measured"] == 1 and b2["pchembl_median"] == pytest.approx(7.3)
    assert b2["n_censored"] == 0
    assert set(pairs) == {(IK_A, "T1"), (IK_B, "T1"), (IK_B, "T2")}


def test_funnel_is_monotone(chembl_dir):
    _, steps = _pairs(chembl_dir)
    counts = [n for _, n in steps]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 13
