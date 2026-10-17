import polars as pl
import pytest

from oncora.data_prep import sensitivity_table
from oncora.data_prep.sensitivity_table import build_sensitivity

COMPOUNDS = pl.DataFrame({
    "pert_id": ["P1", "P1b", "P2", "PD"],
    "inchi_key": ["AAAAAAAAAAAAAA-UHFFFAOYSA-N", "AAAAAAAAAAAAAA-UHFFFAOYSA-N",
                  "BBBBBBBBBBBBBB-UHFFFAOYSA-N", "DDDDDDDDDDDDDD-UHFFFAOYSA-N"],
    "connectivity": ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "DDDDDDDDDDDDDD"],
    "canonical_smiles": ["CCO", "CCO", "CCN", "CCC"],
})


@pytest.fixture(autouse=True)
def no_assert(monkeypatch):
    monkeypatch.setattr(sensitivity_table, "assert_table", lambda *a, **k: None)


def _dss3(rows):
    return pl.DataFrame(rows, schema=["pert_id", "model_id", "dss3"], orient="row")


def test_merge_by_inchikey_and_mixed_flag_survives():
    dss3 = _dss3([
        ("P1", "L1", 0.2), ("P1b", "L1", 40.0),   # same molecule; P1b is mixed-scale
        ("P1", "L2", 0.1), ("P1b", "L2", 0.3),
        ("P2", "L1", 0.5), ("P2", "L2", 0.0),
        ("P2", "L9", 0.9),                        # L9 has no expression -> dropped
    ])
    table, demo, stats = build_sensitivity(dss3, COMPOUNDS, ["L1", "L2"], holdout={})
    a = table.filter(pl.col("connectivity") == "AAAAAAAAAAAAAA").sort("model_id")
    assert a["pert_ids"].to_list() == [["P1", "P1b"], ["P1", "P1b"]]
    assert a["dss3"].to_list() == pytest.approx([20.1, 0.2])
    assert a["dss3_rescaled"].to_list() == pytest.approx([0.3, 0.2])   # 40 -> 0.4 before the median
    assert a["mixed_scale"].all()
    b = table.filter(pl.col("connectivity") == "BBBBBBBBBBBBBB")
    assert b.height == 2 and not b["mixed_scale"].any()
    assert stats["rows_merged_away"] == 2


def test_demo_holdout_by_connectivity():
    dss3 = _dss3([("P2", "L1", 0.5), ("PD", "L1", 0.4)])
    table, demo, _ = build_sensitivity(dss3, COMPOUNDS, ["L1"], holdout={"DDDDDDDDDDDDDD": "demo"})
    assert table["connectivity"].to_list() == ["BBBBBBBBBBBBBB"]
    assert demo["connectivity"].to_list() == ["DDDDDDDDDDDDDD"]


def test_unknown_pert_id_raises():
    with pytest.raises(ValueError, match="without an InChIKey"):
        build_sensitivity(_dss3([("NOPE", "L1", 0.1)]), COMPOUNDS, ["L1"], holdout={})
