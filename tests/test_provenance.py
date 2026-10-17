import polars as pl
import pytest

from oncora.data_prep import provenance
from oncora.data_prep.provenance import (
    DataIntegrityError,
    assert_table,
    read_provenance,
    write_parquet,
)


def test_provenance_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    df = pl.DataFrame({"a": [1, 2]})
    src = tmp_path / "upstream.parquet"
    src.touch()
    out = write_parquet(df, tmp_path / "x.parquet", sources=[src], created_by="test",
                        filters=["a > 0"], config={"seed": 42})
    meta = read_provenance(out)
    assert meta["created_by"] == "test"
    assert meta["filters"] == ["a > 0"]
    assert meta["config"] == {"seed": 42}
    assert meta["shape"] == [2, 1]
    assert meta["sources"] == [{"path": "upstream.parquet", "sha256": None}]
    assert pl.read_parquet(out).equals(df)


def test_missing_provenance_raises(tmp_path):
    p = tmp_path / "plain.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(p)
    with pytest.raises(DataIntegrityError, match="no Oncora provenance"):
        read_provenance(p)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"required_cols": ["b"]}, "missing columns"),
        ({"required_cols": ["a"], "min_rows": 10}, "expected at least 10"),
        ({"required_cols": ["a"], "max_null_rate": {"a": 0.0}}, "null rate"),
        ({"required_cols": ["a"], "unique_key": ["a"]}, "duplicate rows"),
    ],
)
def test_assert_table_failures(kwargs, match):
    df = pl.DataFrame({"a": [1, 1, None]})
    with pytest.raises(DataIntegrityError, match=match):
        assert_table(df, name="t", **kwargs)


def test_assert_table_passes():
    assert_table(pl.DataFrame({"a": [1, 2]}), name="t", required_cols=["a"],
                 max_null_rate={"a": 0.0}, unique_key=["a"])
