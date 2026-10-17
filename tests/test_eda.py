import numpy as np
import polars as pl
import pytest

from oncora.eda.target import mixed_scale_compounds, variance_decomposition


def _grid(compound_effect, line_effect, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = [
        {"pert_id": f"c{i}", "model_id": f"l{j}", "y": ce + le + noise * rng.standard_normal()}
        for i, ce in enumerate(compound_effect)
        for j, le in enumerate(line_effect)
    ]
    return pl.DataFrame(rows)


def test_decomposition_pure_compound_effect():
    d = variance_decomposition(_grid([0, 1, 2, 3], [0, 0, 0]), "y")
    assert d["compound"] == pytest.approx(1.0)
    assert d["cell_line"] == pytest.approx(0.0)


def test_decomposition_additive_recovers_both():
    d = variance_decomposition(_grid([0, 2, 4], [0, 1]), "y")
    assert d["additive"] == pytest.approx(1.0)
    assert d["compound"] > d["cell_line"] > 0


def test_decomposition_with_missing_cells():
    df = _grid([0, 2, 4, 6], [0, 1, 3], noise=0.0).filter(
        ~((pl.col("pert_id") == "c0") & (pl.col("model_id") == "l2"))
    )
    assert variance_decomposition(df, "y")["additive"] == pytest.approx(1.0, abs=1e-6)


def test_mixed_scale_detection():
    df = pl.DataFrame(
        {
            "pert_id": ["a", "a", "b", "b"],
            "name": ["A", "A", "B", "B"],
            "model_id": ["l1", "l2", "l1", "l2"],
            "dss3": [0.1, 0.2, 0.3, 45.0],
        }
    )
    m = mixed_scale_compounds(df)
    assert m["pert_id"].to_list() == ["b"]
    assert m["frac_high"][0] == pytest.approx(0.5)
