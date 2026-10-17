import numpy as np
import polars as pl
import pytest

from oncora.chem.featurize import CompoundFeatureConfig
from oncora.features.cellline import CellLineFeatureConfig
from oncora.models.sensitivity.aggregate import DiseaseMapMissing, aggregate, load_disease_map
from oncora.models.sensitivity.design import (
    EntityMatrix,
    StructureOnlyViolation,
    assert_structure_only,
    build_design_matrix,
)

CCFG = CompoundFeatureConfig(n_bits=8, descriptors=("MolWt",))
LCFG = CellLineFeatureConfig("landmark", ("G1 (1)", "G2 (2)"), (0.0, 0.0), (1.0, 1.0))


def test_structure_only_accepts_configured_features():
    assert_structure_only(CCFG.feature_names + LCFG.feature_names, CCFG, LCFG)


def test_structure_only_rejects_signature_feature():
    with pytest.raises(StructureOnlyViolation, match="l1000_sig"):
        assert_structure_only(CCFG.feature_names + ["l1000_sig_EGFR"], CCFG, LCFG)


def test_design_matrix_gathers_by_key():
    comps = EntityMatrix(["A", "B"], np.array([[1, 2], [3, 4]]), ["c1", "c2"])
    cells = EntityMatrix(["L1", "L2"], np.array([[10], [20]]), ["e1"])
    rows = pl.DataFrame({"inchi_key": ["B", "A", "B"], "model_id": ["L1", "L2", "L2"]})
    x = build_design_matrix(rows, comps, cells)
    assert x.tolist() == [[3, 4, 10], [1, 2, 20], [3, 4, 20]]
    assert build_design_matrix(rows, comps, None).shape == (3, 2)
    with pytest.raises(KeyError):
        build_design_matrix(pl.DataFrame({"inchi_key": ["Z"], "model_id": ["L1"]}), comps, cells)


def test_aggregate_top_k_low_support_and_unmapped():
    preds = pl.DataFrame({
        "model_id": ["a1", "a2", "a3", "a4", "b1", "x1"],
        "predicted_percentile": [0.9, 0.8, 0.7, 0.1, 0.95, 0.99],
        "p_sensitive": [0.9, 0.6, 0.1, 0.0, 0.8, 0.9],
    })
    dmap = pl.DataFrame({"model_id": ["a1", "a2", "a3", "a4", "b1"],
                         "primary_disease": ["Lung", "Lung", "Lung", "Lung", "Skin"],
                         "lineage": ["Lung", "Lung", "Lung", "Lung", "Skin"]})
    ranked, unmapped = aggregate(preds, dmap, k=3, min_lines=3)
    assert unmapped == 1                                          # x1 has no disease: counted, not guessed
    assert [c.primary_disease for c in ranked] == ["Skin", "Lung"]
    lung = ranked[1]
    assert lung.score == pytest.approx(0.8) and lung.frac_sensitive == pytest.approx(0.5)
    assert ranked[0].low_support and not lung.low_support


def test_disease_map_missing_and_parsing(tmp_path):
    with pytest.raises(DiseaseMapMissing, match="Model.csv"):
        load_disease_map(tmp_path / "Model.csv")
    p = tmp_path / "Model.csv"
    p.write_text("ModelID,CellLineName,OncotreePrimaryDisease,OncotreeLineage\nACH-000001,NIHOVCAR3,Ovarian Epithelial Tumor,Ovary/Fallopian Tube\n")
    m = load_disease_map(p)
    assert m.columns == ["model_id", "cell_line_name", "primary_disease", "lineage"]
    assert m.row(0) == ("ACH-000001", "NIHOVCAR3", "Ovarian Epithelial Tumor", "Ovary/Fallopian Tube")
