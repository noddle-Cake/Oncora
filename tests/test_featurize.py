import numpy as np
import polars as pl
import pytest

from oncora.chem.featurize import CompoundFeatureConfig, featurize_mols, murcko_scaffold
from oncora.chem.validate import parse_smiles
from oncora.features import compound
from oncora.features.compound import feature_matrix, featurize_table

CFG = CompoundFeatureConfig(n_bits=256)
ASPIRIN, ASPIRIN_ALT = "CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1OC(C)=O"


def test_equivalent_smiles_give_identical_features():
    a, b = parse_smiles(ASPIRIN).mol, parse_smiles(ASPIRIN_ALT).mol
    fa, fb = featurize_mols([a, b], CFG)
    assert fa.shape == (CFG.n_features,)
    np.testing.assert_array_equal(fa, fb)
    assert set(np.unique(fa[: CFG.n_bits])) <= {0.0, 1.0}


def test_features_are_deterministic():
    m = parse_smiles("c1ccc2[nH]ccc2c1").mol
    np.testing.assert_array_equal(featurize_mols([m], CFG), featurize_mols([m], CFG))


@pytest.mark.parametrize("smiles, scaffold", [
    ("CCCCO", ""),
    ("CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"),
    ("Cc1ccc(CCc2ccncc2)cc1", "c1ccc(CCc2ccncc2)cc1"),
])
def test_murcko_scaffold(smiles, scaffold):
    assert murcko_scaffold(parse_smiles(smiles).mol) == scaffold


def test_config_round_trip():
    assert CompoundFeatureConfig.from_dict(CFG.to_dict()) == CFG
    assert len(CFG.feature_names) == CFG.n_features


def test_featurize_table_caches_and_aligns(tmp_path, monkeypatch):
    monkeypatch.setattr("oncora.data_prep.provenance.ROOT", tmp_path)
    df = pl.DataFrame({"inchi_key": ["K2", "K1", "K3"], "canonical_smiles": [ASPIRIN, "CCO", "not_smiles"]})
    cache = tmp_path / "f.parquet"
    t1 = featurize_table(df, CFG, cache, n_jobs=1)
    assert t1.filter(pl.col("inchi_key") == "K3")["ok"].to_list() == [False]

    calls = []
    real = compound._featurize_parallel
    monkeypatch.setattr(compound, "_featurize_parallel", lambda s, c, n: calls.append(len(s)) or real(s, c, n))
    featurize_table(df.vstack(pl.DataFrame({"inchi_key": ["K4"], "canonical_smiles": ["CCN"]})), CFG, cache, n_jobs=1)
    assert calls == [1]  # only the new key is featurized

    calls.clear()
    again = featurize_table(df, CFG, cache, n_jobs=1)  # everything cached: no work, no rewrite
    assert calls == [] and sorted(again["inchi_key"]) == ["K1", "K2", "K3"]

    m = feature_matrix(pl.read_parquet(cache), ["K2", "K1"], CFG)
    np.testing.assert_array_equal(m[0], featurize_mols([parse_smiles(ASPIRIN).mol], CFG)[0])
    with pytest.raises(KeyError):
        feature_matrix(pl.read_parquet(cache), ["K3"], CFG)
