import numpy as np
import polars as pl
import pytest

from oncora.chem.featurize import CompoundFeatureConfig, fingerprint_bits
from oncora.chem.similarity import TrainingIndex, pack_bits, tanimoto
from oncora.chem.validate import parse_smiles
from oncora.features import cellline

CFG = CompoundFeatureConfig()


def _packed(smiles):
    return pack_bits(fingerprint_bits(parse_smiles(smiles).mol, CFG))


def test_tanimoto_matches_definition():
    a = np.array([1, 1, 0, 0, 1, 0, 0, 0] * 2, dtype=np.uint8)
    b = np.array([1, 0, 1, 0, 1, 0, 0, 0] * 2, dtype=np.uint8)
    sim = tanimoto(pack_bits(a), pack_bits(np.vstack([a, b])))
    assert sim.tolist() == pytest.approx([1.0, 2 / 4])


def test_nearest_and_near_seen(tmp_path):
    train = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1", "CCN(CC)CC"]
    idx = TrainingIndex(["ASP", "IND", "TEA"], np.vstack([_packed(s) for s in train]))
    hit = idx.nearest(_packed("OC(=O)c1ccccc1OC(C)=O"))     # aspirin written differently
    assert hit.inchi_key == "ASP" and hit.max_tanimoto == pytest.approx(1.0) and hit.near_seen
    miss = idx.nearest(_packed("CC(=O)Nc1ccc(O)cc1"))       # paracetamol: related, not identical
    assert not miss.near_seen and miss.max_tanimoto < 0.99
    idx.save(tmp_path / "i.npz")
    again = TrainingIndex.load(tmp_path / "i.npz")
    assert again.nearest(_packed(train[1])).inchi_key == "IND"


@pytest.fixture
def fake_ccle(tmp_path, monkeypatch):
    df = pl.DataFrame({
        "model_id": ["L1", "L2", "L3", "L4"],
        "G1 (1)": [1.0, 3.0, 5.0, 100.0],    # L4 is a test line with an extreme value
        "G2 (2)": [2.0, 2.0, 2.0, 9.0],      # constant on training lines
        "G3 (3)": [0.0, 1.0, 0.0, 0.0],
    })
    df.write_parquet(tmp_path / "ccle.parquet")
    pl.DataFrame({"gene_col": ["G1 (1)", "G2 (2)", "G3 (3)"], "gene_symbol": ["G1", "G2", "G3"],
                  "entrez_id": [1, 2, 3]}).write_parquet(tmp_path / "genes.parquet")
    monkeypatch.setattr(cellline, "CCLE_PARQUET", tmp_path / "ccle.parquet")
    monkeypatch.setattr(cellline, "CCLE_GENES", tmp_path / "genes.parquet")
    monkeypatch.setattr(cellline, "landmark_gene_columns", lambda: ("G1 (1)", "G2 (2)"))


def test_scaling_fit_on_training_lines_only(fake_ccle):
    cfg = cellline.fit_cellline_features(["L1", "L2", "L3"])
    assert cfg.genes == ("G1 (1)", "G2 (2)")
    assert cfg.means[0] == pytest.approx(3.0)        # L4's 100 is not in the mean
    assert cfg.stds[1] == 1.0                        # constant gene -> unit std, no division by zero
    x = cellline.transform(["L4", "L2"], cfg)
    assert x[1].tolist() == pytest.approx([0.0, 0.0])
    assert x[0, 0] > 10


def test_variance_strategy_ignores_test_lines(fake_ccle):
    cfg = cellline.fit_cellline_features(["L1", "L2", "L3"], strategy="variance", k=1)
    assert cfg.genes == ("G1 (1)",)                  # G2 only varies because of test line L4
    with pytest.raises(KeyError):
        cellline.transform(["NOPE"], cfg)
    assert cellline.CellLineFeatureConfig.from_dict(cfg.to_dict()) == cfg
