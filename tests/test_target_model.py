"""End-to-end target model on a synthetic dataset (spec 05 testing strategy)."""

import json

import numpy as np
import polars as pl
import pytest

from oncora.chem.featurize import CompoundFeatureConfig
from oncora.features.compound import FeatureStore
from oncora.models.target.calibration import apply_calibrator, fit_calibrator
from oncora.models.target.predict import TargetModel
from oncora.models.target.train import train

CFG = CompoundFeatureConfig(n_bits=64)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n = 300
    keys = [f"K{i:04d}-X" for i in range(n)]
    bits = rng.integers(0, 2, (n, CFG.n_bits), dtype=np.uint8)
    desc = rng.normal(size=(n, len(CFG.descriptors))).astype(np.float32)
    store = FeatureStore(keys, np.packbits(bits, axis=1), desc, CFG)
    rows = []
    for t, bit in (("T1", 0), ("T2", 1), ("T3", 2)):           # activity is driven by one bit per target
        for i in rng.choice(n, 200, replace=False):
            rows.append({"inchi_key": keys[i], "group": keys[i][:5], "scaffold": f"S{i % 25}",
                         "target_chembl_id": t, "label": int(bits[i, bit] == 1)})
    table = pl.DataFrame(rows, schema_overrides={"label": pl.Int8})
    meta = pl.DataFrame({"target_chembl_id": ["T1", "T2", "T3"], "target_name": ["a", "b", "c"],
                         "target_gene_symbol": ["GA", "GB", "GC"], "uniprot_accession": ["P1", "P2", "P3"]})
    return table, store, meta, bits, keys


def test_train_save_load_predict(tmp_path, synthetic):
    table, store, meta, bits, keys = synthetic
    run_dir = train(n_jobs=1, table=table, store=store, models_dir=tmp_path, report_dir=None,
                    metadata=meta, log_mlflow=False, with_ci=False)
    card = json.loads((run_dir / "model_card.json").read_text())
    assert card["training_data"]["targets"] == 3
    model_metrics = card["metrics"]["model"]
    assert model_metrics["roc_auc_macro"] > 0.9          # the signal bit is learnable
    assert card["metrics"]["prior_popularity"]["roc_auc_macro"] == pytest.approx(0.5)

    model = TargetModel(run_dir)
    x = store.matrix(keys[:5])
    m1, m2 = model.predict_matrix(x), TargetModel(run_dir).predict_matrix(x)
    np.testing.assert_allclose(m1, m2)                    # load round trip is deterministic
    top = model.predict(store.matrix([keys[0]]), top_n=3)
    assert [p.rank for p in top] == [1, 2, 3] and {p.target_chembl_id for p in top} == {"T1", "T2", "T3"}
    assert model.nearest_training(store.packed_rows([keys[0]])[0]).max_tanimoto <= 1.0


def test_attribution_points_at_present_substructures(tmp_path):
    import lightgbm as lgb
    from rdkit import Chem
    from oncora.chem.featurize import featurize_mols
    from oncora.models.target.attribution import explain
    cfg = CompoundFeatureConfig(n_bits=256)
    smiles = ["c1ccccc1O", "c1ccccc1N", "CCCCO", "CCCCN", "c1ccncc1", "CC(=O)O"] * 20
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    x = featurize_mols(mols, cfg)
    y = np.array([1 if "c1ccccc1" in s else 0 for s in smiles])   # activity = has a benzene ring
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 2}, lgb.Dataset(x, y), 20)
    attr = explain(booster, Chem.MolFromSmiles("c1ccccc1O"), cfg, k=3)
    assert attr and attr[0].contribution > 0
    fp_attr = [a for a in attr if a.fragment]
    assert fp_attr and all(a.atoms for a in fp_attr)


def test_global_split_holds_out_test_compounds_from_every_target(tmp_path, synthetic):
    table, store, meta, *_ = synthetic
    from oncora.evaluation.protocol import make_partition
    from oncora.models.target.train import build_tasks
    part = make_partition(table, "compound")
    tasks, tests = build_tasks(table, part)
    test_keys = set(table.filter(part.test)["inchi_key"])
    for t in tasks:
        assert not test_keys & set(t.train_keys) and not test_keys & set(t.val_keys)


def test_calibrators():
    rng = np.random.default_rng(1)
    raw = rng.uniform(size=400)
    y = (rng.uniform(size=400) < raw).astype(int)
    iso = fit_calibrator(raw, y)
    assert iso["method"] == "isotonic"
    platt = fit_calibrator(raw[:60], y[:60])
    assert platt["method"] == "platt"
    assert fit_calibrator(raw[:10], np.ones(10))["method"] == "identity"
    # perfectly separable tiny validation set: no calibration, so no saturation to 1.0
    assert fit_calibrator(np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8]), np.array([0, 0, 0, 1, 1, 1]))["method"] == "identity"
    # separable but large enough for Platt: regularization keeps the slope moderate
    sep_raw = np.r_[rng.uniform(0, 0.4, 15), rng.uniform(0.6, 1, 15)]
    sep = fit_calibrator(sep_raw, np.r_[np.zeros(15), np.ones(15)].astype(int))
    assert sep["method"] == "platt" and sep["a"] < 10
    assert apply_calibrator(sep, np.array([0.42]))[0] < 0.999
    for cal in (iso, platt):
        p = apply_calibrator(cal, np.array([0.1, 0.5, 0.9]))
        assert ((p >= 0) & (p <= 1)).all() and p[0] <= p[2]
