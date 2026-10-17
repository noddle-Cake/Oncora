import json
import tarfile

import polars as pl
import pytest

from oncora.serve import artifacts


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr("oncora.data_prep.provenance.ROOT", tmp_path)
    interim, processed = tmp_path / "data" / "interim", tmp_path / "data" / "processed"
    interim.mkdir(parents=True)
    processed.mkdir(parents=True)
    pl.DataFrame({"pert_id": ["P1", "P2"], "inchi_key": ["AAAAAAAAAAAAAA-XX-N", None],
                  "connectivity": ["AAAAAAAAAAAAAA", None], "cmap_names": [["a"], ["b"]], "moas": [["m"], []],
                  "targets": [["EGFR"], []], "canonical_smiles": ["C", None]}).write_parquet(interim / "compound_bridge.parquet")
    pl.DataFrame({"inchi_key": ["AAAAAAAAAAAAAA-XX-N"] * 2, "connectivity": ["AAAAAAAAAAAAAA"] * 2,
                  "target_chembl_id": ["T1", "T2"], "target_name": ["egfr", "other"], "pchembl_median": [7.0, 5.0],
                  "n_measured": [1, 1]}).write_parquet(interim / "chembl_activities.parquet")
    pl.DataFrame({"target_chembl_id": ["T1", "T2"], "target_gene_symbol": ["EGFR", "ABC"]}).write_parquet(
        processed / "target_pairs.parquet")
    pl.DataFrame({"gene_symbol": ["EGFR", "XYZ"], "gene_col": ["EGFR (1956)", "XYZ (1)"]}).write_parquet(
        interim / "ccle_genes.parquet")
    pl.DataFrame({"model_id": [f"ACH-{i:06d}" for i in range(120)], "EGFR (1956)": [float(i) for i in range(120)],
                  "XYZ (1)": [0.0] * 120}).write_parquet(interim / "ccle_expression.parquet")
    for kind, files in (("target", ["model_card.json", "lgbm/T1.txt"]), ("sensitivity", ["model_card.json", "regressor.txt"])):
        for f in files:
            p = tmp_path / "models" / kind / "run1" / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"{kind}:{f}")
    return tmp_path


def test_tables_keep_only_what_inference_reads(tree):
    serving = tree / "data" / "serving"
    rows = artifacts.build_tables(serving, tree / "data" / "interim", tree / "data" / "processed")
    assert rows["compound_bridge"] == 1                          # rows without a structure are dropped
    act = pl.read_parquet(serving / "chembl_actives.parquet")
    assert act["target_chembl_id"].to_list() == ["T1"]           # actives only
    assert act["target_gene_symbol"].to_list() == ["EGFR"]       # gene symbol pre-joined
    assert pl.read_parquet(serving / "ccle_expression.parquet").columns == ["model_id", "EGFR (1956)"]
    assert "l1000_consensus" not in rows                         # optional extract absent -> skipped


def test_package_extract_verify_roundtrip_and_tamper_detection(tree):
    serving = tree / "data" / "serving"
    artifacts.build_tables(serving, tree / "data" / "interim", tree / "data" / "processed")
    m = tree / "models"
    archive = artifacts.package(m / "target" / "run1", m / "sensitivity" / "run1", tree / "dist",
                                root=tree, serving=serving)
    out = tree / "extracted"
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    manifest = artifacts.verify(out)
    assert manifest["model_runs"] == {"target": "run1", "sensitivity": "run1"}
    assert "models/target/run1/lgbm/T1.txt" in manifest["files"]
    assert "data/serving/chembl_actives.parquet" in manifest["files"]
    assert json.loads((out / "MANIFEST.json").read_text())["licenses"]
    (out / "models" / "target" / "run1" / "lgbm" / "T1.txt").write_text("tampered")
    with pytest.raises(RuntimeError, match="missing or corrupted"):
        artifacts.verify(out)
    sha_line = (tree / "dist" / (archive.name + ".sha256")).read_text()
    assert sha_line.endswith(f"  {archive.name}\n")
