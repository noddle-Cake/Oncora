from pathlib import Path

import polars as pl
import pytest

from oncora.io import profile


@pytest.fixture
def lincs_tsv(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "DATA_RAW", tmp_path)
    p = tmp_path / "cells.txt"
    p.write_text(
        "cell_id\tcellosaurus_id\tsmiles\n"
        'ACH-000001\tCVCL_0001\tC[C@H](N)C(=O)O\n'
        'ACH-000002\t""\tC/C=C\\C\n'
        'ACH-000003\tCVCL_0003\t""\n'
    )
    return p


def test_double_quote_token_is_null(lincs_tsv):
    prof = profile.profile_delimited("cells", lincs_tsv)
    cols = {c.name: c for c in prof.columns}
    assert prof.n_rows == 3
    assert cols["cellosaurus_id"].null_rate == pytest.approx(1 / 3)
    assert cols["smiles"].null_rate == pytest.approx(1 / 3)


def test_backslash_in_smiles_survives(lincs_tsv):
    prof = profile.profile_delimited("cells", lincs_tsv)
    smiles = next(c for c in prof.columns if c.name == "smiles")
    assert any("\\" in e for e in smiles.examples)


def test_id_patterns_detected(lincs_tsv):
    cols = {c.name: c for c in profile.profile_delimited("cells", lincs_tsv).columns}
    assert cols["cell_id"].id_pattern == "DepMap ModelID"
    assert cols["cellosaurus_id"].id_pattern == "Cellosaurus"
    assert cols["smiles"].id_pattern is None


def test_ccle_profile_counts_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "DATA_RAW", tmp_path)
    df = pl.DataFrame(
        {
            "": [0, 1, 2],
            "SequencingID": ["S1", "S2", "S3"],
            "ModelConditionID": ["MC1", "MC2", "MC3"],
            "ModelID": ["ACH-000001", "ACH-000001", "ACH-000002"],
            "IsDefaultEntryForMC": ["Yes", "Yes", "Yes"],
            "IsDefaultEntryForModel": ["Yes", "No", "Yes"],
            "TP53 (7157)": [1.0, 2.0, 3.0],
            "FLAT (1)": [0.5, 0.5, 0.5],
        }
    )
    prof = profile.profile_ccle(df, tmp_path / "ccle.csv")
    notes = " ".join(prof.notes)
    assert "2 distinct ModelIDs in 3 rows; 2 rows are the default entry" in notes
    assert "1 genes have zero variance" in notes
    assert next(c for c in prof.columns if c.name == "ModelID").id_pattern == "DepMap ModelID"
