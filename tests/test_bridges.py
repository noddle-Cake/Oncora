import polars as pl
import pytest

from oncora.data_prep.bridges import (
    RESTRICTED,
    build_cellline_bridge,
    build_compound_bridge,
    read_lincs_tsv,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_IK = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


@pytest.fixture
def compoundinfo(tmp_path):
    p = tmp_path / "compoundinfo.txt"
    p.write_text(
        "pert_id\tcmap_name\ttarget\tmoa\tcanonical_smiles\tinchi_key\tcompound_aliases\n"
        f"BRD-1\taspirin\tPTGS1\tCOX inhibitor\t{ASPIRIN}\t{ASPIRIN_IK}\t\"\"\n"
        f"BRD-1\taspirin\tPTGS2\tCOX inhibitor\t{ASPIRIN}\t{ASPIRIN_IK}\t\"\"\n"
        f"BRD-2\tacetylsalicylic acid\t\"\"\t\"\"\t{ASPIRIN}\t{ASPIRIN_IK}\t\"\"\n"
        "BRD-3\tsecret\t\"\"\t\"\"\trestricted\trestricted\t\"\"\n"
    )
    return read_lincs_tsv(p, extra_nulls=(RESTRICTED,))


def test_read_lincs_nulls(compoundinfo):
    assert compoundinfo.filter(pl.col("pert_id") == "BRD-2")["target"][0] is None
    assert compoundinfo.filter(pl.col("pert_id") == "BRD-3")["canonical_smiles"][0] is None


def test_duplicate_rows_collapse_to_lists(compoundinfo):
    b = build_compound_bridge(compoundinfo, {"BRD-1", "BRD-2"})
    row = b.filter(pl.col("pert_id") == "BRD-1").row(0, named=True)
    assert row["targets"] == ["PTGS1", "PTGS2"]
    assert row["moas"] == ["COX inhibitor"]
    assert row["n_source_rows"] == 2
    assert row["rdkit_inchi_key"] == ASPIRIN_IK
    assert b.height == 3


def test_in_dss3_flag_and_restricted(compoundinfo):
    b = build_compound_bridge(compoundinfo, {"BRD-1"})
    flags = dict(zip(b["pert_id"], b["in_dss3"]))
    assert flags == {"BRD-1": True, "BRD-2": False, "BRD-3": False}
    restricted = b.filter(pl.col("pert_id") == "BRD-3").row(0, named=True)
    assert restricted["canonical_smiles"] is None and restricted["rdkit_inchi_key"] is None


def test_structure_conflict_raises(compoundinfo):
    bad = pl.concat([
        compoundinfo,
        compoundinfo.filter(pl.col("pert_id") == "BRD-1").head(1).with_columns(
            pl.lit("CCO").alias("canonical_smiles")
        ),
    ])
    with pytest.raises(ValueError, match="more than one structure"):
        build_compound_bridge(bad, set())


def test_cellline_bridge_sets():
    b = build_cellline_bridge({"ACH-1", "ACH-2"}, {"ACH-2", "ACH-3"})
    assert b["model_id"].to_list() == ["ACH-1", "ACH-2", "ACH-3"]
    assert b.filter("in_ccle", "in_dss3")["model_id"].to_list() == ["ACH-2"]
    assert b["primary_disease"].null_count() == 3
