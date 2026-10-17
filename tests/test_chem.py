import pytest

from oncora.chem.validate import parse_smiles

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_ALT = "OC(=O)c1ccccc1OC(C)=O"
ASPIRIN_SODIUM = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
ASPIRIN_IK = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


@pytest.mark.parametrize(
    "smiles, fragment",
    [
        ("", "Empty input"),
        ("   ", "Empty input"),
        ("xyz", "Could not parse"),
        ("C1CC", "Could not parse"),          # unclosed ring
        ("C(C)(C)(C)(C)C", "valence"),        # pentavalent carbon
        ("CC O", "spaces"),
    ],
)
def test_invalid_inputs_have_reasons(smiles, fragment):
    p = parse_smiles(smiles)
    assert not p.valid
    assert fragment.lower() in p.error.lower()
    assert p.canonical_smiles is None and p.mol is None


def test_equivalent_smiles_canonicalize_identically():
    a, b = parse_smiles(ASPIRIN), parse_smiles(ASPIRIN_ALT)
    assert a.valid and b.valid
    assert a.canonical_smiles == b.canonical_smiles
    assert a.inchi_key == b.inchi_key == ASPIRIN_IK


def test_canonicalization_is_idempotent():
    once = parse_smiles(ASPIRIN).canonical_smiles
    assert parse_smiles(once).canonical_smiles == once


def test_salt_standardizes_to_parent():
    p = parse_smiles(ASPIRIN_SODIUM)
    assert p.valid
    assert p.canonical_smiles == parse_smiles(ASPIRIN).canonical_smiles
    assert p.inchi_key == ASPIRIN_IK


def test_whitespace_is_trimmed():
    assert parse_smiles(f"  {ASPIRIN}\n").valid
