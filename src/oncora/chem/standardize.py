"""Molecule standardization (spec 03, Req 2; FR-3).

The same compound arrives in many forms: as a hydrochloride salt, with a counter-ion, charged
or neutral, with explicit or implicit hydrogens. Standardizing to one parent form before
featurizing means equivalent inputs give identical features. That holds for training data from
several sources and for whatever a user pastes in. Steps, each from RDKit's MolStandardize:

1. Cleanup: normalize functional-group representations (e.g. nitro groups) and reionize.
2. Largest fragment: drop salts, counter-ions and solvents.
3. Uncharge: neutralize where a neutral form exists.
"""

from functools import lru_cache

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


@lru_cache(maxsize=1)
def _tools():
    return rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()


def standardize(mol: Chem.Mol) -> Chem.Mol:
    chooser, uncharger = _tools()
    with rdBase.BlockLogs():  # MolStandardize logs every step at INFO level
        mol = rdMolStandardize.Cleanup(mol)
        mol = chooser.choose(mol)
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    return mol


def standardize_smiles(smiles: str) -> str | None:
    """Canonical SMILES of the standardized parent, or None if the input cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(standardize(mol))
    except Exception:
        return None
