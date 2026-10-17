"""Substructure attribution for one target prediction (spec 05, Req 5; FR-8, STRETCH).

LightGBM's `pred_contrib` returns SHAP values: each feature's additive contribution to this
prediction's log-odds relative to the model's average. For ECFP features, a contributing bit
maps back to the atom environments that set it in the query molecule (RDKit's bit-info map),
so the explanation points at real substructures. These are explanations of the *model*, not
evidence of mechanism.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from oncora.chem.featurize import CompoundFeatureConfig, descriptor_values


@dataclass
class Attribution:
    feature: str
    contribution: float        # SHAP value on the log-odds scale
    atoms: list[int]           # query atoms in the environment (empty for descriptors)
    fragment: str | None       # SMILES of the environment, when it is a fingerprint bit


def _environment(mol: Chem.Mol, atom: int, radius: int) -> tuple[list[int], str]:
    if radius == 0:
        return [atom], mol.GetAtomWithIdx(atom).GetSymbol()
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
    atoms = sorted({a for b in bonds for a in (mol.GetBondWithIdx(b).GetBeginAtomIdx(),
                                               mol.GetBondWithIdx(b).GetEndAtomIdx())} | {atom})
    return atoms, Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, bondsToUse=list(bonds) or None)


def explain(booster: lgb.Booster, mol: Chem.Mol, cfg: CompoundFeatureConfig, k: int = 5) -> list[Attribution]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=cfg.radius, fpSize=cfg.n_bits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    bits = gen.GetFingerprintAsNumPy(mol, additionalOutput=ao).astype(np.float32)
    info = ao.GetBitInfoMap()
    x = np.concatenate([bits, descriptor_values(mol, cfg)])[None, :]
    contrib = booster.predict(x, pred_contrib=True)[0][:-1]   # last column is the bias term
    names = cfg.feature_names
    out = []
    for j in np.argsort(-contrib, kind="stable"):
        if contrib[j] <= 0 or len(out) == k:
            break
        if j < cfg.n_bits:
            if j not in info:
                continue   # an absent bit can carry SHAP mass; only present substructures are shown
            atom, radius = info[j][0]
            atoms, frag = _environment(mol, atom, radius)
            out.append(Attribution(names[j], float(contrib[j]), atoms, frag))
        else:
            out.append(Attribution(names[j], float(contrib[j]), [], None))
    return out
