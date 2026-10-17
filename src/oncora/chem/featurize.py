"""Structure-only compound features (spec 03, Req 3; FR-10).

Technique: a compound becomes a fixed-length vector computed from its structure alone, so any
new SMILES can be featurized at inference without measured data.

- **ECFP4** (Morgan, radius 2, 2,048 bits): each bit marks the presence of a circular atom
  neighbourhood up to 2 bonds out, hashed into a fixed bit vector. It is the standard strong
  baseline for bioactivity. It is achiral by default, which is why splits group stereoisomers
  together (D-01).
- **Physicochemical descriptors:** size, lipophilicity, polarity, flexibility and aromaticity,
  complementing the substructure bits with whole-molecule properties.
- **Bemis–Murcko scaffold:** ring systems plus linkers with side chains stripped, used only for
  scaffold-disjoint splitting (spec 04), never as a feature.

Fingerprints are stored bit-packed (256 bytes per compound) and unpacked only when a model needs
a dense matrix.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from oncora.chem.standardize import standardize

DEFAULT_DESCRIPTORS = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
)
_DESCRIPTOR_FNS = dict(Descriptors.descList)


@dataclass(frozen=True)
class CompoundFeatureConfig:
    radius: int = 2
    n_bits: int = 2048
    descriptors: tuple[str, ...] = DEFAULT_DESCRIPTORS
    rdkit_version: str = field(default=rdkit.__version__)

    @property
    def n_features(self) -> int:
        return self.n_bits + len(self.descriptors)

    @property
    def feature_names(self) -> list[str]:
        return [f"ecfp_{i}" for i in range(self.n_bits)] + [f"desc_{d}" for d in self.descriptors]

    def to_dict(self) -> dict:
        return asdict(self) | {"descriptors": list(self.descriptors)}

    @classmethod
    def from_dict(cls, d: dict) -> "CompoundFeatureConfig":
        return cls(**(d | {"descriptors": tuple(d["descriptors"])}))


def _generator(cfg: CompoundFeatureConfig):
    return rdFingerprintGenerator.GetMorganGenerator(radius=cfg.radius, fpSize=cfg.n_bits)


def fingerprint_bits(mol: Chem.Mol, cfg: CompoundFeatureConfig, generator=None) -> np.ndarray:
    return (generator or _generator(cfg)).GetFingerprintAsNumPy(mol).astype(np.uint8)


def descriptor_values(mol: Chem.Mol, cfg: CompoundFeatureConfig) -> np.ndarray:
    vals = np.array([_DESCRIPTOR_FNS[name](mol) for name in cfg.descriptors], dtype=np.float64)
    # Rare odd valences give NaN/inf; zero is a neutral fill and the count is tracked by callers.
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def murcko_scaffold(mol: Chem.Mol) -> str:
    """Canonical SMILES of the Bemis–Murcko scaffold; "" for acyclic molecules."""
    if mol.GetRingInfo().NumRings() == 0:
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def featurize_mols(mols: list[Chem.Mol], cfg: CompoundFeatureConfig = CompoundFeatureConfig()) -> np.ndarray:
    """Dense float32 matrix (n, n_bits + n_descriptors) for already-standardized molecules."""
    gen = _generator(cfg)
    out = np.zeros((len(mols), cfg.n_features), dtype=np.float32)
    for i, mol in enumerate(mols):
        out[i, : cfg.n_bits] = fingerprint_bits(mol, cfg, gen)
        out[i, cfg.n_bits:] = descriptor_values(mol, cfg)
    return out


def featurize_smiles_batch(smiles: list[str], cfg_dict: dict) -> dict[str, list]:
    """Worker: standardize and featurize a batch. Returns parallel lists; failures have ok=False.

    Takes a plain dict config so it pickles cleanly into worker processes.
    """
    cfg = CompoundFeatureConfig.from_dict(cfg_dict)
    gen = _generator(cfg)
    res = {"ok": [], "std_smiles": [], "std_inchi_key": [], "scaffold": [], "fp": [], "desc": [], "n_nonfinite": []}
    with rdBase.BlockLogs():
        for s in smiles:
            mol = Chem.MolFromSmiles(s) if s else None
            try:
                mol = standardize(mol) if mol is not None else None
            except Exception:
                mol = None
            if mol is None or mol.GetNumAtoms() == 0:
                for k, v in (("ok", False), ("std_smiles", None), ("std_inchi_key", None), ("scaffold", None),
                             ("fp", None), ("desc", None), ("n_nonfinite", 0)):
                    res[k].append(v)
                continue
            raw = np.array([_DESCRIPTOR_FNS[n](mol) for n in cfg.descriptors], dtype=np.float64)
            res["ok"].append(True)
            res["std_smiles"].append(Chem.MolToSmiles(mol))
            res["std_inchi_key"].append(Chem.MolToInchiKey(mol) or None)
            res["scaffold"].append(murcko_scaffold(mol))
            res["fp"].append(np.packbits(fingerprint_bits(mol, cfg, gen)).tobytes())
            res["desc"].append(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            res["n_nonfinite"].append(int((~np.isfinite(raw)).sum()))
    return res


def unpack_fingerprints(packed: list[bytes], n_bits: int) -> np.ndarray:
    buf = np.frombuffer(b"".join(packed), dtype=np.uint8).reshape(len(packed), -1)
    return np.unpackbits(buf, axis=1, count=n_bits)
