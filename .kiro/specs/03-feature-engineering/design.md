# Design Document — 03 Feature Engineering

## Overview

Two pure, deterministic feature pipelines whose configs are serialized next to every model:

```
SMILES ─▶ validate ─▶ standardize ─▶ canonical SMILES + InChIKey
                                      ├─▶ ECFP4 (2048) ┐
                                      ├─▶ descriptors  ├─▶ compound feature vector
                                      └─▶ Murcko scaffold (for splitting only, not a feature)

CCLE (interim parquet) ─▶ gene subset (fit on train lines) ─▶ cell-line feature matrix
```

## Components and Interfaces

### `src/oncora/chem/validate.py`
```python
@dataclass(frozen=True)
class ParsedMol:
    valid: bool
    input_smiles: str
    canonical_smiles: str | None
    inchi_key: str | None
    error: str | None          # human-readable reason when invalid

def parse_smiles(smiles: str) -> ParsedMol
```
RDKit logs errors to stderr. Capture them with `RDLogger` / `rdBase.LogToPythonStderr` redirection
so the reason reaches the user.

### `src/oncora/chem/standardize.py`
- `standardize(mol) -> Mol`: `rdMolStandardize` (Cleanup → LargestFragmentChooser → Uncharger).

### `src/oncora/chem/featurize.py`
```python
@dataclass(frozen=True)
class CompoundFeatureConfig:
    radius: int = 2
    n_bits: int = 2048
    descriptors: tuple[str, ...] = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                                    "NumRotatableBonds", "RingCount", "NumAromaticRings",
                                    "FractionCSP3", "HeavyAtomCount")
    rdkit_version: str = ...
def featurize(mols, cfg) -> np.ndarray          # shape (n, n_bits + len(descriptors)), float32
def murcko_scaffold(mol) -> str                  # generic=False; acyclic → ""
def featurize_table(df, smiles_col, cfg, cache_path) -> pl.DataFrame   # keyed by inchi_key
```
Use `rdFingerprintGenerator.GetMorganGenerator`, the current API. The deprecated
`GetMorganFingerprintAsBitVect` is not used.

### `src/oncora/chem/similarity.py`
- `class TrainingIndex`: holds the training fingerprints as `ExplicitBitVect`s.
  `.nearest(fp) -> (max_tanimoto, inchi_key)` uses `BulkTanimotoSimilarity`, which is fast enough
  for about 100k compounds on CPU.

### `src/oncora/features/cellline.py`
```python
@dataclass(frozen=True)
class CellLineFeatureConfig:
    strategy: Literal["variance", "landmark", "landmark+variance"]
    genes: tuple[str, ...]          # explicit, fitted list (the persisted truth)
    standardize: bool
    means: tuple[float, ...] | None
    stds: tuple[float, ...] | None
def fit_cellline_features(train_model_ids, strategy, k) -> CellLineFeatureConfig
def transform(model_ids, cfg) -> pl.DataFrame   # model_id + feature columns
```

## Data Models

- `data/processed/compound_features.parquet`: `inchi_key, canonical_smiles, scaffold, f_0..f_N`
  (or the fingerprint stored as a packed binary column plus separate descriptor columns).
- Feature configs serialize to JSON as part of `model_card.json` (spec 05/06).

## Error Handling

- Invalid SMILES → `ParsedMol(valid=False, error=...)`, never an exception on the inference path.
- A descriptor that returns NaN or inf (rare for odd valences) is set to 0 and counted. The count is logged at training time.
- A cell line requested at inference that is missing from the CCLE matrix → a clear `KeyError` naming it.

## Testing Strategy

- `tests/test_chem.py`:
  - invalid valence (`C(C)(C)(C)(C)C`) and garbage (`xyz`) are rejected with reasons;
  - aspirin written two ways gives the same canonical SMILES, InChIKey, and features;
  - canonicalization is idempotent;
  - a salt form (aspirin sodium) standardizes to the parent;
  - the scaffold of an acyclic molecule is `""`.
- `tests/test_cellline_features.py`: the gene list is fit on the train IDs only (a test-only
  high-variance gene is not selected), and the feature count matches the config.
