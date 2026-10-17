# Implementation Plan — 03 Feature Engineering

- [x] 1. SMILES validation and standardization
- [x] 1.1 Implement `chem/validate.py::parse_smiles` with captured RDKit error messages
  - _Requirements: 1.1, 1.2, 1.3_
- [x] 1.2 Implement `chem/standardize.py` and emit canonical SMILES and InChIKey
  - _Requirements: 2.1, 2.2_
- [x] 1.3 Unit tests: invalid inputs, equivalence, idempotence, salt stripping
  - Learn: molecular graphs, canonical forms, InChIKey as identity
  - _Requirements: 1.2, 2.2, 2.3_

- [x] 2. Compound featurizer
- [x] 2.1 Implement `CompoundFeatureConfig`, `featurize`, and `murcko_scaffold`
  - _Requirements: 3.1, 3.2, 3.4_
- [x] 2.2 Implement `featurize_table` with an InChIKey-keyed Parquet cache; featurize sensitivity and target compounds
  - _Requirements: 3.3, 3.5_
- [x] 2.3 Unit tests for feature determinism and scaffold extraction
  - Learn: ECFP fingerprints, Bemis–Murcko scaffolds
  - _Requirements: 2.3, 3.2_

- [ ] 3. Near-seen detection (STRETCH)
  - Implement `chem/similarity.py::TrainingIndex`; test a known molecule against itself → 1.0
  - Learn: Tanimoto similarity
  - _Requirements: 4.1, 4.2_

- [ ] 4. Cell-line expression features
- [ ] 4.1 Implement `fit_cellline_features` / `transform` with train-only fitting and a persisted gene list
  - _Requirements: 5.1, 5.2, 5.3_
- [ ] 4.2 Verify coverage of the 985 overlapping lines and the feature count; leakage unit test
  - Learn: high-dimensional feature selection, train/inference parity
  - _Requirements: 5.2, 5.4_
