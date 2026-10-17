# Implementation Plan — 01 Environment & EDA

- [x] 1. Scaffold project and environment
  - Create `pyproject.toml` (Python 3.11, uv) with the pinned deps; generate `uv.lock`
  - Create the directory skeleton from `steering/structure.md`, with `__init__.py` files
  - Add `.gitignore` (raw/interim/processed data, `*.tar`, `*.dmp`, `models/` except cards, `mlruns/`, `.env`)
  - Write `src/oncora/config.py` (paths, `RAW` map, `SEED`, `NULL_TOKENS`)
  - Add `tests/test_smoke.py` that imports every pinned package
  - `git init` and make an initial commit (confirm no large files are tracked)
  - Learn: reproducible environments and lockfiles
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 2. Raw-data integrity check
  - Implement `io/checksums.py::verify_raw` against `data/raw/SHA256SUMS`
  - Test with a tampered temp file
  - _Requirements: 2.6_

- [x] 3. Schema and quality profiler
- [x] 3.1 Implement `io/profile.py` with DuckDB-backed `profile_table` and markdown rendering
  - Handle the `""` null token; detect ID patterns (`ACH-`, `BRD-`)
  - Use SQL aggregates only for `instinfo.txt`
  - _Requirements: 2.1, 2.3, 2.4_
- [x] 3.2 Profile all sources and write `reports/eda_profile.md`
  - Check the numbers against `steering/data-sources.md`; record any drift
  - Learn: data profiling, columnar/out-of-core processing
  - _Requirements: 2.2, 2.5_
- [x] 3.3 Convert the CCLE matrix and DSS3 file to interim Parquet
  - Filter CCLE to `IsDefaultEntryForModel = Yes`; write gene sidecar with parsed symbol/Entrez
  - _Requirements: 2.3_

- [ ] 4. Identity-bridge analysis
- [ ] 4.1 Build the compound bridge and report coverage and `pert_id` conflicts
  - _Requirements: 3.1, 3.2, 3.5_
- [ ] 4.2 Build the cell-line bridge and report the CCLE ∩ DSS3 overlap
  - _Requirements: 3.3, 3.5_
- [ ] 4.3 Quantify disease/lineage coverage for the 985 overlapping lines (A4)
  - Try `cellinfo.txt` via `ccle_name`; report how many lines can be mapped and which external file would close the gap
  - Learn: entity resolution, structure identity via InChIKey
  - _Requirements: 3.4_
- [ ] 4.4 Unit tests for bridges on fixture files
  - _Requirements: 3.1, 3.3_

- [ ] 5. DSS3 target and CCLE feature-space EDA
- [ ] 5.1 Notebook `01_dss3_target`: distribution, zero mass, transforms, variance decomposition (cell line vs. compound), threshold proposal
  - Learn: distribution diagnostics, variance-stabilizing transforms
  - _Requirements: 4.1, 4.2, 4.3, 4.5_
- [ ] 5.2 Notebook `02_ccle_features`: gene variance and landmark-gene overlap; propose subset size
  - Learn: dimensionality reduction and feature selection
  - _Requirements: 4.4, 4.5_

- [ ] 6. Architecture-commit checkpoint
  - Write `reports/eda_summary.md` with a decision log (transform, threshold, gene subset, conflict rule, A4 source)
  - Get the developer's sign-off; copy the decisions into `steering/data-sources.md`
  - _Requirements: 5.1, 5.2, 5.3, 5.4_
