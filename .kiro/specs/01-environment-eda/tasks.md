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

- [x] 4. Identity-bridge analysis
- [x] 4.1 Build the compound bridge and report coverage and `pert_id` conflicts
  - _Requirements: 3.1, 3.2, 3.5_
- [x] 4.2 Build the cell-line bridge and report the CCLE ∩ DSS3 overlap
  - _Requirements: 3.3, 3.5_
- [x] 4.3 Quantify disease/lineage coverage for the 985 overlapping lines (A4)
  - Try `cellinfo.txt` via `ccle_name`; report how many lines can be mapped and which external file would close the gap
  - Learn: entity resolution, structure identity via InChIKey
  - _Requirements: 3.4_
- [x] 4.4 Unit tests for bridges on fixture files
  - _Requirements: 3.1, 3.3_

- [x] 5. DSS3 target and CCLE feature-space EDA
  - Implemented as modules (`oncora.eda.target`, `oncora.eda.features`) writing `reports/eda_target.md` and `reports/eda_features.md`, not notebooks, so the analysis is reproducible and diffable
- [x] 5.1 DSS3 target analysis: distribution, zero mass, transforms, variance decomposition (cell line vs. compound), threshold proposal
  - Learn: distribution diagnostics, variance-stabilizing transforms
  - _Requirements: 4.1, 4.2, 4.3, 4.5_
- [x] 5.2 CCLE feature-space analysis: gene variance and landmark-gene overlap; propose subset size
  - Learn: dimensionality reduction and feature selection
  - _Requirements: 4.4, 4.5_

- [ ] 5.3 Profile the L1000 Level 4 GCTX (new file; optional-evidence only, A2 revised)
  - Add a GCTX reader dependency (`cmapPy`, or `h5py` for a lighter dependency) — do NOT load the
    matrix whole; it is ~90 GB in RAM
  - Confirm disk headroom (≥ ~100 GB) and that `*.gctx` is gitignored
  - Read only metadata + a small slab: confirm shape (1,805,898 × 12,328), `rid`/`cid` axes, and a
    sample of z-score values; verify `rid` joins to `geneinfo.txt` (Entrez) and `cid` joins to
    `instinfo.sample_id` → `pert_id`
  - Quantify how many of the 972 DSS3 `pert_id`s have at least one `trt_cp` profile (the compounds
    for which signature-based evidence is even possible)
  - Write findings into `reports/eda_signatures.md`
  - Record in the decision log that signatures are evidence-only and NEVER a sensitivity-model
    inference feature (FR-10 / A2 revised in `steering/data-sources.md`)
  - Learn: HDF5/GCTX out-of-core access, subsetting by row/column indices
  - _Requirements: 4.5, 5.3_

- [x] 6. Architecture-commit checkpoint
  - Write `reports/eda_summary.md` with a decision log (transform, threshold, gene subset, conflict rule, A4 source)
  - Get the developer's sign-off; copy the decisions into `steering/data-sources.md`
  - _Requirements: 5.1, 5.2, 5.3, 5.4_
