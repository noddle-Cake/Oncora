# Implementation Plan — 02 Data Pipeline

- [x] 1. Provenance and integrity helpers
  - Implement `data_prep/provenance.py` (`write_parquet`, `read_provenance`, `assert_table`)
  - Unit-test the metadata round-trip and each assertion failure
  - _Requirements: 5.1, 5.2_

- [ ] 2. ChEMBL restore and export
- [ ] 2.1 Implement `io/chembl_export.py` (extract, restore, export via DuckDB postgres extension, spot check)
  - _Requirements: 1.1, 1.3_
- [ ] 2.2 **Get developer approval**, install PostgreSQL, run the restore and export, record the row counts
  - Spot-check CHEMBL941 (imatinib) → ABL1/KIT/PDGFR among its targets
  - Learn: relational → columnar ETL
  - _Requirements: 1.2, 1.4, 1.5_

- [ ] 3. ChEMBL activity filtering
- [ ] 3.1 Implement `filter_activities` with assay-type, confidence, organism, and validity filters
  - _Requirements: 2.1, 2.2, 2.6_
- [ ] 3.2 Implement the censoring policy and dedup; report before/after counts
  - Learn: assay heterogeneity, pChEMBL, censored measurements
  - _Requirements: 2.3, 2.4, 2.5_
- [ ] 3.3 Unit tests for the censoring and dedup rules
  - _Requirements: 2.4, 2.5_

- [ ] 4. Target-pair label construction
- [ ] 4.1 Implement `build_target_pairs` with the threshold, measured negatives, seeded sampled negatives, and minimum-positives filter
  - _Requirements: 3.1, 3.2, 3.3, 3.4_
- [ ] 4.2 Write the label policy to provenance; unit-test determinism and collision-freedom
  - Learn: positive/negative framing, "absence ≠ inactive"
  - _Requirements: 3.5_

- [ ] 5. Sensitivity modeling table
  - Implement `build_sensitivity`: InChIKey merge, CCLE restriction, shape report, tests
  - Apply the `DEMO_HOLDOUT` exclusion to both tables (choose the compound in spec 09 task 1 first)
  - Add `connectivity` and `mixed_scale` columns (D-01, D-03)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 6. Wire up the stage CLIs and add a pipeline run note to `reports/eda_summary.md`
  - _Requirements: 5.3_
