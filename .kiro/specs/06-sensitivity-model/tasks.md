# Implementation Plan — 06 Compound → Sensitivity Model

- [ ] 1. Design matrix
  - Implement `build_design_matrix` with index-based assembly (no joined DataFrame) and `assert_structure_only`; tests
  - Learn: pairwise (drug × cell) modeling, memory-efficient feature assembly
  - _Requirements: 1.1, 1.4_

- [ ] 2. Training
- [ ] 2.1 Implement the fold loop with per-fold cell-line gene-subset fitting
  - _Requirements: 1.5_
- [ ] 2.2 Train the LightGBM regressor on the transformed DSS3 and the thresholded classifier; MLflow logging
  - Learn: regression on skewed targets, variance-stabilizing transforms
  - _Requirements: 1.2, 1.3, 1.6_

- [ ] 3. Evaluation on compound and scaffold splits
  - Overall, per-compound, and per-cell-line ρ; RMSE; thresholded AUC; CIs; global-mean and cell-line-mean baselines; acceptance verdict
  - Learn: rank correlation, why per-cell-line ρ isolates compound learning
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 4. Inference and artifact
  - Final fit; save the artifact bundle; implement `SensitivityModel.predict` and `load_sensitivity_model`; round-trip test; time single-SMILES inference
  - _Requirements: 3.1, 3.2, 3.3, 5.1, 5.2_

- [ ] 5. Cancer-type aggregation (blocked on A4)
- [ ] 5.1 Build `disease_map.parquet` from the A4 source; report unmapped lines
  - _Requirements: 4.1, 4.4_
- [ ] 5.2 Implement `aggregate` (top-k mean, frac_sensitive, low-support flag); fixture tests
  - Learn: aggregating predictions to an ontology level
  - _Requirements: 4.2, 4.3_
- [ ] 5.3 Plausibility check on a held-out known drug; record it in the eval report
  - _Requirements: 4.5_
