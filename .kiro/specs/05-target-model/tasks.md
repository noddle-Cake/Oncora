# Implementation Plan — 05 Compound → Target Model

- [ ] 1. Training pipeline
- [ ] 1.1 Implement the global compound-disjoint holdout plus per-target train/val/test views
  - _Requirements: 1.1, 1.4_
- [ ] 1.2 Implement the per-target LightGBM trainer with early stopping and seeded config; MLflow logging
  - Learn: gradient-boosted trees, early stopping
  - _Requirements: 1.2, 1.5_
- [ ] 1.3 Implement the logistic-regression-on-ECFP comparison model
  - _Requirements: 1.3_

- [ ] 2. Calibration and ranking
  - Fit isotonic or Platt calibration per target on val (selected by Brier score); implement `TargetModel.predict` with target metadata
  - Learn: probability calibration
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 3. Evaluation
  - Run `EvalReport` against the prior and popularity baselines; write `reports/eval/target/`; record the beats-baseline verdict
  - Learn: multi-label evaluation, PR-AUC under imbalance, top-k retrieval
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 4. Artifact and loader
  - Save models, calibrators, `targets.parquet`, `train_index.parquet`, and `model_card.json`; implement `load_target_model`; round-trip test
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 5. Substructure attribution (STRETCH)
  - Map SHAP bit contributions to atom environments via `bitInfoMap`
  - Learn: feature attribution, SHAP
  - _Requirements: 5.1, 5.2_
