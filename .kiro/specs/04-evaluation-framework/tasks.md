# Implementation Plan — 04 Evaluation Framework

- [x] 1. Splitters
- [x] 1.1 Implement `compound_folds`, `holdout`, `SplitManifest`, `assert_disjoint`
  - _Requirements: 1.1, 1.2, 1.3, 1.4_
- [x] 1.2 Implement `scaffold_folds` with balanced greedy assignment and per-compound groups for acyclic compounds
  - _Requirements: 2.1, 2.2, 2.3_
- [x] 1.3 Anti-leakage test suite, including the random-split negative control
  - Learn: data leakage, grouped vs. scaffold splitting
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 2. Baselines
  - Implement the prior, popularity, global-mean, and cell-line-mean baselines with train-only fitting; tests
  - Learn: baselines as a sanity floor
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 3. Metrics
- [x] 3.1 Implement `target_metrics` (PR/ROC-AUC macro and weighted, top-k)
  - _Requirements: 5.1_
- [x] 3.2 Implement `sensitivity_metrics` (overall and per-compound Spearman, RMSE on the transform, thresholded AUC)
  - _Requirements: 5.2_
- [x] 3.3 Implement compound-level `bootstrap_ci`; unit tests on hand-computed examples
  - Learn: ROC vs. PR under imbalance, rank correlation, bootstrap CIs
  - _Requirements: 5.3_

- [x] 4. Evaluation report
  - Implement `EvalReport` (md/json, `beats_baselines`); smoke-run the baselines alone on both modeling tables to produce the first floor numbers
  - _Requirements: 6.1, 6.2_
