# Requirements Document — 04 Evaluation Framework

## Introduction

This spec is the credibility backbone. It is built and tested **before any model**, so every
metric reported is trustworthy from the first run. It provides leakage-safe splitters, metric
functions chosen for imbalance and skew, trivial baselines, and a standard evaluation report.

**Depends on:** 03 (InChIKeys, scaffolds).
**Global IDs implemented:** NFR-1, AC-2, AC-3, plus the metric plumbing for AC-4 and AC-5.

## Requirements

### Requirement 1: Compound-disjoint splitting (NFR-1, AC-2)

**User Story:** As a judge, I want the headline metrics measured on compounds the model never
saw, so that the results reflect real unseen-compound performance.

#### Acceptance Criteria

1. THE splitter SHALL partition rows by the **InChIKey connectivity block** (the first 14 characters) so that no connectivity block appears in more than one of train, validation, or test. Stereoisomers share a block and have identical achiral fingerprints, so splitting by the full key would leak (see `reports/eda_bridges.md`).
2. THE splitter SHALL support k-fold grouped CV and a single holdout, both seeded.
3. FOR the sensitivity data, all rows of a compound (across every cell line) SHALL fall in the same partition.
4. THE splitter SHALL return indices together with a manifest (seed, strategy, group counts, InChIKey hash per partition) for the model card.

### Requirement 2: Scaffold-disjoint splitting (AC-2)

**User Story:** As a judge, I want a harder split based on new chemotypes, so that I can see how
the model generalizes beyond close analogs.

#### Acceptance Criteria

1. THE splitter SHALL group compounds by Bemis–Murcko scaffold so that no scaffold appears in both train and test.
2. Acyclic compounds (empty scaffold) SHALL each be treated as their own group, not as one giant shared group.
3. Scaffold groups SHALL be assigned to folds with a balanced, seeded strategy so that test sets are not dominated by one large scaffold.

### Requirement 3: Automated anti-leakage tests (AC-2)

**User Story:** As the developer, I want tests that fail if any split leaks, so that leakage
cannot creep in during refactoring.

#### Acceptance Criteria

1. THE test suite SHALL assert zero connectivity-block overlap (which implies zero full-InChIKey overlap) between partitions for every split strategy.
2. THE test suite SHALL assert zero scaffold overlap for the scaffold strategy.
3. THE test suite SHALL include negative controls: a random row split, and a full-InChIKey split on data containing stereoisomers, SHALL both be detected as leaky by the same check.

### Requirement 4: Trivial baselines (AC-3)

**User Story:** As a judge, I want each model compared with a trivial predictor on the same
split, so that I can tell whether it learned anything.

#### Acceptance Criteria

1. FOR target prediction, THE framework SHALL provide a class-prior baseline per target and a global target-popularity ranking baseline.
2. FOR sensitivity, THE framework SHALL provide global-mean and cell-line-mean baselines, both fit on training rows only.
3. Baselines SHALL be evaluated with exactly the same splits and metric functions as the models.

### Requirement 5: Metrics (AC-4, AC-5)

**User Story:** As the developer, I want metrics that match each problem's imbalance and skew, so
that the reported numbers are meaningful.

#### Acceptance Criteria

1. Target metrics SHALL include per-target PR-AUC and ROC-AUC (macro and support-weighted) and per-compound top-k hit rate (k = 1, 5, 10).
2. Sensitivity metrics SHALL include Spearman ρ (overall and the mean of per-compound ρ across cell lines), RMSE on the transformed target, and ROC/PR-AUC at the recorded sensitive/resistant threshold.
3. THE framework SHALL report bootstrap 95% confidence intervals (resampling compounds, seeded) for headline metrics.

### Requirement 6: Evaluation report

**User Story:** As the developer, I want a standard model-vs-baseline report, so that every
model is judged the same way.

#### Acceptance Criteria

1. THE framework SHALL emit a markdown and JSON report comparing model and baselines per split strategy, with CIs.
2. THE report SHALL state explicitly whether the model beats each baseline on the leakage-safe split (the AC-3 acceptance gate).
