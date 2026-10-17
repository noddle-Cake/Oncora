# Requirements Document — 06 Compound → Sensitivity Model & Cancer-Type Aggregation

## Introduction

Predict DSS3 drug sensitivity for a SMILES against every CCLE cell line, rank the most sensitive
lines, and aggregate them into ranked cancer types or lineages. Compound features come from
**structure only**, so a brand-new SMILES can be scored against any CCLE line without a measured
LINCS signature. This is the core product capability.

**Depends on:** 02, 03, 04. Requirement 4 also depends on resolving open item A4 (cell line → disease source).
**Global IDs implemented:** FR-9, FR-10, FR-11, FR-12, FR-22, FR-23 (artifact side), AC-2, AC-3, AC-5.

## Requirements

### Requirement 1: Training (FR-9, FR-10, FR-12)

**User Story:** As the developer, I want a sensitivity model trained on structure plus
expression features, so that I can rank cell lines for compounds never measured.

#### Acceptance Criteria

1. THE training row SHALL be `concat(compound_features(inchi_key), cellline_features(model_id)) → transformed DSS3`.
2. THE trainer SHALL use the DSS3 transform recorded at the spec 01 checkpoint (log1p or rank) and SHALL document why.
3. THE trainer SHALL train a LightGBM regressor on the transformed target and a parallel LightGBM classifier at the recorded sensitive/resistant threshold.
4. Compound features SHALL NOT include any LINCS signature or measured biological profile. A test SHALL assert the feature columns come only from the compound and cell-line feature configs.
5. The cell-line feature gene list SHALL be fit on training cell lines only, within each fold.
6. Each run SHALL be logged to MLflow.

### Requirement 2: Evaluation (AC-2, AC-3, AC-5)

**User Story:** As a judge, I want sensitivity metrics on unseen compounds and unseen scaffolds,
compared with strong baselines, so that I can see whether the model learned compound effects.

#### Acceptance Criteria

1. THE evaluation SHALL report Spearman ρ (overall, the mean per-compound ρ across cell lines, and the mean per-cell-line ρ across compounds), RMSE on the transform, and thresholded ROC/PR-AUC.
2. Metrics SHALL be reported on **both** the compound-disjoint and the scaffold-disjoint splits, with bootstrap CIs.
3. THE evaluation SHALL compare against the global-mean and cell-line-mean baselines and SHALL state whether the model beats each.
4. THE report SHALL highlight per-cell-line ρ, because the cell-line-mean baseline scores ≈ 0 there by construction. That metric isolates whether the model learned anything about compounds.
5. THE model SHALL be accepted only if it beats the cell-line-mean baseline on the compound-disjoint split on at least one compound-sensitive metric (per-cell-line ρ or thresholded PR-AUC).
6. THE evaluation SHALL include a compound-features-only ablation and SHALL report whether the full model beats the cell-line-mean baseline on per-compound ρ (selectivity). IF it does not, THEN the report and UI SHALL state that cell-line rankings reflect compound potency plus the line's average sensitivity, not learned selectivity.
7. Rows from mixed-scale compounds (spec 01 finding) SHALL be handled as decided at the spec 01 checkpoint, and the handling SHALL be recorded in the model card.

### Requirement 3: Inference over all cell lines (FR-9)

**User Story:** As a user, I want a ranked list of the cell lines most sensitive to my compound,
so that I know where to look experimentally.

#### Acceptance Criteria

1. WHEN given a valid SMILES THEN the model SHALL score it against every cell line in the persisted cell-line feature matrix and return them ranked by predicted sensitivity.
2. Each result SHALL include `model_id`, cell-line name, predicted DSS3 (back-transformed where meaningful), sensitive-class probability, and rank.
3. Inference for one SMILES SHALL complete within a few seconds on CPU once loaded (NFR-4, STRETCH).

### Requirement 4: Cancer-type aggregation (FR-11)

**User Story:** As a user, I want cancer types ranked by predicted sensitivity, so that I can see
which diseases warrant investigation.

#### Acceptance Criteria

1. THE system SHALL map each cell line to `primary_disease` and lineage using the source chosen for A4.
2. THE system SHALL aggregate per-cell-line predictions per disease (default: mean of the top-k lines with k = 3, plus the fraction predicted sensitive), and rank the diseases.
3. Diseases with fewer than a configured minimum number of cell lines (default 3) SHALL be flagged as low-support, not silently dropped.
4. Cell lines with no disease mapping SHALL be counted and reported, and SHALL NOT be assigned a guessed disease.
5. WHEN checked on a known drug held out of training THEN the aggregation's ranking SHALL be inspected for plausibility and recorded in the eval report.

### Requirement 5: Versioned artifact (FR-22, FR-23)

**User Story:** As the developer, I want the sensitivity model saved with its feature configs and
cell-line matrix, so that inference matches training exactly.

#### Acceptance Criteria

1. THE trainer SHALL write `models/sensitivity/<run_id>/` with the regressor, classifier, compound and cell-line feature configs, the precomputed cell-line feature matrix, the disease map, the training index for near-seen detection, and `model_card.json`.
2. A `load_sensitivity_model(run_id)` function SHALL restore an equivalent predictor without retraining.
