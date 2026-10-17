# Requirements Document — 05 Compound → Target Model

## Introduction

Predict a ranked list of likely protein targets for a SMILES, trained on quality-filtered ChEMBL
activities (spec 02) using structure-only features (spec 03). Validation uses the leakage-safe
framework (spec 04). The MVP bar is a defensible baseline that beats trivial predictors on
unseen compounds. It does not need to be state-of-the-art.

**Depends on:** 02, 03, 04. Can proceed in parallel with 06.
**Global IDs implemented:** FR-5, FR-6, FR-7, FR-8 (STRETCH), FR-22, FR-23 (artifact side), AC-3, AC-4.

## Requirements

### Requirement 1: Training (FR-5, FR-6, FR-7)

**User Story:** As the developer, I want a reproducible target-model training run, so that I can
produce scored target rankings for any SMILES.

#### Acceptance Criteria

1. THE trainer SHALL train on `processed/target_pairs.parquet` only, inheriting its label policy (threshold, negatives).
2. THE trainer SHALL train one LightGBM binary classifier per qualifying target on compound features, with seeded hyperparameters from config.
3. THE trainer SHALL also train a logistic-regression-on-ECFP comparison model under the same splits.
4. THE split SHALL be **globally** compound-disjoint: a test InChIKey is excluded from training for every target.
5. Each run SHALL be logged to MLflow (params, metrics, artifacts).

### Requirement 2: Scoring and ranking (FR-5)

**User Story:** As a user, I want a ranked list of targets with interpretable scores, so that I
can see which targets are most likely.

#### Acceptance Criteria

1. WHEN given a valid SMILES THEN the model SHALL return all qualifying targets ranked by score, with the top N (configurable, default 20) surfaced.
2. Scores SHALL be calibrated on held-out data (isotonic or Platt, chosen by validation Brier score) so that they are comparable across targets.
3. Each result SHALL include the target ChEMBL ID, preferred name, gene symbol, and UniProt accession where available.

### Requirement 3: Evaluation (AC-3, AC-4)

**User Story:** As a judge, I want target metrics on unseen compounds compared with trivial
baselines, so that I can trust the ranking.

#### Acceptance Criteria

1. THE evaluation SHALL report per-target PR-AUC (primary) and ROC-AUC, plus per-compound top-k hit rate, on the compound-disjoint split with bootstrap CIs.
2. THE evaluation SHALL compare against the prior and popularity baselines from spec 04 and SHALL state whether the model beats them.
3. THE report SHALL justify PR-AUC as primary given class imbalance.
4. THE model SHALL be accepted only if it beats the baselines on the leakage-safe split.

### Requirement 4: Versioned artifact (FR-22, FR-23)

**User Story:** As the developer, I want the trained model saved with everything needed to
reproduce it, so that inference never retrains and every metric is traceable.

#### Acceptance Criteria

1. THE trainer SHALL write `models/target/<run_id>/` containing the models, calibrators, target metadata, and `model_card.json`.
2. `model_card.json` SHALL include the training-data hash, feature config, split manifest, label and negative policy, metrics, seed, and library versions.
3. THE artifact SHALL include the training InChIKeys and fingerprints needed for near-seen detection.
4. A `load_target_model(run_id)` function SHALL restore an equivalent predictor without retraining.

### Requirement 5: Substructure attribution (FR-8, STRETCH)

**User Story:** As a scientist, I want to see which substructures drive a target prediction, so
that I can judge whether it is chemically plausible.

#### Acceptance Criteria

1. WHERE requested, THE system SHALL return the top contributing fingerprint bits for a prediction (SHAP values from LightGBM `pred_contrib`) mapped to the atom environments that set them in the query molecule.
2. Attribution SHALL be presented as model explanation, not as mechanism.
