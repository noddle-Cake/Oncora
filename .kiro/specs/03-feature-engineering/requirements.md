# Requirements Document — 03 Feature Engineering

## Introduction

Build the two feature sources: **compound features derived only from structure** (RDKit), and
**precomputed cell-line features** from CCLE expression. Structure-only compound features are what
make prediction for unseen compounds possible. Inference must use exactly the same featurization as training.

**Depends on:** 02.
**Global IDs implemented:** FR-1, FR-2, FR-3, FR-4, FR-10 (feature side), DR-7.

## Requirements

### Requirement 1: SMILES validation (FR-1, FR-2)

**User Story:** As a user, I want invalid SMILES rejected with a clear reason, so that I never
get predictions for a molecule the system could not read.

#### Acceptance Criteria

1. WHEN a string is submitted THEN the system SHALL parse it with RDKit before any featurization.
2. IF parsing or sanitization fails THEN the system SHALL return `valid = false` and a human-readable reason (for example, an invalid valence on a named atom, or an unparseable token), and SHALL NOT featurize.
3. WHEN the input is empty or only whitespace THEN the system SHALL reject it with a specific message.

### Requirement 2: Canonicalization and identity (FR-3)

**User Story:** As the developer, I want equivalent SMILES to produce identical features, so that
results do not depend on how a molecule was written.

#### Acceptance Criteria

1. THE system SHALL standardize valid molecules (strip salts and solvents, keep the largest fragment, neutralize charges) and emit RDKit canonical SMILES and an InChIKey.
2. Canonicalization SHALL be idempotent: canonicalizing a canonical SMILES returns the same string.
3. Two different SMILES for the same molecule SHALL produce identical feature vectors.

### Requirement 3: Compound featurizer (FR-10)

**User Story:** As the developer, I want a fixed, versioned compound feature vector, so that both
models consume identical structure features at train and inference time.

#### Acceptance Criteria

1. THE featurizer SHALL produce ECFP4 (Morgan radius 2, 2,048 bits) and a documented set of RDKit physicochemical descriptors.
2. THE featurizer SHALL extract the Bemis–Murcko scaffold (as canonical SMILES) for scaffold splitting.
3. THE feature configuration (radius, bit count, descriptor list, RDKit version) SHALL be serialized and stored with each model artifact.
4. THE featurizer SHALL NOT use any LINCS signature or other measured biological profile.
5. WHEN featurizing the ~1k sensitivity compounds or the ChEMBL compounds THEN results SHALL be cached to Parquet keyed by InChIKey.

### Requirement 4: Near-seen detection (FR-4, STRETCH)

**User Story:** As a user, I want to know when my compound is nearly identical to a training
compound, so that I do not mistake a memorized result for a real unseen-compound prediction.

#### Acceptance Criteria

1. THE system SHALL compute the maximum Tanimoto similarity (ECFP4) between a query and each model's training compounds.
2. WHERE max Tanimoto ≥ 0.99 (configurable) THE system SHALL set `near_seen = true` and report the nearest training compound.

### Requirement 5: Cell-line expression features (DR-7)

**User Story:** As the developer, I want a compact, fixed cell-line feature matrix, so that the
sensitivity model is not overwhelmed by 19k genes and does not leak.

#### Acceptance Criteria

1. THE builder SHALL produce one feature row per CCLE default-entry cell line, using the gene-subset strategy recorded at the spec 01 checkpoint.
2. WHERE the subset is chosen by a data-driven criterion (for example, variance) THE selection SHALL be fit on training cell lines only and stored as an explicit gene list.
3. THE gene list and any scaling parameters SHALL be persisted as feature config for inference parity.
4. THE matrix SHALL cover all 985 cell lines that overlap with DSS3, and its feature count SHALL match the config.
