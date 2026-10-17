# Requirements Document — 01 Environment & Exploratory Data Analysis

## Introduction

Before any modeling architecture is committed, Oncora needs a reproducible Python environment
and a written, numbers-backed understanding of every raw dataset. This spec covers project
scaffolding, schema/quality profiling, identity-bridge analysis, target and feature-space EDA,
and a formal architecture checkpoint. It ends with a decision log that later specs rely on.

**Global IDs implemented:** DR-1, DR-2 (analysis), DR-3, DR-7 (analysis), FR-12 (analysis),
FR-21 (partial), NFR-2, NFR-3, NFR-8, AC-1.

## Requirements

### Requirement 1: Reproducible project scaffold (FR-21, NFR-2, NFR-8)

**User Story:** As the developer, I want a pinned, installable project skeleton, so that every
later stage runs the same way on any checkout.

#### Acceptance Criteria

1. WHEN `uv sync` is run on a clean checkout THEN the environment SHALL resolve from a committed lockfile on macOS/arm64 without CUDA.
2. THE project SHALL pin rdkit, duckdb, polars, pyarrow, lightgbm, scikit-learn, mlflow, fastapi, uvicorn, streamlit, anthropic, and pytest.
3. THE repository SHALL contain the directory skeleton defined in `.kiro/steering/structure.md`.
4. THE `.gitignore` SHALL exclude `data/raw/`, `data/interim/`, `data/processed/`, `*.tar`, `*.dmp`, `models/` (except `model_card.json`), `mlruns/`, and `.env`.
5. WHEN an import smoke test runs THEN every pinned package SHALL import successfully.
6. THE system SHALL centralize paths, seeds, and thresholds in `src/oncora/config.py`.

### Requirement 2: Schema & quality profile of every raw source (DR-1, AC-1, NFR-3)

**User Story:** As the developer, I want a documented profile of each raw file, so that design
decisions rest on measured facts rather than assumptions.

#### Acceptance Criteria

1. THE profiler SHALL report, per raw file, its columns, dtypes, row and column counts, null rates, identifier spaces, and value ranges.
2. THE profiler SHALL cover the CCLE matrix, the DSS3 sensitivity file, and every file in `data/raw/lincs/StandardAnnotations/`.
3. WHEN profiling `instinfo.txt` (655 MB) or the CCLE matrix (305 MB) THEN the profiler SHALL use streaming or columnar access (DuckDB/Polars) and SHALL NOT load the file as a full pandas DataFrame.
4. THE profiler SHALL treat the literal `""` token in annotation files as null.
5. WHEN the profile is generated THEN `reports/eda_profile.md` SHALL reproduce the grounding numbers in `data-sources.md` (1,775 × 19,221 CCLE; 1,719 distinct ModelIDs; 512,950 DSS3 rows; 972 compounds; 1,204 cell lines).
6. THE profiler SHALL verify raw-file checksums against `data/raw/SHA256SUMS` and SHALL fail if any file is missing or altered.

### Requirement 3: Identity-bridge analysis (DR-2, DR-3, AC-1)

**User Story:** As the developer, I want the compound and cell-line join keys quantified, so
that I know exactly how many training rows the joins can support.

#### Acceptance Criteria

1. THE system SHALL build a compound bridge `pert_id → canonical_smiles, inchi_key, target, moa` from `compoundinfo.txt` and report coverage of sensitivity compounds (expected 972 / 972).
2. WHEN a `pert_id` maps to multiple `compoundinfo` rows THEN the system SHALL report the conflict and apply a documented resolution rule.
3. THE system SHALL build a cell-line bridge between CCLE `ModelID` and sensitivity `cell_line`, using only `IsDefaultEntryForModel = Yes` CCLE rows, and report the overlap (expected 985).
4. THE system SHALL report how many of the 985 overlapping cell lines can be mapped to a primary disease and lineage from the available sources (open item A4).
5. Every unmatched key SHALL be counted and either explained or listed in the report.

### Requirement 4: DSS3 target & CCLE feature-space EDA (FR-12, DR-7)

**User Story:** As the developer, I want the sensitivity target distribution and the gene feature
space characterized, so that I can choose a target transform and a feature-reduction strategy on evidence.

#### Acceptance Criteria

1. THE EDA SHALL characterize DSS3 skew, zero mass, per-compound and per-cell-line variance, and the effect of log1p and rank transforms.
2. THE EDA SHALL propose a sensitive/resistant DSS3 threshold with a rationale.
3. THE EDA SHALL quantify how much DSS3 variance is explained by cell-line identity versus compound identity. This motivates the cell-line-mean baseline.
4. THE EDA SHALL analyze CCLE gene variance and propose a candidate gene subset (variance filter and/or LINCS landmark genes from `geneinfo.txt`, `feature_space = landmark`) with its size and rationale.
5. Figures SHALL be saved under `reports/figures/` and referenced from `reports/eda_profile.md`.

### Requirement 5: Architecture-commit checkpoint

**User Story:** As the developer, I want a formal gate that turns EDA findings into recorded
decisions, so that modeling starts from confirmed assumptions.

#### Acceptance Criteria

1. THE checkpoint SHALL produce `reports/eda_summary.md` summarizing what the data supports.
2. THE checkpoint SHALL record decisions for: DSS3 transform and threshold, gene-subset strategy, `pert_id` conflict rule, and the cell-line → disease source (A4).
3. IF A4 cannot be resolved with the files in the workspace THEN the checkpoint SHALL name the external file required and SHALL block spec 06 Requirement 4 until it is added.
4. WHEN the developer signs off THEN the decisions SHALL be copied into `.kiro/steering/data-sources.md` so that all later specs see them.
