# Requirements Document — 02 Data Pipeline

## Introduction

Turn raw sources into the two modeling tables the models consume: labeled compound–target pairs
(from ChEMBL) and compound × cell-line sensitivity rows (from PharmacoDB DSS3, restricted to
cell lines with CCLE expression). ChEMBL is restored once from its PostgreSQL dump and exported to
Parquet, so the rest of the pipeline is Postgres-free.

**Depends on:** 01 (interim Parquet, bridges, checkpoint decisions).
**Global IDs implemented:** DR-2, DR-4, DR-5, DR-6, FR-6, FR-7, FR-21, NFR-6, NFR-7, NFR-8.

## Requirements

### Requirement 1: One-time ChEMBL restore and Parquet export (DR-5)

**User Story:** As the developer, I want the needed ChEMBL tables as Parquet, so that the
pipeline never depends on a running database.

#### Acceptance Criteria

1. THE export script SHALL extract the dump from `data/raw/chembl/chembl_37_postgresql.tar`, restore it with `pg_restore` into a local PostgreSQL, and export `activities`, `assays`, `target_dictionary`, `target_components`, `component_sequences`, `component_synonyms`, `molecule_dictionary`, and `compound_structures` to `data/interim/chembl/*.parquet`.
2. THE restore SHALL run only after explicit developer approval, because it installs and populates a database.
3. Database credentials SHALL come from environment variables, never source.
4. WHEN the export completes THEN the script SHALL report row counts per table and spot-check one known compound (for example imatinib, CHEMBL941) against expected targets.
5. AFTER export, no later stage SHALL require PostgreSQL.

### Requirement 2: Activity filtering and potency standardization (DR-5, FR-6)

**User Story:** As the developer, I want ChEMBL activities filtered and put on one potency scale,
so that labels are not corrupted by mixed assay types and units.

#### Acceptance Criteria

1. THE filter SHALL keep activities from binding (`B`) and functional (`F`) assays on single-protein human targets with assay `confidence_score` ≥ a configured minimum (default 8).
2. THE filter SHALL drop rows that have a non-null `data_validity_comment` or that are flagged as `potential_duplicate`.
3. THE system SHALL use `pchembl_value` as the standardized potency for `=` relations.
4. WHEN `standard_relation` is `>` or `>=` and the implied pChEMBL is below the active threshold THEN the row SHALL be kept as a measured inactive. Rows with `<` or `<=` SHALL be dropped as ambiguous. The policy SHALL be recorded in provenance.
5. THE system SHALL deduplicate to one row per (InChIKey, target) by taking the median pChEMBL, and SHALL report the before and after counts.
6. Every ChEMBL compound SHALL carry an InChIKey, either `standard_inchi_key` or one computed with RDKit, so that it can be compared across sources.

### Requirement 3: Compound–target label construction (DR-4, DR-6, FR-7)

**User Story:** As the developer, I want an explicit binary label with documented negatives, so
that the target model learns from defensible positives and negatives.

#### Acceptance Criteria

1. A pair SHALL be labeled active WHEN its pChEMBL ≥ the configured threshold (default 6.0, i.e. ≤ 1 µM).
2. Measured inactives SHALL be labeled inactive with `negative_source = "measured"`.
3. WHERE a target has fewer measured inactives than `neg_ratio × positives`, THE system SHALL add constructed negatives sampled from compounds active on other targets but unmeasured on this one, with `negative_source = "sampled"`, and a seeded, recorded sampling ratio.
4. THE system SHALL keep only targets with at least a configured number of positives (default 30) and SHALL report how many targets were dropped.
5. The threshold, negative ratio, minimum positives, and the "absence ≠ inactive" assumption SHALL be written to the table's provenance metadata for the model card.

### Requirement 4: Sensitivity modeling table (DR-4)

**User Story:** As the developer, I want sensitivity rows keyed by structure and restricted to
expression-backed cell lines, so that the sensitivity model can train and generalize by compound.

#### Acceptance Criteria

1. THE table SHALL contain `(inchi_key, pert_id, model_id, dss3)` only for cell lines present in the CCLE default-entry matrix.
2. WHEN multiple `pert_id`s share an InChIKey THEN their rows SHALL be merged per cell line using the median DSS3, and the merge count SHALL be reported.
3. THE system SHALL report the final shape (rows, compounds, cell lines).
4. Rows whose InChIKey is listed in `config.DEMO_HOLDOUT` SHALL be excluded from **both** modeling tables (target pairs and sensitivity) and written to `processed/demo_holdout.parquet` for the spec 09 demo comparison.

### Requirement 5: Integrity and provenance (NFR-6, NFR-7)

**User Story:** As the developer, I want every stage to fail loudly and record its lineage, so
that silent data bugs cannot reach a model.

#### Acceptance Criteria

1. Each stage SHALL assert non-empty output, expected columns, and null-rate bounds, and SHALL raise an error naming the file and filter on violation.
2. Each output Parquet SHALL record its source files (with SHA256), the filters applied, config values, and the module that created it.
3. Each stage SHALL be runnable on its own with `python -m oncora.data_prep.<stage>`.
