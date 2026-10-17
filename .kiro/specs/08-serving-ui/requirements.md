# Requirements Document — 08 Serving & UI

## Introduction

Expose the full pipeline through a typed FastAPI inference service and a Streamlit demo UI that
consumes it. Models load once from versioned artifacts. The research disclaimer is present on
every results surface.

**Depends on:** 05, 06, 07.
**Global IDs implemented:** FR-1, FR-2 (surface), FR-17, FR-18, FR-19, FR-20 (STRETCH), FR-23, NFR-4 (STRETCH), AC-9.

## Requirements

### Requirement 1: Inference API (FR-19, FR-23, AC-9)

**User Story:** As a developer or the UI, I want one endpoint that returns every prediction for a
SMILES as typed JSON, so that clients get a single consistent contract.

#### Acceptance Criteria

1. THE service SHALL load the target model, sensitivity model, feature configs, and annotation tables once at startup from versioned artifacts, without retraining.
2. `POST /predict {smiles}` SHALL return `valid`, `canonical_smiles`, `inchi_key`, `targets[]`, `sensitive_cell_lines[]`, `cancer_types[]`, `evidence_summary`, `near_seen`, `model_versions`, and `disclaimer`.
3. IF the SMILES is invalid THEN the service SHALL return HTTP 422 with `valid = false`, the parse error reason, and the disclaimer, and SHALL NOT run models.
4. `GET /health` SHALL report readiness and the loaded model run IDs.
5. THE `disclaimer` field SHALL be present on every successful and every validation-error response.
6. WHERE the disease map is unavailable (A4), `cancer_types` SHALL be an empty list with an explanatory note, and the other outputs SHALL still be returned.

### Requirement 2: Streamlit UI (FR-17, FR-18)

**User Story:** As a judge, I want to paste a SMILES and see the molecule, rankings, and
explanation on one screen, so that I can evaluate the system interactively.

#### Acceptance Criteria

1. THE UI SHALL accept a SMILES string, call the API, and render the 2D molecule (RDKit), ranked targets, ranked sensitive cell lines, ranked cancer types, and the evidence summary.
2. THE research disclaimer SHALL be visible on every results view without scrolling past the results.
3. Prediction and annotation content SHALL be visually distinguished.
4. WHERE `near_seen` is true, THE UI SHALL show a clear notice with the nearest training compound.
5. Invalid SMILES SHALL show the API's reason inline.
6. THE UI SHALL offer a few example SMILES, including at least one held-out compound from the spec 09 demo.

### Requirement 3: Latency (NFR-4, STRETCH)

**User Story:** As a user, I want results in a few seconds, so that the demo feels interactive.

#### Acceptance Criteria

1. Single-SMILES `/predict` SHALL respond within 5 s on the M1 Max once loaded, excluding the optional Claude call.
2. THE response SHALL include per-stage timings for diagnosis.

### Requirement 4: Batch input (FR-20, STRETCH)

**User Story:** As a user, I want to submit several SMILES at once, so that I can compare compounds.

#### Acceptance Criteria

1. `POST /predict/batch {smiles: [...]}` SHALL accept up to a configured maximum (default 20) and return one result per input, in order, with invalid inputs reported individually.
2. Batch requests SHALL use the template renderer only, to respect the LLM cost controls (NFR-10).
