# Requirements Document — 07 Biological Interpretation Layer

## Introduction

Combine the outputs of the two independent models with database annotations into a structured,
deterministic **evidence bundle**. Render it as a plain-language evidence summary, either with a
deterministic template (always available) or, optionally, with Claude, constrained to restate the
same bundle. Every claim cites its source. Model predictions and database annotations are kept
visibly separate.

**Depends on:** 05, 06. Steering: `llm-grounding.md`.
**Global IDs implemented:** FR-13, FR-13a, FR-14, FR-15, FR-16 (STRETCH), FR-16a, NFR-9, NFR-10, AC-7.

## Requirements

### Requirement 1: Evidence bundle (FR-13a, FR-14)

**User Story:** As the developer, I want one structured, deterministic object that holds all the
evidence for a query, so that any explanation can be generated from it and checked against it.

#### Acceptance Criteria

1. THE builder SHALL assemble, for one query: the input and canonical SMILES, the InChIKey, the top targets with scores, the top sensitive cell lines with scores, the top cancer types with scores and support, the near-seen flags, and the model run IDs.
2. THE builder SHALL attach database annotations found by **exact InChIKey match**: `compoundinfo.target` and `moa`, and ChEMBL known actives above the threshold. Each SHALL be labeled `kind = "annotation"` with its source file and field.
3. Every predicted item SHALL be labeled `kind = "prediction"` with its model and run ID.
4. WHEN no annotation matches (a novel compound) THEN the bundle SHALL say so explicitly rather than omit the section.
5. The bundle SHALL be deterministic: the same query and the same artifacts give a byte-identical serialized bundle.
6. Every field SHALL have a stable path (for example, `predictions.targets[0]`) that a claim can cite.

### Requirement 2: Cross-evidence signals without overclaiming (FR-13, FR-15, FR-16)

**User Story:** As a scientist, I want to see where predictions and annotations agree, framed as
correlational, so that I get insight without false certainty.

#### Acceptance Criteria

1. THE builder SHALL compute agreement between predicted target gene symbols and annotated targets, and report which ones overlap.
2. WHERE a predicted target's gene is in the CCLE matrix, THE builder SHALL report its expression percentile in the predicted sensitive lines versus all lines (FR-16, STRETCH).
3. All cross-evidence SHALL be phrased and typed as correlational. No field or rendered sentence SHALL assert that one model's output proves another's.
4. WHERE the L1000 Level 4 `trt_cp` GCTX is available AND the query compound (by InChIKey → `pert_id` via `compoundinfo`/`instinfo`) has one or more measured signatures, THE builder MAY attach, as `kind = "annotation"` evidence, a compact summary of that compound's own measured expression response (e.g. top up/down-regulated genes), clearly labeled as **measured signature for this compound**, not a prediction (FR-16, STRETCH). This path SHALL be skipped silently when the GCTX is absent or the compound has no profile, and SHALL never be a precondition for producing the bundle. It reads only the needed `rid`/`cid` slab, never the whole matrix.

### Requirement 3: Template renderer (FR-13a, NFR-9, AC-7)

**User Story:** As a judge, I want an evidence summary that works offline with no API key, so
that the interpretation is fully reproducible.

#### Acceptance Criteria

1. THE template renderer SHALL produce a summary that names the predicted targets, the sensitive lineages, and any supporting annotations, with a source citation on every claim.
2. THE summary SHALL present predictions and annotations in separate, labeled sections.
3. THE summary SHALL end with the research disclaimer.
4. THE renderer SHALL run with no network access and no API key, and SHALL be deterministic.

### Requirement 4: Claude grounded renderer (FR-13a, FR-16a, NFR-10, STRETCH)

**User Story:** As a user, I want more readable mechanism prose when an API key is available,
without losing grounding or reproducibility.

#### Acceptance Criteria

1. WHERE `ANTHROPIC_API_KEY` is set and the LLM is enabled, THE system SHALL send **only** the serialized bundle for the single query, with no raw dataset rows or bulk data.
2. THE prompt SHALL instruct Claude to restate only facts in the bundle, cite a bundle path for each claim, keep predictions and annotations separate, and avoid causal language between the models.
3. THE response SHALL be structured (a list of claims, each with text and cited bundle paths) and validated: every cited path SHALL exist in the bundle and every number in the text SHALL appear in the bundle. Claims that fail validation SHALL be dropped. If none survive, the system SHALL fall back to the template.
4. IF the key is absent, the call errors, or it times out (configurable, default 20 s) THEN the system SHALL silently use the template renderer.
5. Each explanation SHALL use at most one API call with a configured `max_tokens`. The model ID SHALL be configurable. Token usage SHALL be logged to track spend against the ~$100 budget.
6. The disclaimer SHALL be appended by the system, not left to the LLM.
