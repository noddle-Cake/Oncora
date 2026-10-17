# Requirements Document — 09 Evaluation Hardening & Submission

## Introduction

Turn the working system into a judgeable submission: an end-to-end demo on a truly unseen
compound, a verified reproducibility path from raw data to a reported metric, and a writeup that
maps every judging criterion to concrete evidence.

**Depends on:** 01–08.
**Global IDs implemented:** AC-6, AC-7, AC-8, NFR-2, FR-21.

## Requirements

### Requirement 1: Unseen-compound demo (AC-6, AC-7)

**User Story:** As a judge, I want to see the system predict for a compound it never trained on,
so that I can verify the core claim directly.

#### Acceptance Criteria

1. THE demo SHALL use at least one compound whose InChIKey is absent from **both** models' training sets, verified programmatically against the stored training indexes.
2. Preferably, the demo compound SHALL have known annotations or literature targets, so that predictions can be checked against known biology. The InChIKey SHALL be excluded before training if necessary.
3. `scripts/demo_unseen.py` SHALL run the full pipeline (via the API or in-process) and save the full output to `reports/demo/`.
4. THE saved evidence summary SHALL name the predicted targets, the sensitive lineages, and the supporting annotation fields, with model and annotation separated (AC-7).
5. THE demo output SHALL be reviewed and a short commentary written (what matches known biology, what does not).

### Requirement 2: Reproducibility pass (AC-8, NFR-2, FR-21)

**User Story:** As a judge, I want a documented command sequence that rebuilds everything and
reproduces a reported metric, so that I can trust the results are not one-off.

#### Acceptance Criteria

1. THE README SHALL document an ordered command sequence: environment, checksum verification, each data stage, featurization, training, evaluation, serving.
2. WHEN the sequence is run on a fresh checkout with the raw files in `data/raw/` THEN it SHALL reproduce each model's headline metric within the seeded tolerance (exact match expected; report any nondeterminism).
3. A `make reproduce` (or `scripts/reproduce.sh`) target SHALL run the sequence end to end.
4. THE pass SHALL be run with no network and no `ANTHROPIC_API_KEY`, and SHALL produce all judgeable artifacts (NFR-9).

### Requirement 3: Submission writeup

**User Story:** As a judge, I want a concise report that maps each criterion to evidence, so that
I can evaluate quickly.

#### Acceptance Criteria

1. `reports/submission.md` SHALL map each criterion (scientific soundness, validation on unseen compounds, biological insight, originality, reproducibility) to specific artifacts and numbers.
2. THE writeup SHALL report headline metrics on the compound-disjoint (and, for sensitivity, scaffold-disjoint) splits next to the baselines, with CIs.
3. THE writeup SHALL state the limitations honestly: constructed negatives, the A4 disease-map source, the skewed DSS3 target, no L1000 signatures, CPU-only baselines.
4. THE writeup SHALL carry the research disclaimer.
