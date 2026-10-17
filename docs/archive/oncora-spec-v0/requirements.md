# Oncora — Requirements

> **Mission:** Speeding cancer drug discovery with explainable AI.
> **Challenge:** eMerge Hackathon Challenge 6 — Compound → Target → Cancer.
>
> Oncora accepts a chemical compound as a **SMILES string** and predicts (1) likely
> protein targets, (2) sensitive cancer cell lines, (3) cancer types warranting
> investigation, and (4) plausible biological explanations. It must work for compounds
> **not seen during training**.
>
> **These are computational research predictions, not validated clinical conclusions.**
> This disclaimer is a first-class product requirement and must appear on every output surface.

---

## 0. How to read this document

Requirements use **EARS**-style phrasing (Easy Approach to Requirements Syntax) and are grouped into:

- **MVP** — must be true for the hackathon submission to be defensible and judgeable.
- **STRETCH** — valuable if time allows; does not block the MVP.
- **FUTURE** — explicitly out of scope for the hackathon (e.g. Challenge 7 generative work).

Each requirement has a stable ID (`FR-x`, `NFR-x`, `DR-x`, `AC-x`) so design and tasks can reference it.

---

## 1. Grounding facts (verified from the actual data in this workspace)

These facts were confirmed by inspecting the real files, and the requirements below depend on them.

| Fact | Value | Why it matters |
|---|---|---|
| CCLE expression matrix | `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`, 1,775 rows × 19,221 cols | Cell-line features; genes named `SYMBOL (ENTREZID)`, values are log1p(TPM) |
| CCLE cell-line key | `ModelID` column = `ACH-xxxxxx` (DepMap) | Join key to sensitivity data |
| Sensitivity labels | `L1000_cp_pertid_Cellosaurus_celllineid_PharmacoDB_dss3.csv`, 512,950 rows | 972 compounds × 1,204 cell lines; target = DSS3 drug sensitivity score |
| Sensitivity score shape | median ≈ 0.087, p90 ≈ 0.41, max ≈ 96, ~20% ≈ 0 | Heavily right-skewed; drives regression vs. classification decisions |
| Cell-line overlap | 985 `ACH-` IDs present in **both** CCLE and sensitivity | Trainable sensitivity rows are bounded by this overlap |
| Compound→structure bridge | `StandardAnnotations/compoundinfo.txt`: `pert_id → canonical_smiles, inchi_key, target, moa` | **972 / 972** sensitivity compounds have SMILES — enables structure-based generalization |
| ChEMBL 37 | `chembl_37_postgresql.tar` → PostgreSQL custom-format dump (~2.16 GB, needs `pg_restore`, PG 10.15 dump) | Source of compound→target supervised labels and SMILES |
| LINCS annotations | `pertinfo.txt`, `cellinfo.txt`, `geneinfo.txt`, `Small_Molecule_Metadata.txt`, `instinfo.txt` | Interpretation/evidence layer; `instinfo.txt` is 655 MB (instance-level) |

**Not yet verified / assumed (flagged for review):**

- ChEMBL schema not read directly (Postgres client tools not installed). Spec relies on the
  well-known ChEMBL 37 schema (`activities`, `assays`, `target_dictionary`, `molecule_dictionary`,
  `compound_structures`, `target_components`, `component_sequences`). **[ASSUMPTION A1 — confirmed
  verifiable at task 1.1]**
- L1000 Level 4 signature matrices (the actual z-score profiles) are **not present** in the
  workspace yet — only annotations. Design treats L1000 signatures as an optional evidence layer,
  never a required inference input. **[ASSUMPTION A2]**

**Confirmed by the developer (locked decisions):**

- **Compute [A3 → RESOLVED]:** Apple **M1 Max, 32 GB RAM, macOS/arm64**. CPU-only pipeline
  (LightGBM + RDKit) is the target; no CUDA. 32 GB permits in-memory work on the CCLE matrix, but
  out-of-core discipline (DuckDB/Polars/chunked) is retained for `instinfo.txt` (655 MB) and the
  512k-row joins (NFR-3). Dependencies pinned to arm64-compatible wheels.
- **ChEMBL handling [A1 approach → APPROVED]:** one-time `pg_restore` into a local PostgreSQL,
  export needed tables to Parquet, then run Postgres-free.
- **Target negatives [DR-6 → DECIDED]:** **binary classification with constructed negatives.**
- **Cancer-type granularity [FR-11 → DECIDED]:** rank by **`cellinfo.primary_disease` + CCLE
  lineage.**

---

## 2. Scope decisions locked for this spec (review these first)

These are the product-level defaults chosen so the spec is actionable now. Each is an explicit,
overridable assumption — challenge any of them and the dependent requirements/tasks adjust.

- **[D1] Both models ship as end-to-end baselines in the MVP.** Compound→Target (ChEMBL) and
  Compound→Sensitivity (PharmacoDB + CCLE). The judging criteria reward unseen-compound validation
  across the pipeline, so neither model is cut. The *bar* for MVP is a defensible baseline with
  honest validation, not a tuned state-of-the-art model.
- **[D2] UI = Streamlit demo + a thin FastAPI inference service.** Fastest path, keeps the
  developer in the Python/data-science loop. React is **FUTURE**.
- **[D3] Compute target = single machine, CPU-only.** Baselines use RDKit fingerprints +
  LightGBM / scikit-learn. PyTorch deep models are **STRETCH/FUTURE** and must never be on the
  MVP critical path. (Revisit if a GPU is confirmed.)
- **[D4] The two models are trained and validated independently.** Their outputs are combined
  only in the interpretation layer. A target prediction is never treated as evidence that a
  sensitivity prediction is correct, or vice versa.
- **[D5] LLM (Claude) is used only as a grounded explanation generator, not as a predictor.**
  A budget of up to ~$100 in Claude credits is available. It is spent exclusively on the
  biological-interpretation layer (turning already-computed structured evidence into readable
  mechanism prose) and, optionally, on development/scaffolding via the Kiro workflow. It is
  **never** used to produce target or sensitivity predictions, and **never** to process raw
  numeric data rows. The interpretation layer SHALL have a deterministic, template-based fallback
  so the full pipeline is reproducible and judgeable with **no API key present** (see FR-13a,
  NFR-9). This keeps "scientific soundness," "validation on unseen compounds," and
  "reproducibility" independent of any LLM.

---

## 3. Functional Requirements

### 3.1 Input & validation

- **FR-1 (MVP).** WHEN a user submits a text string, THE SYSTEM SHALL validate it as a parseable
  SMILES (via RDKit) before any prediction, and SHALL reject invalid input with a clear message.
- **FR-2 (MVP).** WHERE a submitted SMILES fails to parse, THE SYSTEM SHALL NOT attempt
  prediction and SHALL explain what failed (e.g. invalid valence, unparseable token).
- **FR-3 (MVP).** THE SYSTEM SHALL canonicalize valid SMILES (RDKit canonical form) before
  featurization so equivalent inputs yield identical features.
- **FR-4 (STRETCH).** WHERE a submitted compound is structurally near-identical (e.g. Tanimoto
  ≥ 0.99) to a training compound, THE SYSTEM SHALL flag the prediction as "seen/near-seen" so
  the user understands it is not a true unseen-compound result.

### 3.2 Protein-target prediction (ChEMBL)

- **FR-5 (MVP).** THE SYSTEM SHALL predict, for a valid SMILES, a **ranked list of candidate
  protein targets** with a calibrated-as-possible score per target.
- **FR-6 (MVP).** THE SYSTEM SHALL train target prediction only on ChEMBL activities that pass
  documented quality filters (assay type, confidence, standardized units/relation; see DR-5).
- **FR-7 (MVP).** THE SYSTEM SHALL define the supervised target label explicitly (active/inactive
  per compound–target pair under a stated potency threshold) and record the threshold in metadata.
- **FR-8 (STRETCH).** THE SYSTEM SHALL expose the top contributing substructures/features behind
  each target prediction (interpretability).

### 3.3 Cancer sensitivity prediction (PharmacoDB + CCLE)

- **FR-9 (MVP).** THE SYSTEM SHALL predict a cancer cell-line **sensitivity score (or
  sensitive/resistant class)** for a valid SMILES against cell lines that have CCLE expression
  features, and SHALL return a ranked list of most-sensitive cell lines.
- **FR-10 (MVP).** THE SYSTEM SHALL compute sensitivity features from **chemical structure only**
  (fingerprints/descriptors) plus **precomputed cell-line expression features**, so that
  **no experimentally measured LINCS signature is required for a new compound at inference time**
  (hard constraint — traces to the challenge).
- **FR-11 (MVP).** THE SYSTEM SHALL aggregate per-cell-line predictions to a **ranked list of
  cancer types / lineages** using the cell-line→disease mapping (`cellinfo.txt` / CCLE lineage).
- **FR-12 (MVP).** THE SYSTEM SHALL document how the skewed DSS3 target is handled (transform
  and/or thresholded classification) and justify the choice (see ML strategy in design).

### 3.4 Biological interpretation / explainability

- **FR-13 (MVP).** THE SYSTEM SHALL produce, per prediction set, a plain-language **evidence
  summary** linking predicted targets, predicted sensitive lineages, and known annotations
  (`compoundinfo.target`/`moa`) into a plausible mechanism statement.
- **FR-14 (MVP).** THE SYSTEM SHALL cite the evidence source for each claim (which dataset/field
  supports it) and SHALL clearly separate **model prediction** from **database annotation**.
- **FR-15 (MVP).** THE SYSTEM SHALL NOT assert that a target prediction proves a sensitivity
  prediction (or vice versa); combination is correlational evidence only (traces to D4).
- **FR-13a (MVP).** THE SYSTEM SHALL generate the evidence summary from a **structured evidence
  bundle** (predicted targets + scores, predicted sensitive lineages + scores, matching
  `compoundinfo` target/moa annotations, optional target-gene expression in those lineages). WHEN
  no LLM API key is configured, THE SYSTEM SHALL render the summary from a deterministic template
  over that bundle. WHERE a Claude API key is configured, THE SYSTEM MAY instead produce prose
  from the **same bundle**, constrained to only restate facts present in the bundle and to cite
  each claim's source field.
- **FR-16 (STRETCH).** WHERE CCLE expression shows the predicted target gene is highly expressed
  in the predicted sensitive lineages, THE SYSTEM SHALL surface that as corroborating evidence.
- **FR-16a (MVP).** THE SYSTEM SHALL NOT send raw dataset rows, SMILES-derived predictions beyond
  the finished evidence bundle, or any bulk numeric data to an external LLM; only the compact,
  already-computed evidence bundle for a single query may be sent.

### 3.5 Application interface

- **FR-17 (MVP).** THE SYSTEM SHALL provide a UI that accepts a SMILES string and displays the
  molecule render, ranked target predictions, ranked sensitive cell lines/cancer types, and the
  evidence summary.
- **FR-18 (MVP).** THE SYSTEM SHALL display the research-disclaimer prominently on every results view.
- **FR-19 (MVP).** THE SYSTEM SHALL expose a programmatic inference API (FastAPI) that the UI
  consumes, returning the same ranked predictions as structured JSON.
- **FR-20 (STRETCH).** THE SYSTEM SHALL accept a small batch of SMILES and return results for each.

### 3.6 Reproducibility & operations

- **FR-21 (MVP).** THE SYSTEM SHALL make every data-prep and training step reproducible from raw
  inputs via scripted, ordered pipeline stages with pinned dependencies and fixed random seeds.
- **FR-22 (MVP).** THE SYSTEM SHALL version model artifacts together with the metadata needed to
  reproduce them (training data snapshot/hash, feature config, split strategy, metrics, seed).
- **FR-23 (MVP).** THE SYSTEM SHALL load models at inference from versioned artifacts without
  retraining.

---

## 4. Non-Functional Requirements

- **NFR-1 Scientific honesty (MVP).** Reported performance SHALL come from a held-out split that
  prevents compound leakage (see AC-2). Optimistic random-row splits are prohibited as the
  headline metric.
- **NFR-2 Reproducibility (MVP).** A clean checkout + documented steps SHALL reproduce the
  pipeline deterministically (seeded) on CPU.
- **NFR-3 Resource ceiling (MVP).** Training and inference SHALL run on a single CPU machine
  within commodity RAM. Operations on the 305 MB CCLE matrix and 655 MB `instinfo.txt` SHALL use
  streaming/columnar access (DuckDB/Polars/chunked) rather than naive full-DataFrame loads. **[A3: confirm available RAM]**
- **NFR-4 Latency (STRETCH).** Single-SMILES inference SHALL return within a few seconds on CPU
  once models are loaded.
- **NFR-5 Maintainability / learning (MVP).** Each pipeline stage SHALL be independently runnable
  and testable, with a short rationale comment explaining the data-science technique used (this
  project doubles as a learning artifact).
- **NFR-6 Error handling (MVP).** Data-prep stages SHALL fail loudly with actionable messages on
  schema drift, empty joins, or unexpected nulls — never silently emit empty training sets.
- **NFR-7 Data provenance (MVP).** Every derived table SHALL record which raw files and filters
  produced it.
- **NFR-8 Security/safety (MVP).** The ChEMBL dump and large raw files SHALL NOT be committed to
  git; secrets (DB credentials, LLM API keys) SHALL come from env/config, never source.
- **NFR-9 LLM-optional reproducibility (MVP).** Every judgeable artifact (metrics, rankings,
  template evidence summary) SHALL be reproducible with no network and no LLM key. LLM prose is an
  enhancement layer only; its absence SHALL NOT change any prediction, metric, or ranking.
- **NFR-10 LLM cost control (MVP).** LLM calls SHALL be per-query and on-demand (one bounded call
  per explanation), with the model and token budget configurable, so total spend stays within the
  ~$100 ceiling. No batch/bulk LLM processing of datasets.

---

## 5. Data Requirements

- **DR-1 (MVP).** Provide a documented schema profile for each raw dataset (columns, types,
  row/col counts, null rates, identifier spaces) as the first deliverable (EDA).
- **DR-2 (MVP).** Establish and document the **compound identity bridge**: `pert_id` (BRD) →
  `canonical_smiles`/`inchi_key` via `compoundinfo.txt`; ChEMBL `molregno` →
  `compound_structures.canonical_smiles`. Use InChIKey as the cross-source dedup key.
- **DR-3 (MVP).** Establish and document the **cell-line identity bridge**: CCLE `ModelID` (ACH)
  ↔ sensitivity `cell_line` (ACH) ↔ `cellinfo.txt` (Cellosaurus / disease). Report overlap counts.
- **DR-4 (MVP).** Produce the **modeling tables**: (a) compound→target pairs with labels from
  ChEMBL; (b) compound×cell-line sensitivity rows joined to CCLE features. Report final shapes.
- **DR-5 (MVP).** Document ChEMBL activity filtering: keep binding/functional assays, standard
  relations `=` (and document handling of `>`/`<` censored values), standard units normalized to
  a common potency (e.g. pChEMBL / nM), drop invalid/duplicated measurements with a stated
  dedup rule. **[A1 depends on ChEMBL schema]**
- **DR-6 (MVP) — DECIDED: binary with constructed negatives.** Absence of a compound–target
  activity is not a confirmed negative. THE SYSTEM SHALL frame compound→target as **binary
  classification**, deriving positives from measured actives (pChEMBL ≥ threshold) and
  **constructing negatives** from (a) measured inactives (`>`-threshold / below-cutoff rows) where
  available and (b) a documented decoy/negative-sampling scheme for targets lacking enough measured
  inactives (e.g. compounds active elsewhere but with no measured activity on the target, with the
  assumption and sampling ratio recorded). The construction rule and its limitations SHALL be
  documented in the model card.
- **DR-7 (STRETCH).** Reduce CCLE's 19,221 genes to a documented feature subset (variance filter,
  or landmark/pathway genes) and justify the selection to control dimensionality and leakage risk.

---

## 6. Acceptance Criteria (measurable)

- **AC-1 (MVP).** Running the EDA stage produces a written data-profile report covering DR-1–DR-4
  with the real numbers from this workspace (reproduces the grounding table in §1).
- **AC-2 (MVP) — leakage-safe validation.** Both models report their headline metric under a
  **compound-disjoint split** (no compound, by InChIKey, appears in both train and test). For the
  sensitivity model, a **scaffold-based split** (Bemis–Murcko) is additionally reported.
- **AC-3 (MVP) — baseline comparison.** Each model reports metrics against at least one trivial
  baseline (e.g. target: most-frequent/prior; sensitivity: global-mean / cell-line-mean
  predictor). The model must beat the trivial baseline on the leakage-safe split to be accepted.
- **AC-4 (MVP) — target metrics.** Compound→target reports ranking/classification metrics
  appropriate to the label framing (e.g. per-target ROC-AUC / PR-AUC, and top-k retrieval), with
  the metric choice justified against class imbalance.
- **AC-5 (MVP) — sensitivity metrics.** Compound→sensitivity reports regression metrics
  (e.g. Spearman ρ, RMSE on the chosen target transform) **and** a thresholded
  sensitive/resistant AUC, both on the compound-disjoint split.
- **AC-6 (MVP) — unseen-compound demo.** A scripted demo predicts for at least one SMILES held
  out of all training (true unseen compound) and the output is inspected end-to-end.
- **AC-7 (MVP) — interpretation.** For the demo compound, the evidence summary names the
  predicted target(s), the predicted sensitive lineage(s), and the supporting annotation/field,
  with model-vs-annotation clearly separated.
- **AC-8 (MVP) — reproducibility.** A documented command sequence rebuilds derived tables and
  reloads a saved model to reproduce a reported metric within seeded tolerance.
- **AC-9 (MVP) — disclaimer.** The research disclaimer is present on every results surface
  (UI + API response field).

---

## 7. Out of scope for the hackathon (FUTURE)

- **F-1.** Generative design / ranking of novel candidates (Challenge 7). Architecture leaves room
  for it (structure-in, ranked-out), but it is not built.
- **F-2.** React production frontend (Streamlit is the MVP UI).
- **F-3.** GPU/deep-learning target or sensitivity models (GNNs, ChemBERTa) beyond optional
  benchmarking.
- **F-4.** Live/online retraining, user accounts, persistence of user queries.
- **F-5.** Clinical, dosing, or any patient-facing interpretation.

---

## 8. Open questions — all resolved by the developer

1. **Compute [A3]:** ✅ Resolved — M1 Max, 32 GB RAM, CPU-only (no CUDA). See §1.
2. **ChEMBL restore [A1]:** ✅ Approved — one-time `pg_restore` → Parquet export → Postgres-free.
3. **Negative evidence [DR-6]:** ✅ Decided — **binary classification with constructed negatives**.
4. **Cancer-type granularity [FR-11]:** ✅ Decided — **`cellinfo.primary_disease` + CCLE lineage**.

Only remaining assumption: **A2** (L1000 Level 4 signature matrices absent → optional evidence
only). No action needed unless signature matrices are later added.
