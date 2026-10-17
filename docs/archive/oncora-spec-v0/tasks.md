# Oncora — Implementation Roadmap

> Phased plan. **Phase 0 (EDA) comes before any modeling-architecture commitment**, per the
> handoff. Each task states: **What**, **Why** (scientific/technical), **Prereqs**, **Verify**
> (how completion is checked), **Learn** (data-science concept), **Deliverable**.
>
> Checkboxes track status. Nothing here is executed until the spec is approved. Tasks reference
> requirement IDs (`FR-/NFR-/DR-/AC-`) from `requirements.md`.
>
> Legend: **[MVP]** required for submission · **[STRETCH]** if time allows · **[FUTURE]** post-hackathon.

---

## Phase 0 — Environment & Exploratory Data Analysis (do this first)

- [ ] **0.1 Project & environment scaffold [MVP]**
  - **What:** `pyproject.toml` (uv), pinned deps (rdkit, duckdb, polars, lightgbm, scikit-learn,
    mlflow, fastapi, streamlit, anthropic, pytest), repo skeleton from design §3, `.gitignore`
    excluding `data/raw`, the ChEMBL tar/dmp, `models/`, `mlruns/`.
  - **Why:** Reproducibility starts at dependency pinning (NFR-2); large files must never hit git (NFR-8).
  - **Prereqs:** none. **Verify:** clean install resolves; `import` smoke test passes; `git status`
    shows no large data files tracked.
  - **Learn:** reproducible Python environments & lockfiles. **Deliverable:** installable repo skeleton.

- [ ] **0.2 Schema & quality profiling of every raw source [MVP]** *(DR-1, AC-1)*
  - **What:** DuckDB/Polars profilers producing, per file: columns, dtypes, row/col counts, null
    rates, identifier spaces, value ranges. Cover CCLE, PharmacoDB DSS3, and all
    `StandardAnnotations/*`. Stream the 305 MB / 655 MB files; never full-load `instinfo.txt`.
  - **Why:** You cannot choose an architecture before you know the data (handoff mandate); reveals
    nulls/skew/id mismatches early.
  - **Prereqs:** 0.1. **Verify:** report reproduces the §1 grounding numbers (1,775×19,221; 972
    compounds; DSS3 median≈0.087; etc.).
  - **Learn:** data profiling, columnar/out-of-core processing. **Deliverable:** `reports/eda_profile.md` + notebook.

- [ ] **0.3 Identity-bridge analysis [MVP]** *(DR-2, DR-3, AC-1)*
  - **What:** Build & quantify the compound bridge (`pert_id→SMILES/InChIKey` via `compoundinfo`)
    and cell-line bridge (CCLE `ModelID` ↔ PharmacoDB `cell_line` ↔ `cellinfo`). Report coverage.
  - **Why:** Joins are the backbone (design §4.1); confirms the 972/972 SMILES and 985 cell-line
    overlap hold and are usable.
  - **Prereqs:** 0.2. **Verify:** coverage counts match (972 with SMILES; 985 overlap); zero
    unexplained unmatched keys, or each exception documented.
  - **Learn:** entity resolution, structure identity via InChIKey. **Deliverable:** bridge tables + coverage report.

- [ ] **0.4 DSS3 target & CCLE feature-space EDA [MVP]** *(DR-7, FR-12)*
  - **What:** Characterize DSS3 distribution (skew, zero-mass), test transforms (log1p/rank);
    analyze CCLE gene variance to scope a feature subset (variance filter / landmark genes).
  - **Why:** Drives the regression-vs-classification and transform decisions (design §5.2);
    19,221 genes need principled reduction to limit overfitting.
  - **Prereqs:** 0.3. **Verify:** documented transform recommendation + a candidate gene-subset
    size with rationale.
  - **Learn:** distribution diagnostics, variance-stabilizing transforms, dimensionality/feature
    selection. **Deliverable:** section in `reports/eda_profile.md` with figures.

- [ ] **0.5 EDA synthesis & architecture-commit checkpoint [MVP]**
  - **What:** One-page summary of what the data supports; confirm/adjust the design assumptions
    (A1–A3, DR-6 framing, FR-11 granularity).
  - **Why:** Formal gate before building models (handoff: inspect before committing architecture).
  - **Prereqs:** 0.2–0.4. **Verify:** developer review sign-off; open questions in requirements §8 resolved.
  - **Learn:** translating EDA into modeling decisions. **Deliverable:** `reports/eda_summary.md` + decisions log.

---

## Phase 1 — Reproducible data preparation

- [ ] **1.1 ChEMBL restore → Parquet export [MVP]** *(DR-5)*
  - **What:** One-time `pg_restore` of `chembl_37_postgresql.dmp` into a local PG; export the
    needed tables (`activities`, `assays`, `target_dictionary`, `molecule_dictionary`,
    `compound_structures`, `target_components`, `component_sequences`) to Parquet; then work
    Postgres-free. **Needs explicit approval (DB setup) before running.**
  - **Why:** Decouples the pipeline from a running DB → reproducible, fast (design §2).
  - **Prereqs:** 0.5; PostgreSQL available. **Verify:** Parquet row counts sane vs. ChEMBL 37 docs;
    spot-check a known compound's activities.
  - **Learn:** relational→columnar ETL, schema-targeted extraction. **Deliverable:** `interim/chembl_*.parquet` + export script.

- [ ] **1.2 ChEMBL activity filtering & potency standardization [MVP]** *(DR-5, FR-6, FR-7)*
  - **What:** Filter to binding/functional assays with adequate confidence; normalize units to
    pChEMBL; handle `=`/`>`/`<` relations per policy; dedup by (compound, target, assay) with a
    stated rule.
  - **Why:** ChEMBL mixes assay types/units; naive use corrupts labels (design §5.1).
  - **Prereqs:** 1.1. **Verify:** before/after counts; unit histogram collapses to pChEMBL;
    censored-value handling unit-tested.
  - **Learn:** assay heterogeneity, standardized potency, censored measurements. **Deliverable:** `interim/chembl_activities.parquet`.

- [ ] **1.3 Build modeling tables [MVP]** *(DR-4, DR-6)*
  - **What:** `processed/target_pairs.parquet` (compound×target + binary label + negative-handling
    policy) and `processed/sensitivity.parquet` (compound×cell-line×DSS3 restricted to the 985
    expression-backed cell lines).
  - **Why:** These are the exact inputs the two models consume; final shapes must be known.
  - **Prereqs:** 1.2, 0.3. **Verify:** non-empty, expected columns, documented shapes; loud failure
    on empty joins (NFR-6).
  - **Learn:** label construction, positive/negative framing, join integrity. **Deliverable:** two processed tables + provenance.

---

## Phase 2 — Feature engineering

- [ ] **2.1 Compound featurizer (RDKit) [MVP]** *(FR-1, FR-3, DR-2)*
  - **What:** validate→canonicalize→ECFP4 (2048 bit) + physchem descriptors; Bemis–Murcko scaffold
    extraction; Tanimoto-to-training utility (near-seen flag, FR-4).
  - **Why:** Structure-only features are what enable unseen-compound prediction (design §5).
  - **Prereqs:** 1.3. **Verify:** unit tests on known molecules; invalid SMILES rejected (FR-2);
    canonicalization idempotent.
  - **Learn:** molecular fingerprints, scaffolds, Tanimoto similarity. **Deliverable:** `src/oncora/chem` + tests.

- [ ] **2.2 Cell-line expression feature builder [MVP]** *(DR-7)*
  - **What:** From CCLE, build the per-cell-line feature matrix using the gene subset chosen in
    0.4; persist feature config for inference parity.
  - **Why:** Cell-line side of the sensitivity model; fixed feature config prevents train/infer skew.
  - **Prereqs:** 0.4, 1.3. **Verify:** matrix covers the 985 cell lines; feature count matches config.
  - **Learn:** high-dim feature selection, train/inference feature parity. **Deliverable:** expression feature matrix + config.

---

## Phase 3 — Models (baselines first, honest validation)

- [ ] **3.1 Leakage-safe splitters + trivial baselines [MVP]** *(NFR-1, AC-2, AC-3)*
  - **What:** InChIKey-disjoint splitter, Bemis–Murcko scaffold splitter, compound-grouped CV;
    trivial baselines (prior/most-frequent; global-mean & cell-line-mean).
  - **Why:** This is the credibility backbone — build and test it *before* any model so metrics are
    trustworthy from day one (design §5.3).
  - **Prereqs:** 2.1. **Verify:** automated tests assert zero InChIKey/scaffold overlap across
    folds; baselines produce metrics.
  - **Learn:** data leakage, grouped/scaffold splitting, baseline flooring. **Deliverable:** `src/oncora/evaluation` + anti-leakage tests.

- [ ] **3.2 Compound→Target baseline model [MVP]** *(FR-5, FR-6, FR-7, AC-4)*
  - **What:** LightGBM (per-target/multi-label) on ECFP+descriptors; logistic-regression baseline;
    MLflow logging; model card.
  - **Why:** Cleanest supervised problem; anchors the pipeline.
  - **Prereqs:** 3.1, 2.1. **Verify:** per-target ROC/PR-AUC + top-k on compound-disjoint split;
    beats trivial baseline (AC-3/AC-4).
  - **Learn:** multi-label classification, ROC vs PR under imbalance, GBDT. **Deliverable:** target model artifact + eval report.

- [ ] **3.3 Compound→Sensitivity baseline model [MVP]** *(FR-9, FR-10, FR-12, AC-5)*
  - **What:** LightGBM regressor on concat(compound, cell-line) features with the DSS3 transform;
    parallel thresholded classifier; cell-line-mean baseline; MLflow + model card. Enforce
    structure-only compound features (no LINCS signature) — FR-10.
  - **Why:** Core product capability; must generalize to unseen compounds.
  - **Prereqs:** 3.1, 2.1, 2.2. **Verify:** Spearman ρ + RMSE(transform) + thresholded AUC on
    **both** compound-disjoint and scaffold splits; beats cell-line-mean baseline (AC-5).
  - **Learn:** regression on skewed targets, rank correlation, grouped generalization. **Deliverable:** sensitivity model artifact + eval report.

- [ ] **3.4 Cancer-type aggregation [MVP]** *(FR-11)*
  - **What:** Aggregate per-cell-line predictions → ranked cancer types/lineages via
    `cellinfo.primary_disease` + CCLE lineage.
  - **Why:** Delivers the "which cancer types" output (challenge item 3).
  - **Prereqs:** 3.3. **Verify:** ranked list is sensible for a known drug on the unseen-compound demo.
  - **Learn:** prediction aggregation, mapping to ontology/lineage. **Deliverable:** aggregation module.

---

## Phase 4 — Interpretation (with optional Claude)

- [ ] **4.1 Evidence-bundle builder [MVP]** *(FR-13a, FR-14, FR-15)*
  - **What:** Assemble the structured bundle (top targets+scores, top lineages+scores,
    `compoundinfo` target/moa annotations clearly labeled, optional target-gene expression in
    predicted lineages). Keep model-vs-annotation separated.
  - **Why:** Deterministic grounding source for any explanation (design §6).
  - **Prereqs:** 3.2, 3.3, 3.4. **Verify:** bundle schema test; every field traces to a source.
  - **Learn:** evidence modeling, separating prediction from annotation. **Deliverable:** `EvidenceBundle` + builder.

- [ ] **4.2 Template renderer (no-LLM fallback) [MVP]** *(FR-13a, NFR-9)*
  - **What:** Deterministic text summary from the bundle with per-claim source citations + disclaimer.
  - **Why:** Guarantees offline reproducibility and a judgeable artifact with no API key.
  - **Prereqs:** 4.1. **Verify:** runs with no network/key; output cites every claim (AC-7).
  - **Learn:** template-based NLG, reproducibility discipline. **Deliverable:** template renderer.

- [ ] **4.3 Claude grounded renderer (optional) [STRETCH]** *(FR-13a, FR-16a, NFR-10)*
  - **What:** Anthropic client that takes the **same bundle**, prompt-constrained to restate-only +
    cite source fields; per-query, token-budgeted; key from env. Falls back to template if no key.
  - **Why:** Nicer prose for the demo within the ~$100 budget, without compromising reproducibility.
  - **Prereqs:** 4.2. **Verify:** with key, prose contains no facts absent from the bundle (manual
    + a factuality check); without key, silently uses template.
  - **Learn:** grounded LLM prompting, guardrails, cost control. **Deliverable:** optional LLM renderer.

---

## Phase 5 — Serving & UI

- [ ] **5.1 FastAPI inference service [MVP]** *(FR-19, FR-23, AC-9)*
  - **What:** Load versioned artifacts once; `POST /predict {smiles}` → ranked targets, sensitive
    cell lines, cancer types, evidence summary, near-seen flag, disclaimer; `GET /health`.
  - **Why:** Clean contract the UI consumes; disclaimer as a response field.
  - **Prereqs:** 3.2–3.4, 4.2. **Verify:** contract test; valid/invalid SMILES behave per FR-1/FR-2;
    disclaimer present.
  - **Learn:** typed inference APIs, artifact loading. **Deliverable:** `src/oncora/serve` + test.

- [ ] **5.2 Streamlit UI [MVP]** *(FR-17, FR-18)*
  - **What:** SMILES input → molecule render + three ranked tables + evidence summary + disclaimer
    on every view; calls FastAPI.
  - **Why:** The demo surface judges interact with.
  - **Prereqs:** 5.1. **Verify:** end-to-end on the unseen-compound demo (AC-6); disclaimer visible.
  - **Learn:** rapid data-app UIs. **Deliverable:** `app/` Streamlit app.

- [ ] **5.3 Batch input [STRETCH]** *(FR-20)* — accept multiple SMILES; same outputs per compound.

---

## Phase 6 — Evaluation hardening & submission

- [ ] **6.1 Unseen-compound demo script [MVP]** *(AC-6, AC-7)*
  - **What:** Scripted prediction for a SMILES held out of all training; inspect full output + evidence.
  - **Why:** Direct evidence for "validation on unseen compounds" criterion.
  - **Prereqs:** 5.1, 4.2. **Verify:** runs clean; output reviewed. **Learn:** end-to-end eval. **Deliverable:** demo script + recorded output.

- [ ] **6.2 Reproducibility pass [MVP]** *(NFR-2, AC-8)*
  - **What:** Documented command sequence rebuilds derived tables and reloads a model to reproduce
    a reported metric within seeded tolerance.
  - **Why:** "Reproducibility" judging criterion.
  - **Prereqs:** Phases 1–5. **Verify:** fresh run reproduces a metric. **Learn:** deterministic pipelines. **Deliverable:** `README` run guide.

- [ ] **6.3 Submission writeup & judging-criteria map [MVP]**
  - **What:** Short report mapping each judging criterion to concrete evidence (see §Alignment below).
  - **Prereqs:** 6.1, 6.2. **Verify:** every criterion cites an artifact. **Deliverable:** `reports/submission.md`.

---

## Phase 7 — Future (post-hackathon, Challenge 7 readiness) [FUTURE]

- [ ] GNN/ChemBERTa benchmarks (GPU); React frontend; generative candidate design & ranking
  (structure-in/ranked-out interface already fits); richer L1000 Level 4 evidence if signature
  matrices become available; model registry.

---

## Hackathon evaluation alignment (criterion → evidence)

| Criterion | What the finished project provides |
|---|---|
| **Scientific soundness** | Standardized ChEMBL potency (pChEMBL), documented label/negative policy, trivial-baseline comparisons, metrics matched to imbalance/skew (1.2, 3.1–3.3). |
| **Validation on unseen compounds** | InChIKey-disjoint **and** scaffold-disjoint splits as the headline metric; automated anti-leakage tests; unseen-compound demo (3.1, 3.3, 6.1; AC-2/AC-6). |
| **Biological insight** | Evidence bundle linking predicted targets ↔ sensitive lineages ↔ `compoundinfo` target/moa ↔ CCLE expression, with model-vs-annotation separated (4.1–4.3; FR-13..16). |
| **Originality** | Dual-model pipeline unified in a grounded interpretation layer with structure-only inference enabling true unseen-compound use; Challenge-7-ready interface. |
| **Reproducibility** | Pinned env, seeds, provenance-tagged tables, MLflow runs, Postgres-free Parquet pipeline, offline template summary, documented rebuild (0.1, 1.1, 6.2; NFR-2/NFR-9; AC-8). |
