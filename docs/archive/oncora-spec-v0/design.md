# Oncora — System Design

> Companion to `requirements.md`. This document justifies the architecture, the technology
> choices, and the machine-learning strategy. Every non-obvious choice states **why** — this
> project doubles as an applied-data-science learning artifact, so the reasoning is part of the
> deliverable, not decoration.

---

## 1. Design principles (what drives every decision below)

1. **Scientific honesty over leaderboard scores.** The headline numbers must come from
   leakage-safe splits. A simple model with trustworthy validation beats a fancy model with a
   hidden leak — and it wins the "scientific soundness" judging criterion.
2. **Simplicity that a small team can finish.** Prefer the fewest moving parts that satisfy the
   requirement. Add a technology only when a requirement demands it.
3. **Structure-in, structure-out.** The system's only required input is a SMILES string; every
   required feature must be derivable from structure (+ precomputed cell-line features). This is
   what makes unseen-compound prediction real and leaves the door open for Challenge 7.
4. **Deterministic core, optional intelligence.** Predictions and metrics are deterministic and
   reproducible offline. The LLM is a thin, grounded garnish with a template fallback.
5. **Each stage independently runnable and testable.** Supports incremental learning and makes
   failures easy to localize.

---

## 2. Technology choices — and the ones deliberately NOT used

| Concern | Choice | Why this, and why not the alternatives |
|---|---|---|
| Target platform | **M1 Max, 32 GB RAM, macOS/arm64, CPU-only** | Confirmed by developer. All deps chosen for native arm64 wheels; no CUDA. 32 GB allows in-memory CCLE work while out-of-core discipline is kept for the 655 MB / 512k-row paths. |
| Language | **Python 3.11** | Ecosystem for cheminformatics + ML; matches the learning goal; clean arm64 support. |
| Env/deps | **`uv` + `pyproject.toml`, pinned** | Fast, reproducible, lockfile. (Conda works but is heavier; pip-tools is fine too.) |
| Heavy table I/O & joins | **DuckDB** (+ **Polars** for in-memory transforms) | DuckDB reads CSV/Parquet larger-than-RAM, does SQL joins on the 512k sensitivity rows and the 305 MB / 655 MB files without full-DataFrame loads (NFR-3). Pandas is used only for small final frames. |
| ChEMBL access | **`pg_restore` once → export needed tables to Parquet → query with DuckDB** | Avoids keeping a Postgres server in the loop for every run. One documented restore+export; afterwards the pipeline is Postgres-free and reproducible. (Keeping live Postgres is the alternative; rejected for reproducibility/latency and because only a handful of tables are needed.) |
| Cheminformatics | **RDKit** | Canonical SMILES, validity check, Morgan/ECFP fingerprints, Bemis–Murcko scaffolds, descriptors, molecule rendering. The de-facto standard; no credible lighter alternative. |
| Baseline models | **LightGBM** (primary) + **scikit-learn** (logistic/ridge baselines, pipelines, metrics) | Gradient-boosted trees are strong on tabular fingerprint+descriptor features, train fast on CPU, handle non-linearities, give feature importances for interpretability (FR-8). scikit-learn supplies the trivial baselines (AC-3) and metric/CV plumbing. |
| Deep models | **Deferred (STRETCH/FUTURE)** | GNNs/ChemBERTa need GPU and tuning time; they are a benchmark, not the MVP. D3. |
| Experiment tracking | **MLflow (local file backend)** | Logs params/metrics/artifacts per run with zero server setup; satisfies FR-22. (Weights & Biases rejected: external service, account, overkill here.) |
| Artifact storage | **Local `models/` dir, versioned by run id + a `model_card.json`** | Simple, inspectable; pairs with MLflow. No registry service needed for a hackathon. |
| Inference API | **FastAPI** | Typed request/response, auto OpenAPI docs, trivial to call from Streamlit; matches developer's API experience (FR-19). |
| UI | **Streamlit** | Python-native, renders molecules + tables + text fast, keeps the developer in the data-science loop (D2). React is FUTURE. |
| LLM explanation | **Anthropic Claude via a thin client, behind an interface with a template fallback** | Grounded prose for the evidence bundle (D5); template fallback preserves offline reproducibility (NFR-9). |
| PostgreSQL (as a runtime dependency) | **Not used at runtime** | Only needed once to crack open the ChEMBL dump. The app does not depend on a running DB. |

**Guiding rule applied:** we did *not* reach for every tool in the prompt. No React (time), no
PyTorch on the critical path (compute), no Postgres at runtime (reproducibility), no cloud
services (simplicity).

---

## 3. Repository layout

```
Oncora/
├─ data/
│  ├─ raw/            # symlinks/paths to the provided files (gitignored, never committed)
│  ├─ interim/        # cleaned, standardized per-source Parquet (provenance-tagged)
│  └─ processed/      # final modeling tables (compound-target pairs; sensitivity+expr matrix)
├─ src/oncora/
│  ├─ config.py       # paths, thresholds, seeds, feature config (single source of truth)
│  ├─ io/             # duckdb/polars loaders; chembl parquet export; schema profilers
│  ├─ chem/           # RDKit: validate, canonicalize, fingerprints, scaffolds, descriptors
│  ├─ data_prep/      # identity bridges, filters, joins -> processed tables
│  ├─ features/       # compound features + cell-line expression feature builder
│  ├─ models/
│  │  ├─ target/      # compound->target training + inference
│  │  └─ sensitivity/ # compound x cell-line sensitivity training + inference
│  ├─ evaluation/     # splitters (compound-disjoint, scaffold), metrics, baselines
│  ├─ interpret/      # evidence-bundle builder + explanation generator (template | LLM)
│  └─ serve/          # FastAPI app; artifact loading
├─ app/               # Streamlit UI
├─ notebooks/         # EDA (Phase 0) — numbered, outputs committed as the data-profile report
├─ models/            # versioned artifacts + model_card.json (gitignored except cards)
├─ reports/           # EDA report, evaluation reports, figures
├─ tests/             # pytest: chem utils, splitters (anti-leakage), joins, API contract
├─ mlruns/            # MLflow local tracking (gitignored)
├─ pyproject.toml     # pinned deps (uv)
└─ .kiro/specs/oncora # this spec
```

---

## 4. Data architecture & integration

### 4.1 The two identity bridges (the backbone of the whole system)

```
COMPOUND identity:
  PharmacoDB pert_id (BRD-…) ──compoundinfo.txt──▶ canonical_smiles + inchi_key + target + moa
  ChEMBL molregno ──compound_structures──▶ canonical_smiles ──RDKit──▶ inchi_key
  └── dedup/link across sources by **InChIKey** (structure identity, not name)

CELL-LINE identity:
  CCLE ModelID (ACH-…)  ◀──equal──▶  PharmacoDB cell_line (ACH-…)
  └── annotate via cellinfo.txt (Cellosaurus, primary_disease) + CCLE lineage
```

Verified coverage (from real data): **972/972** sensitivity compounds have SMILES via
`compoundinfo.txt`; **985** cell lines overlap between CCLE and sensitivity.

### 4.2 Pipeline stages (each writes a provenance-tagged Parquet and is independently runnable)

1. **Profile** raw sources → `reports/eda_*` (schemas, nulls, id spaces, overlaps). *(DR-1)*
2. **Bridge** → `interim/compound_bridge.parquet`, `interim/cellline_bridge.parquet`. *(DR-2, DR-3)*
3. **ChEMBL export+filter** → `interim/chembl_activities.parquet` (quality-filtered,
   standardized potency, deduped). *(DR-5)*
4. **Build modeling tables** →
   `processed/target_pairs.parquet` (compound × target + label) *(DR-4, DR-6)* and
   `processed/sensitivity.parquet` (compound × cell-line × DSS3, joined to expression keys). *(DR-4)*
5. **Featurize** → compound fingerprints/descriptors; cell-line expression feature matrix
   (variance-filtered / landmark genes). *(DR-7)*

Error handling (NFR-6): each stage asserts non-empty outputs, expected columns, and null-rate
bounds; it fails with the offending file/filter named.

---

## 5. Machine-learning strategy (the scientific core)

### 5.1 Compound → Target (ChEMBL)

- **Label framing.** Binarize compound–target activity by a documented potency threshold
  (default **pChEMBL ≥ 6**, i.e. ≤ 1 µM) = "active"; below = "inactive". The threshold is
  recorded in metadata (FR-7). *Why pChEMBL:* it standardizes IC50/Ki/Kd/EC50 onto one
  −log10(molar) scale, which is exactly the heterogeneity problem ChEMBL assays present.
- **Censored data (`>`, `<`).** `>` (no activity up to a high dose) supports "inactive"; `<` is
  ambiguous and dropped or handled conservatively — documented in DR-5.
- **Negatives (DR-6) — DECIDED: binary with constructed negatives.** Positives = measured actives
  (pChEMBL ≥ threshold). Negatives = measured inactives (`>`-threshold / below-cutoff) where
  available, plus a documented negative-sampling scheme for targets short on measured inactives
  (compounds active elsewhere but unmeasured on the target), with the sampling ratio and its
  "absence ≠ inactive" assumption recorded in the model card. Avoids fabricating confident
  negatives while giving every target a usable two-class training set.
- **Features.** ECFP4 (Morgan radius 2, 2048 bits) + a compact set of RDKit physchem descriptors.
  *Why fingerprints:* they encode substructure presence, generalize across scaffolds, and are the
  standard strong tabular baseline for bioactivity.
- **Model.** LightGBM classifier (per-target or multi-label), with logistic-regression-on-ECFP as
  the simple comparison baseline.
- **Interpretability (FR-8).** Fingerprint-bit → substructure attribution for top contributions.

### 5.2 Compound → Sensitivity (PharmacoDB DSS3 + CCLE)

- **Target variable.** DSS3 is continuous, heavily right-skewed (median ≈ 0.087, max ≈ 96, ~20%
  near zero). Plan: model **two complementary framings** and report both —
  (a) **regression** on a variance-stabilizing transform (log1p or rank/quantile), reported with
  **Spearman ρ** (rank-focused, robust to the skew) and RMSE on the transform;
  (b) **classification** sensitive-vs-resistant at a documented DSS3 threshold, reported with
  ROC/PR-AUC. *Why both:* ranking cell lines is the real product goal (ρ), while a threshold gives
  an interpretable, imbalance-aware metric (AC-5).
- **Features (the critical constraint, FR-10).**
  compound side = **structure-only** (ECFP4 + descriptors);
  cell-line side = **precomputed CCLE expression features** (variance-filtered genes or a
  landmark/pathway subset, DR-7).
  A row = concat(compound features, cell-line features) → DSS3. **No LINCS signature is ever a
  required feature**, so a brand-new SMILES against any CCLE cell line is predictable at inference
  (A2). This is the design's answer to "must not require a measured signature for new compounds."
- **Model.** LightGBM regressor (and a parallel classifier for the thresholded framing).
  Trivial baselines (AC-3): global mean, and **cell-line-mean** predictor (strong baseline that
  exposes whether the model learns compound effects vs. just cell-line averages).
- **Cancer-type aggregation (FR-11).** Map predicted per-cell-line sensitivity →
  `cellinfo.primary_disease` + CCLE lineage, aggregate (e.g. mean of top-k), rank cancer types.

### 5.3 Validation strategy — the part that makes or breaks credibility

- **Leakage is the central risk** (requirements NFR-1, AC-2). The naive trap: random row split lets
  the *same compound* sit in train and test, so the model memorizes compounds and looks great but
  fails on unseen ones. Prohibited as the headline.
- **Compound-disjoint split (both models):** partition by **InChIKey** so no compound crosses
  train/test. This directly measures unseen-compound generalization — the judged capability.
- **Scaffold split (sensitivity, additionally):** Bemis–Murcko scaffold-disjoint split, a harder
  and more realistic generalization test (new chemotypes).
- **Grouping for sensitivity:** since each compound appears across many cell lines, the split is
  grouped by compound (`GroupKFold`-style) — never by row.
- **Baselines reported alongside (AC-3)** so "beats trivial" is explicit.
- **Metrics chosen for imbalance/skew:** PR-AUC and top-k for targets; Spearman ρ + thresholded
  AUC for sensitivity.
- **Uncertainty/interpretability (FR-8, FR-13):** report score distributions, flag near-seen
  compounds (Tanimoto to training set), and expose feature importances.

**Concepts the developer learns here (called out for the learning goal):** data leakage &
grouped/scaffold splitting, class imbalance vs. metric choice (ROC vs PR), skewed-target
transforms & rank correlation, standardized potency (pChEMBL), censored measurements, trivial
baselines as a sanity floor, feature attribution.

---

## 6. Biological interpretation layer (explainability)

```
predictions ──▶ EvidenceBundle (structured, deterministic)
                 • top targets + scores (model)
                 • top sensitive lineages + scores (model)
                 • compoundinfo.target / moa  (annotation)   ← labeled as annotation, not prediction
                 • optional: target-gene expression in those lineages (CCLE)   [FR-16]
                 │
      ┌──────────┴───────────┐
      ▼                      ▼
 Template renderer     Claude renderer (optional, same bundle, grounded)
 (always available)    (restate-only + cite source field; no new facts)
                 │
                 ▼
        Evidence summary (model-vs-annotation separated, every claim cited, disclaimer attached)
```

- **Grounding contract (FR-13a, FR-14, FR-16a):** the LLM receives **only** the compact bundle,
  is instructed to restate only what's in it and cite each claim's field, and never to add
  biology. The template renderer proves the same output is achievable with zero LLM — so the
  pipeline is fully reproducible and judge-able offline (NFR-9).
- **No cross-proof (FR-15/D4):** copy explicitly frames target+sensitivity co-occurrence as
  correlational evidence, not proof.

---

## 7. Serving architecture

```
Streamlit (app/) ──HTTP──▶ FastAPI (src/oncora/serve)
                              ├─ load target model + sensitivity model + feature config (versioned artifacts)
                              ├─ POST /predict {smiles} -> {valid, canonical_smiles,
                              │      targets[], sensitive_cell_lines[], cancer_types[],
                              │      evidence_summary, near_seen_flag, disclaimer}
                              └─ GET  /health
```

- Models loaded once at startup from `models/<run_id>/` using the `model_card.json` (FR-23).
- UI renders the molecule (RDKit), three ranked tables, the evidence summary, and the disclaimer
  on every view (FR-17, FR-18).
- API returns the disclaimer as a response field (AC-9).

---

## 8. Reproducibility, testing, error handling

- **Reproducibility (NFR-2, FR-21):** `uv` lockfile, fixed seeds in `config.py`, ordered stage
  scripts, provenance tags on every derived table, MLflow run per training. A documented command
  sequence rebuilds tables and reloads a model to reproduce a metric (AC-8).
- **Testing (tests/):** unit tests for chem utils (validity/canonicalization), **anti-leakage
  tests that assert train/test InChIKey disjointness and scaffold disjointness**, join/shape
  assertions, and a FastAPI contract test. (These tests are part of the spec's testing plan; the
  prompt's "don't auto-add tests" rule is about unrequested tests — here they are a stated
  requirement.)
- **Error handling (NFR-6):** loud failures on schema drift, empty joins, unexpected nulls.

---

## 9. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Hidden compound leakage | Overoptimistic, criterion-failing results | InChIKey-disjoint + scaffold splits; automated disjointness tests (AC-2) |
| ChEMBL restore friction / no Postgres | Blocks target model | One-time restore→Parquet, then Postgres-free; DuckDB thereafter |
| CCLE 19,221-dim + skewed DSS3 | Overfitting / misleading RMSE | Gene feature selection (DR-7); Spearman + thresholded AUC; cell-line-mean baseline |
| L1000 signatures absent/needed | Breaks inference for new compounds | Structure-only compound features; L1000 strictly optional evidence (A2, FR-10) |
| Memory on 305 MB / 655 MB files | Crashes on laptop | DuckDB/Polars streaming; never load `instinfo.txt` whole (NFR-3) |
| LLM hallucination in explanations | Scientific credibility loss | Restate-only grounding + template fallback; offline reproducibility (NFR-9) |
| Scope creep (two models + UI + LLM) | Nothing finishes | Phased roadmap; EDA-first; baselines before fancy models |

---

## 10. Traceability (requirement → design section)

- FR-1..4 → §2 RDKit, §4 chem. FR-5..8 → §5.1. FR-9..12 → §5.2. FR-13..16a → §6.
- FR-17..20 → §7. FR-21..23, NFR-1..10 → §8, §5.3, §6. DR-1..7 → §4, §5.
- AC-1..9 → §4 (EDA), §5.3 (splits/metrics/baselines), §6 (interpretation), §7–8 (repro/disclaimer).
