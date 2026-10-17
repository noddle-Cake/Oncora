# Oncora: architecture handoff (why, alternatives, nuances)

Snapshot: 2026-10-17. Specs 01–04 merged; spec 05 (target model) trained and accepted on branch
`feat/05-target-model`; spec 06 (sensitivity) and 07 (interpretation) code written but not yet
trained or evaluated; specs 08 (API/UI) and 09 (submission) not started.

---

## 0. Instructions for the tutor (paste this whole file into ChatGPT)

> You are tutoring me on a project I built, so I can present it to hackathon judges with
> confidence. Everything below is the ground truth about the project; do not invent details
> beyond it, and tell me when a question goes past what this document says.
>
> Teach in this order: (1) the one-minute story, (2) the end-to-end architecture, (3) each
> decision in section 4, and why the alternatives lost, (4) the results and how to read them,
> (5) the limitations. After each section, quiz me with 2–3 questions a skeptical
> machine-learning or medicinal-chemistry judge would ask, wait for my answer, then grade it
> against this document. Explain any ML or chemistry term I don't know with a concrete example
> from this project. Finish with a mock Q&A using section 7.

---

## 1. The one-minute story

**Oncora** (eMerge Hackathon, Challenge 6: Compound → Target → Cancer). A user pastes a molecule
as a **SMILES** string. Oncora returns:

1. ranked **protein targets** the compound probably binds;
2. ranked **cancer cell lines** likely to be sensitive to it;
3. ranked **cancer types**, made by rolling cell lines up to diseases;
4. a **plain-language explanation** in which every claim cites the evidence it came from.

The judged capability is that it works for compounds **never seen in training**. Nearly every
architectural choice follows from that one requirement:

- features come **from structure only**, since a new molecule has no lab measurements;
- every split is **compound-disjoint**, so the reported numbers really measure new compounds;
- every model must **beat a trivial baseline**, judged with confidence intervals;
- the LLM **only phrases evidence** and never predicts anything.

Required disclaimer, shown everywhere: *"These are computational research predictions, not
validated clinical conclusions."*

**Positioning line:** "We didn't chase state of the art. We built honest baselines whose numbers
you can trust on new compounds, and an explanation layer that cannot make things up."

---

## 2. End-to-end architecture

```
RAW DATA (read-only, checksummed)
  ChEMBL 37 Postgres dump ─┐   DSS3 sensitivity (PharmacoDB) ─┐   CCLE expression ─┐   LINCS annotations
                           ▼                                  ▼                    ▼
  one-time pg_restore → Parquet      identity bridges (InChIKey, ACH- IDs)   interim Parquet
                           │                                  │                    │
                           ▼                                  ▼                    ▼
  ChEMBL quality filter + pChEMBL   ──▶  MODELING TABLES  ◀── sensitivity table (DSS3 × cell line)
  target_pairs (+ constructed negatives)                       │
                           │                                  │
                           ▼                                  ▼
  FEATURES: RDKit standardize → ECFP4 2048 bits + 10 descriptors;  cell lines → 978 landmark genes
                           │
                           ▼
  EVALUATION FRAMEWORK (built BEFORE any model): group splits, baselines, metrics, bootstrap CIs
                           │
           ┌───────────────┴────────────────┐
           ▼                                ▼
  MODEL A: Compound → Target          MODEL B: (Compound, Cell line) → Sensitivity
  1,360 per-target LightGBM           one pairwise LightGBM regressor + classifier
  + per-target calibration            → aggregate cell lines to cancer types
           └───────────────┬────────────────┘     (trained and validated INDEPENDENTLY)
                           ▼
  EVIDENCE BUNDLE (deterministic JSON: predictions | database annotations | derived cross-evidence)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     template renderer           Claude renderer (optional) → claim validator → fallback to template
                           │
                           ▼
  FastAPI /predict  ──▶  Streamlit UI   (disclaimer on every surface)
```

**Stack:** Python 3.11, `uv`, DuckDB + Polars, RDKit, LightGBM + scikit-learn, MLflow (local),
FastAPI, Streamlit, Claude behind an interface. CPU-only on an M1 Max with 32 GB of RAM.

**Code layout** (`src/oncora/`): `io/` (loaders, ChEMBL export, profiler) → `data_prep/` (bridges,
filters, modeling tables, provenance) → `chem/` (validate, standardize, featurize, similarity) →
`features/` → `evaluation/` (splits, baselines, metrics, protocol, report) → `models/target/`,
`models/sensitivity/` → `interpret/` → `serve/` (not yet built). All seeds, thresholds and paths
live in `config.py`.

**Process:** the work was spec-driven (Kiro): 6 steering docs (product, tech, structure, data,
rigor, LLM grounding) and 9 feature specs, each with requirements, design and tasks. Each stage
is a runnable module. Each report in `reports/` is regenerated from code, and a rebuild should
reproduce it byte for byte.

---

## 3. The data and its traps

These data quirks drive many of the decisions, so know them well.

| Source | What it gives | Size |
|---|---|---|
| **ChEMBL 37** | Measured compound–protein potencies: labels for Model A | 24.5M activity rows, 2.9M molecules |
| **DSS3 (PharmacoDB)** | Drug sensitivity score per (compound, cell line): labels for Model B | 512,950 rows; 972 compounds × 1,204 lines |
| **CCLE expression** | Gene expression (log1p TPM) per cell line: cell-line features | 1,775 rows × 19,221 genes |
| **LINCS annotations** | Compound names, SMILES, known targets and MoA: evidence only | 39k compounds |
| **L1000 Level 4 signatures** | Gene-expression response to each compound | ~80 GB; evidence only, never a feature |

Traps found during EDA:

1. **DSS3 mixes scales.** All 9,711 rows with DSS3 ≥ 1 (up to 96) come from 22 `pert_id`s, mostly
   reference drugs from the original CCLE screen. The other 950 compounds lie in [0, 0.995].
   PharmacoDB apparently merged studies on 0–1 and 0–100 scales, and both appear inside a single
   compound (topotecan is about half each), so a row's scale cannot be recovered. Those 22
   `pert_id`s are 20 molecules (lapatinib and saracatinib each have two IDs).
2. **Compound identity explains ~67–73% of DSS3 variance; cell-line identity explains ~4–5%.**
   This is the most important number in the project. Most of the signal is "how potent is this
   compound overall". The part that varies by cell line (selectivity) is small and hard to
   predict.
3. **SMILES lose stereochemistry that the InChIKeys carry** (93 of the 972 DSS3 compounds).
   Stereoisomers like dexamethasone/betamethasone or doxorubicin/epirubicin get identical
   fingerprints.
4. **Salt forms** are registered as separate compounds: 3,945 standardized parents span more than
   one source identity.
5. **CCLE has duplicate profiles** (1,775 rows, 1,719 models). Filter to the default entry.
6. **No workspace file maps cell lines to diseases** (0 of 985). DepMap `Model.csv` was chosen as
   the fix (D-07), and it has not been added yet, which blocks cancer-type ranking.
7. Formatting quirks: `""` and `restricted` stand for null, `-666` placeholders appear, there is
   an Excel `#ERROR`, and one file is Latin-1 rather than UTF-8. The profiler found each of these,
   and each loader handles it.

---

## 4. Decisions: what we chose, why, and what we didn't

Each entry: **Choice**, then **Why**, then **Alternatives and why they lost**, then **Nuance**
(what a sharp judge will ask about).

### 4.1 Platform and data engineering

**One-time ChEMBL export to Parquet; no Postgres at runtime.**
- *Why:* ChEMBL ships as a 2.16 GB Postgres dump. It is restored once (8 tables, ~3 min),
  exported to Parquet via DuckDB's `postgres` extension (streaming, no pandas round-trip), and
  then the database is dropped. Every later stage runs with no database server.
- *Alternatives:* keep Postgres in the loop (heavier setup, worse reproducibility for judges), or
  call the ChEMBL web API (rate-limited, not version-pinned, needs network access).
- *Nuance:* the export is a documented one-time prerequisite. The reproduce script checks the
  Parquet row counts instead of re-running the restore.

**DuckDB + Polars, not pandas.**
- *Why:* DuckDB handles larger-than-RAM SQL joins, such as the 655 MB `instinfo.txt`. Polars
  parses the 19k-column CCLE CSV in ~3 s, where DuckDB takes over 60 s. pandas is used only for
  small final frames.
- *Alternative:* pandas everywhere is simpler but runs out of memory on the big files.

**CPU-only: LightGBM + RDKit; no PyTorch on the critical path.**
- *Why:* the hardware is a single laptop. Gradient-boosted trees are strong on fingerprint and
  tabular features, train fast, and give feature attributions (SHAP) for free.
- *Alternatives:* graph neural networks, ChemBERTa, multitask deep networks. These are possibly
  stronger, but GPU-hungry, slower to iterate on, and harder to explain. They are listed as
  STRETCH/FUTURE benchmarks.

**Provenance and fail-loudly.**
- Each derived Parquet stores its source files (with SHA256), filters, config and code version
  in the file footer.
- Each stage asserts it is non-empty, has the expected columns, and stays within null-rate
  bounds. A violation raises an error naming the file and filter. A stage never writes an empty
  or silently truncated table.

**Single source of truth:** `config.py` holds every seed, threshold and path. Seed = 42.

### 4.2 Compound identity: what counts as "the same compound"

**Choice (D-01): group by the connectivity block (first 14 characters of the InChIKey) of the
RDKit-standardized parent molecule.**
- *Background:* an InChIKey is a hash of the structure. Block 1 encodes the skeleton
  (connectivity), and block 2 encodes stereochemistry and other layers. Standardization
  (Cleanup → LargestFragmentChooser → Uncharger) strips salts and neutralizes charges, so
  "imatinib mesylate" becomes "imatinib".
- *Why:* ECFP fingerprints are achiral by default, so stereoisomers and salt forms get identical
  feature vectors. If one sits in train and another in test, the model has effectively seen the
  test compound. That is leakage.
- *Alternatives:*
  - compound name: unreliable across sources;
  - full InChIKey: leaks stereoisomers (a dedicated test proves it);
  - source connectivity block: leaks salt forms (3,945 cases);
  - Tanimoto clustering (e.g. Butina): a valid, stricter option, but expensive at 1M compounds,
    and its threshold is arbitrary. The scaffold split covers the "harder" case instead.
- *Nuance:* this decision was **refined mid-project**. It started as the source connectivity
  block in spec 01, and featurization in spec 03 revealed the salt-form problem. A good story:
  the data corrected the plan, and the rule is enforced in code rather than by convention.

**Cross-source joins use InChIKey (structure), never names.** compoundinfo ↔ ChEMBL ↔ user input
all go through the same RDKit standardization.

### 4.3 Target labels (Model A's ground truth)

**Quality filter on ChEMBL:** binding and functional assays only; target confidence ≥ 8 (a
direct or homologous single-protein assignment); single human protein; no data-validity flag;
not a duplicate. Funnel: 24.5M → 2.78M usable rows → 2.30M (compound, target) pairs, 1.12M
compounds, 4,910 targets.

**Standardize potency as pChEMBL.** pChEMBL = −log10(molar IC50/Ki/Kd/EC50), so 6 means 1 µM and
7 means 100 nM. This puts different assay types on one comparable scale.

**Active ⇔ median pChEMBL ≥ 6 (≤ 1 µM).**
- *Why:* 1 µM is a common, defensible "meaningful binder" cutoff, and it gives a roughly balanced
  measured set (51% of measured pairs are active).
- *Alternatives:*
  - per-family thresholds (IDG-style: stricter for kinases, looser for ion channels): more
    biologically tuned, more complex;
  - regress pChEMBL directly: richer, but most of the product needs "does it hit or not", and
    regression can't use censored inactives cleanly.

**Censoring policy.** ChEMBL records relations like `>10 µM`:
- `=` → use the value;
- `>` / `>=` with an implied pChEMBL below 6 → a valid inactive ("worse than 10 µM" is certainly
  not ≤ 1 µM);
- `>` above the threshold, or any `<` → dropped as ambiguous.

This recovers 468k censored inactives.

**Dedup:** the median pChEMBL per (compound, target). The median is robust to one outlier assay
(288k pairs were measured more than once).

**Constructed negatives (DR-6), the key modeling call for Model A.**
- *Problem:* ChEMBL records what chemists *tested*, and they test promising compounds, so actives
  dominate. "Never measured" does not mean "inactive".
- *Choice:* three tiers per target:
  - positives;
  - measured negatives;
  - **sampled negatives** topped up to 3× the positives. These are compounds that are active on
    *some other* target but were never measured on this one.
- *Why "active elsewhere":* those compounds are real, drug-like, assay-friendly molecules. Random
  decoys would be trivially different (wrong size, weird chemistry), and the model would learn
  "looks like a drug vs. doesn't" instead of target specificity.
- *Alternatives:*
  - only measured negatives: many targets have almost none, so the classifier would be biased;
  - property-matched decoys (DUD-E style): better controlled, more work;
  - **positive-unlabeled (PU) learning:** the principled framework for "unlabeled ≠ negative",
    but more complex, and its assumptions are harder to explain;
  - a single multi-label model treating every missing cell as negative: the worst label noise.
- *Nuance and honesty:* some sampled negatives are untested true actives, so the reported
  precision is a **lower bound**. This assumption is recorded in provenance and in the model
  card. The median share of negatives that are sampled is 72% per target, so this is a real
  caveat, not a footnote.
- Collision checks use the connectivity block, so a stereoisomer of a measured compound is never
  sampled as a negative. Sampling is seeded per target (CRC32 of the target ID XOR the seed),
  which makes it deterministic.

**Minimum 30 positives per target** → **1,360 targets** kept (of 4,910). There is too little
signal
to learn or evaluate below that. Final table: 4.32M rows (965k positives, 1.25M measured
negatives, 2.10M sampled).

**Collapse before training:** rows sharing a standardized parent have identical features, so
they are merged per (group, target). Measured beats sampled, and the label is recomputed from
the median. 8,707 groups had conflicting labels (stereoisomers or salts measured separately).
4,235,109 rows remain.

### 4.4 Features

**Compound: ECFP4 (Morgan radius 2), 2,048 bits, plus 10 RDKit descriptors** (MolWt, LogP, TPSA,
H-bond donors/acceptors, rotatable bonds, rings, aromatic rings, Fsp3, heavy atoms).
- *Why:* ECFP is the standard, well-understood substructure fingerprint. Each bit roughly means
  "this local atom environment is present", which makes attribution meaningful. The descriptors
  add global physicochemistry that bits miss.
- *Alternatives:*
  - ECFP6 or count fingerprints;
  - FCFP (pharmacophoric);
  - MACCS keys (166 bits, coarse);
  - Mordred (1,800 descriptors, noisy, slow);
  - learned embeddings (GNN, ChemBERTa: GPU, less interpretable).
  These are STRETCH comparisons.
- *Engineering nuance:*
  - uses the current `rdFingerprintGenerator` API, not the deprecated one;
  - fingerprints are bit-packed (256 bytes per compound), so 1.1M compounds take ~0.3 GB in
    memory (`FeatureStore`);
  - 1,099,904 compounds featurized with 0 failures.
- **Murcko scaffold** is computed for *splitting only*, never as a feature.

**Cell line: the 978 LINCS landmark genes, z-scored with training-line statistics (D-06).**
- *Why:*
  - the list is **fixed in advance**, so no gene-selection step can peek at test lines;
  - the genes are biologically curated (chosen to be informative about the whole transcriptome);
  - all 978 are present in CCLE;
  - they keep the design matrix compact (~3k columns), which matters because cell-line identity
    only explains ~4–5% of the variance anyway.
- *Alternatives:*
  - all 19k genes: huge and noisy;
  - top-2,000 by variance or 50–100 PCs: viable, but must be fit inside each fold to avoid
    leakage (STRETCH comparisons);
  - mutations, copy number or pathway scores: not in the provided data.
- *Nuance:* only 106 landmarks are among the top-1,000 most variable genes. Landmarks were picked
  for being informative, not for variance.

**Structure-only rule (FR-10, D-09).** No L1000 perturbation signature can be a model feature.
`assert_structure_only` raises if any feature name falls outside the compound or expression
configs.
- *Why:* a new SMILES pasted by a user has never been profiled, so no signature exists at
  inference time.
- *The trap avoided:* signatures would probably *boost* the offline metrics (they encode the
  compound's biology), but the model could then never serve a new compound. Signatures are
  allowed only as labeled *evidence* in the explanation layer.

### 4.5 Evaluation framework: built *before* any model

**Why first:** the naive trap is a random row split. Each compound appears in hundreds of
sensitivity rows (or several target pairs), so it lands on both sides and the model scores well
by memorizing it. Building and testing the splitters first means no model can ever be judged on
a leaky split.

**Splits:**
- **compound-disjoint** (group = standardized connectivity block): the headline for both models;
- **scaffold-disjoint** (group = stereo-stripped Bemis–Murcko scaffold; acyclic compounds fall
  back to their own block): harder, because it tests new chemotypes, not just new analogs.
  Reported for the sensitivity model.
- Group assignment: largest group first into the currently smallest fold, after a seeded
  shuffle, so folds stay balanced even with huge scaffolds.
- `assert_disjoint` runs **inside every training entry point**, not just in tests, and raises a
  `LeakageError`.
- The test suite includes a **negative control**: a plain random row split must be detected as
  leaky, and a full-InChIKey split must be shown to leak stereoisomers.

**Protocol:** one seeded holdout. 20% of groups are test; 10% of the remaining groups are
validation, used only for early stopping and calibration and never touched by test.
- *Alternative:* k-fold CV.
- *Why holdout:* compute. That would mean 1,360 classifiers × 5 folds, and a ~470k × 3k
  sensitivity matrix per fold.
- Uncertainty comes from **bootstrap confidence intervals that resample compounds, not rows**.
  Rows within a compound are correlated; resampling rows would give falsely narrow CIs. There are
  50 resamples for target metrics (each one recomputes 1,360 per-target AUCs) and 200 for
  sensitivity.

**Acceptance rule:** a model "beats" a baseline only if its **95% CI lower bound exceeds the
baseline's point estimate**. This is stricter than comparing point estimates. Baselines are fit
on train ∪ val, which is slightly generous to them.

**Trivial baselines (must be beaten):**
- *target:* the per-target **prior** (class rate; per-target ROC-AUC = 0.5, PR-AUC = prevalence)
  and **popularity** (rank targets by training positive count) for top-k;
- *sensitivity:* the **global mean**, and the **cell-line mean** (each line's average response).
  The cell-line mean carries zero compound information, so beating it on ranking compounds proves
  the model learned from structure. Its per-compound ranking ("generally sensitive lines first")
  is the real selectivity bar.

**Metrics chosen for imbalance and skew:**
- *target:*
  - **PR-AUC** is primary: precision among top predictions, which is what a chemist acts on;
  - ROC-AUC is reported but flatters when negatives dominate;
  - **top-k**: the share of test compounds whose top-k predicted targets (out of all 1,360)
    include at least one known active.
- *sensitivity:* **Spearman ρ**, a rank correlation. Ranking lines is the product goal, and ρ
  ignores the skew. It is reported:
  - overall;
  - **per cell line** (ranking compounds within a line, i.e. potency);
  - **per compound** (ranking lines within a compound, i.e. selectivity);
  - plus RMSE on the transformed target and thresholded AUCs.

### 4.6 Model A: Compound → Target

**Choice: one LightGBM binary classifier per target (1,360 models), plus a per-target calibrator.**
- *Why per-target:* each target keeps its *own* negatives (from 4.3) and its own calibration;
  each model is inspectable on its own; and training parallelizes trivially across processes.
- *Alternatives:*
  - one multi-label model: forces a shared negative definition, so every unmeasured cell becomes
    negative;
  - a multitask neural net: can share information across related targets, which is the main
    thing per-target models give up, but needs a GPU;
  - nearest-neighbor similarity search (e.g. SEA): a strong, simple comparator worth mentioning.
- **Global split:** the compound split is computed once and applied to every target, so a test
  compound is unseen by *all* 1,360 models. Without this, top-k per compound would be dishonest:
  a compound could be in test for EGFR but in train for HER2.
- **Hyperparameters:** 31 leaves, learning rate 0.05, up to 500 trees with early stopping (30
  rounds) on validation, 50% feature subsampling, 80% row subsampling. No `scale_pos_weight`:
  calibration handles the probability scale, and PR-AUC handles imbalance.
- **Comparison model:** L2 logistic regression on the ECFP bits (a linear model with one weight
  per substructure).

**Calibration rule (fixed in advance, by validation size per class):**

| Validation compounds per class | Method | Why |
|---|---|---|
| < 10 | identity (raw scores) | too little data |
| 10–99 | **regularized Platt** | logistic recalibration with an L2 pull toward the identity mapping |
| ≥ 100 | isotonic | flexible, but needs data |

- *Why calibrate at all:* raw scores aren't comparable across targets (one target's 0.6 can be
  another's 0.2). To rank 1,360 targets for one compound, the scores must mean "probability".
- *Story worth telling:* in the first full run, an unregularized fit on ~4 validation positives
  turned a raw 0.42 into 1.000. That is why the identity floor and the regularized Platt exist.
- Isotonic maps plateau, so ties are broken by the raw score. Per-target AUCs use raw scores;
  pooled metrics and top-k use calibrated ones.

**Shipped artifact = the evaluated model** (fit on train, with validation used for early stopping
and calibration). It is not refit on all data, so test compounds stay truly unseen by the served
model, and the reported metrics describe exactly what is deployed. The trade-off: it was trained
on ~70% of compounds instead of 100%.

**Attribution (STRETCH, written):** LightGBM `pred_contrib` gives SHAP values per feature, and
positive fingerprint bits map back to atom environments in the query molecule via RDKit's
bit-info map, so substructures can be highlighted. These explain the *model*, not the mechanism.

### 4.7 Model B: (Compound, Cell line) → Sensitivity

**Choice: one pairwise model.** Each row is `[compound features ‖ cell-line features]` → DSS3.
- *Why:* both halves can be computed for *any* new SMILES and *any* CCLE line, so it handles
  unseen compounds, and it can even score the 734 CCLE lines that have expression data but no
  DSS3 labels.
- *Alternatives:*
  - one model per cell line (QSAR per line): ~1k models that can't share information across
    lines;
  - matrix factorization / collaborative filtering: can't score a brand-new compound (the
    cold-start problem) without side features, and adding them turns it back into this design;
  - deep drug-response models (e.g. DeepCDR-style): GPU, FUTURE.
- *Engineering:* the ~470k × ~3k float32 matrix (~5.7 GB) is built by fancy-indexing two small
  per-entity arrays (~940 compounds, ~1k lines) instead of materializing a joined DataFrame.

**Handling the mixed scales (D-03):** the 20 mixed-scale compounds are excluded from headline
training and evaluation. They are listed in the model card, and one **sensitivity analysis**
re-runs with values ≥ 1 multiplied by 0.01.
- *Alternatives:*
  - keep the raw values: corrupts labels, and a value of 96 would dominate the loss;
  - rescale everything ≥ 1: wrong for rows that were genuinely on the 0–1 scale;
  - drop only rows ≥ 1: leaves that compound's 0–100-scale rows that happen to fall below 1,
    which are silently mislabeled.
- Headline set: **467,725 rows · 940 compounds (938 connectivity blocks) · 977 cell lines**.

**Regression target = rank-transformed DSS3 (D-04).**
- Each value becomes its percentile under the *training* distribution; ties get midpoint ranks,
  so the 21% exact zeros share one value.
- *Why:*
  - robust to the zero spike and to any leftover scale weirdness;
  - Spearman ρ is unchanged by any monotone transform;
  - fitting the transform on training data only means no test information leaks.
- *Alternatives:*
  - raw DSS3: skewed;
  - log1p: barely helps, because log1p is nearly linear on [0, 1] (skew drops only 1.87 → 1.48).

**"Sensitive" label = within-compound top 10% of lines AND DSS3 > 0 (D-05).**
- *Why:* it measures **selectivity** ("which lines respond most to *this* compound"), which is the
  product question.
- *The alternative:* a global threshold (DSS3 ≥ p90 = 0.359), reported for comparison, mostly
  just labels *potent compounds*.
- The "> 0" guard stops a mostly-zero compound from labeling every tied line as sensitive.
- A separate LightGBM classifier is trained for this label, and its probability becomes
  `p_sensitive`.

**Potency vs. selectivity (D-08): the most important evaluation nuance.** Compound identity
explains ~70% of the variance, so:
- **per-cell-line ρ** (rank compounds within a line) mostly measures *potency*, learned from
  structure. The cell-line-mean baseline is undefined here, by construction.
- **per-compound ρ** (rank lines within a compound) measures *selectivity*, which is small and
  hard. The cell-line-mean baseline is the real competitor: per-compound ρ ≈ 0.275 on the compound
  split, 0.287 on the scaffold split.
- **Compound-only ablation:** the same learner without cell-line features. It can only predict
  potency.
- **Explicit selectivity verdict:** if the full model's per-compound ρ lower CI doesn't beat the
  cell-line mean, the UI and writeup must say that the cell-line rankings mostly reflect
  "compound potency + each line's general sensitivity", not learned selectivity.
- This is planned honesty: the verdict is computed automatically and stored in the model card
  (`learned_selectivity`).

**Cancer-type aggregation:**
- each disease scores the **mean predicted percentile of its top 3 lines**, plus the fraction of
  its lines predicted sensitive;
- diseases with fewer than 3 lines are **flagged low-support, not dropped**;
- lines without a disease are counted, never guessed.
- *Why top-k mean and not the mean of all lines:* a disease with one or two very responsive lines
  is worth investigating, and averaging all of its lines would hide them.
- **Blocked** until DepMap `Model.csv` is added. Until then the API returns cell-line rankings
  with an empty cancer-type list and a note.

### 4.8 Near-seen flag

If a query's max **Tanimoto similarity** to any training compound is ≥ 0.99, it is flagged:
"this partly reflects memory, not generalization". Tanimoto = shared on-bits / union of on-bits.
It is computed with NumPy popcount over the packed uint64 fingerprint words, at well under a
second per query against 1M compounds, with no RDKit objects. Each model has its own training
index.

### 4.9 Interpretation layer: LLM as renderer, not predictor (D5)

**Evidence bundle:** one deterministic JSON object per query, in three explicitly separated
kinds:
- `prediction` (model output);
- `annotation` (what databases already say: LINCS MoA and targets, ChEMBL measured actives);
- `derived` (cross-evidence such as "predicted target also annotated", or target-gene expression
  in the top predicted lines).

More details:
- annotations are matched by exact InChIKey, then by connectivity block, and the **match level is
  recorded**;
- floats are rounded to 3 significant figures, so the JSON is byte-stable and every number is
  checkable;
- every item is path-addressable (`predictions.targets[0].score`).

**Two renderers:**
- **Template (always available):** f-string sentences, each citing its bundle paths. It works
  offline with no API key, which proves the explanation doesn't *need* an LLM.
- **Claude (optional):**
  - input: only the bundle's canonical JSON (never dataset rows);
  - output: constrained by a JSON schema to a list of `{section, text, cites[]}` claims;
  - the system prompt forbids outside biology and causal claims between the two models.

**Validator (`validate_claims`):** every cited path must exist, and **every number in the text
must match a bundle value at 3 significant figures**. Failing claims are dropped. If none
survive, or on any API error, timeout, refusal or truncation, the system falls back to the
template. Users never see that as an error.

**Cost and reproducibility:**
- one bounded call per explanation (low effort, capped tokens, 30 s timeout);
- usage is logged to `reports/llm_usage.jsonl` against a ~$100 budget;
- metrics and rankings **never** depend on the LLM (NFR-9).

**Alternatives:**
- no LLM: safe but stiff prose;
- LLM as predictor ("ask Claude what EGFR inhibitors do"): ungrounded and not reproducible, and
  it can't be validated on unseen compounds;
- free-text RAG over papers: richer, but hallucinations become hard to catch.

**Correlational framing (D4/FR-15).** The two models are trained independently, so "predicted to
hit EGFR" plus "predicted sensitive in lung lines" is *co-occurrence, not proof*. The bundle
states this in its `relationship` field.

### 4.10 Serving, UI, reproducibility (planned: specs 08–09)

- **FastAPI** `/predict`: loads artifacts once at startup and fails fast if one is missing
  (better than serving half a system). An invalid SMILES returns a 422 with the RDKit reason and
  the disclaimer.
- **Streamlit** UI: the disclaimer banner comes first; predictions and annotations are styled
  differently. React was rejected as slower to build for a Python-centric team (FUTURE).
- **Unseen demo:** gefitinib (an EGFR inhibitor) is **removed from both modeling tables before
  training**: 540 ChEMBL pairs and 929 DSS3 rows are set aside. It is truly unseen, and its real
  labels can be compared afterwards. The demo script asserts it is absent from both training
  indexes.
- **Reproducibility:** `uv` lockfile, seeds, checksums on the raw data, provenance footers,
  MLflow runs, a `model_card.json` per artifact, and `reproduce.sh` plus a metric-drift check.

---

## 5. Results so far (and how to read them)

### Model A: Compound → Target (compound-disjoint test, 847k rows, 1,360 targets): **Accepted**

| Metric | LightGBM | LogReg-ECFP | Prior/popularity |
|---|---|---|---|
| PR-AUC, macro | **0.856** [0.854, 0.868] | 0.852 [0.849, 0.865] | 0.225 |
| PR-AUC, weighted | **0.913** [0.911, 0.916] | 0.896 [0.894, 0.899] | 0.233 |
| ROC-AUC, macro | 0.935 | 0.934 | 0.500 |
| Top-1 / 5 / 10 | 0.184 / 0.623 / 0.832 | — | 0.013 / 0.052 / 0.102 |
| Brier (raw → calibrated) | 0.0416 → 0.0427 | | |

How to talk about it:
- **~3.8× the prior on PR-AUC**, on compounds no model has seen. For 83% of new compounds a known
  active target is in the top 10 of 1,360.
- **LightGBM barely beats logistic regression on macro PR-AUC** (CIs overlap), but beats it
  clearly on the weighted metric, which is dominated by large targets. The honest reading: most
  of the signal is linearly accessible from substructures, and trees add the most where data is
  plentiful.
- **Calibration did not improve the pooled Brier score** (slightly worse). LightGBM trained on log
  loss is already roughly calibrated on average. Calibration's job here is making scores
  *comparable across targets* for ranking, and validation sets are small for many targets. Say
  this before a judge does.
- **Precision is a lower bound**, because some sampled negatives are untested actives.
- **Top-1 = 18% looks low** but is out of 1,360 candidates against incomplete labels. A "wrong"
  top-1 may simply be unmeasured.
- Caveat on the CIs: in a resample, small targets can become single-class and drop out of the
  macro average, which nudges the bootstrap slightly upward. Treat the point estimate as the
  reference.

### Model B: Sensitivity: **not trained yet.** Baselines to beat:

| Metric | Cell-line mean (compound split) | Cell-line mean (scaffold split) |
|---|---|---|
| Spearman per compound (selectivity bar) | 0.275 | 0.287 |
| Spearman overall | 0.206 | 0.161 |
| PR-AUC within-top-10% (prevalence 0.10) | 0.192 | 0.161 |
| Spearman per cell line | undefined (no compound info) | undefined |

The acceptance gate is per-cell-line ρ (did it learn potency from structure?). The interesting
and possibly negative result is per-compound ρ vs. 0.275 (did it learn selectivity?). Be ready to
present either outcome as a finding.

### Engineering health

126 tests pass. They include the anti-leakage suite (property-style disjointness over random
seeds and fold counts, the random-split negative control, the stereoisomer-leak demonstration,
same-seed determinism, scaffold disjointness on both keys).

---

## 6. Limitations to state up front

1. **Sampled negatives** (72% of negatives, median per target) add label noise, so precision is
   understated. PU learning would be the principled next step.
2. **Sensitivity data is small on the compound axis:** only ~940 compounds. Every structure → potency
   generalization is learned from those, against a ~1M-compound target dataset.
3. **Selectivity is inherently weak in this data** (cell-line identity is ~4–5% of variance), so
   cell-line rankings may mostly reflect potency. The system measures and reports this rather
   than hiding it.
4. **Cancer types are blocked** on `Model.csv`, and in-vitro cell-line sensitivity ≠ clinical
   efficacy for a cancer type.
5. **Holdout, not k-fold CV:** a single split's partition may be lucky or unlucky. The bootstrap
   CIs cover test-set sampling, not split-to-split variance.
6. **The served model saw ~70% of the compounds** (no final refit). The choice favors honesty over
   coverage.
7. **ECFP is achiral**, so the models cannot distinguish stereoisomers. This is acknowledged, and
   splits are built around it.
8. **Two models, no joint inference.** Target ↔ sensitivity links are correlational by design.
9. Applicability domain: compounds far from both training sets get predictions, but only
   near-identity (≥ 0.99) is flagged. There is no low-similarity warning yet.

---

## 7. Likely judge questions and crisp answers

- **"How do I know you're not memorizing?"** Every split is compound-disjoint by the standardized
  connectivity block, so salts and stereoisomers stay together. Disjointness is asserted inside
  training, not just in tests. A random-row control is proven leaky in the test suite. The
  sensitivity model is also tested scaffold-disjoint. And gefitinib was removed before training
  as a live demo.
- **"Why not a graph neural network?"** CPU-only, hackathon timeline, interpretability. The bar
  was honest validation. A GNN is a STRETCH benchmark, and the evaluation framework would judge it
  identically.
- **"Your negatives are made up."** Partly, and that is documented. They are compounds active
  elsewhere but never tested on this target, which are realistic molecules, not trivial decoys.
  This makes the reported precision a lower bound. Measured negatives (including censored `>`
  values) come first.
- **"Why PR-AUC?"** Negatives outnumber positives about 3:1, and ROC-AUC rewards ranking the easy
  negatives. PR-AUC asks "of what I'd tell a chemist to test, how much is right?"
- **"Why 1 µM?"** It is a standard binder cutoff that splits the measured data roughly 50/50.
  Per-family thresholds are a known refinement.
- **"Why rank-transform DSS3?"** 21% zeros, a heavy skew, and mixed scales. Ranking is the
  product goal, and Spearman ρ is the matching metric.
- **"Does the sensitivity model know anything about cell lines?"** That is exactly what the
  per-compound ρ vs. cell-line-mean comparison and the compound-only ablation test, and the
  verdict is written into the model card and UI.
- **"Could the LLM hallucinate biology?"** It only sees one query's bundle. Every claim must cite
  existing paths, and every number must match the bundle. Failures are dropped or replaced by a
  deterministic template. Rankings never depend on it.
- **"Why not use the L1000 signatures? They're right there."** A new SMILES has no signature, so
  using them would make offline numbers look better while making the product unable to serve its
  core use case.
- **"Why calibrate if Brier got slightly worse?"** To compare scores across 1,360 targets for one
  compound. The pooled Brier is dominated by large, already-calibrated targets. The rule is
  conservative (identity or a pull toward identity when validation data is small), because of an
  observed failure (0.42 → 1.000).
- **"Is a single holdout enough?"** It is a compute trade-off at this scale, with uncertainty
  from a compound-level bootstrap and acceptance by CI lower bound rather than point estimate.
- **"What's original here?"** A dual-model pipeline (target + sensitivity) that runs on structure
  alone, joined only in a grounded, verifiable evidence layer. There is also built-in honesty
  machinery: the potency/selectivity split, automatic verdicts, and near-seen flags.

---

## 8. Glossary

- **SMILES:** a text encoding of a molecule (`CC(=O)Oc1ccccc1C(=O)O` = aspirin).
- **InChIKey:** a 27-character hash of the structure. The first 14 characters (connectivity
  block) encode the skeleton; the next block encodes stereochemistry and other layers.
- **Standardized parent:** the molecule after salt stripping, charge neutralization and cleanup.
- **ECFP4 / Morgan fingerprint:** 2,048 bits, each roughly "a circular atom environment of
  radius ≤ 2 is present".
- **Bemis–Murcko scaffold:** the ring systems plus their linkers with side chains removed. It
  defines a "chemotype".
- **Tanimoto:** fingerprint similarity in [0, 1].
- **pChEMBL:** −log10 of molar potency; 6 = 1 µM, 9 = 1 nM.
- **DSS3:** Drug Sensitivity Score (an area-under-the-dose-response-curve-style summary); higher
  means more sensitive.
- **CCLE / DepMap:** the Cancer Cell Line Encyclopedia. `ACH-` IDs are DepMap model IDs.
- **LINCS L1000:** gene-expression responses to perturbations; the 978 "landmark" genes are
  measured directly.
- **Leakage:** test information reaching training, so scores look better than real-world
  performance.
- **Group split:** all rows of a group (a compound or scaffold) go to the same side.
- **PR-AUC / ROC-AUC:** areas under the precision–recall and ROC curves. PR-AUC is
  prevalence-sensitive and better under imbalance.
- **Spearman ρ:** correlation of ranks.
- **Calibration (Platt / isotonic):** maps scores to probabilities. Platt fits a logistic curve;
  isotonic fits a monotone step function.
- **Brier score:** mean squared error of the predicted probabilities.
- **SHAP:** an additive per-feature contribution to one prediction.
- **Bootstrap CI:** resample the test set (here, compounds) many times and take the 2.5–97.5
  percentiles of the metric.
- **PU learning:** positive-unlabeled learning, for when "unlabeled" ≠ "negative".
- **Model card:** a JSON file shipped with each model: data hashes, features, split, label
  policy, metrics, seed, intended use.

---

## 9. Where things live

| Want | Look at |
|---|---|
| Locked product decisions D1–D5 | `.kiro/steering/product.md` |
| Data facts, quirks, D-01…D-09 | `.kiro/steering/data-sources.md`, `reports/eda_summary.md` |
| Rigor rules (leakage, baselines, metrics) | `.kiro/steering/scientific-rigor.md` |
| LLM grounding contract | `.kiro/steering/llm-grounding.md` |
| Per-feature design | `.kiro/specs/0N-*/design.md` |
| All thresholds and seeds | `src/oncora/config.py` |
| Splits / protocol | `src/oncora/evaluation/splits.py`, `protocol.py` |
| Negatives | `src/oncora/data_prep/target_pairs.py` |
| Target model + calibration | `src/oncora/models/target/train.py`, `calibration.py` |
| Sensitivity model | `src/oncora/models/sensitivity/train.py` |
| Evidence bundle + Claude renderer | `src/oncora/interpret/bundle.py`, `render_claude.py` |
| Numbers | `reports/*.md`, `reports/eval/*.md` |
