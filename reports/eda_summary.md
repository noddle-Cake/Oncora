# EDA summary and architecture-commit decisions

Spec 01, task 6. Signed off by the developer on 2026-10-17. These decisions bind specs 02–09.
They are mirrored in `.kiro/steering/data-sources.md` and `src/oncora/config.py`.

Evidence: [`eda_profile.md`](eda_profile.md) · [`eda_bridges.md`](eda_bridges.md) ·
[`eda_target.md`](eda_target.md) · [`eda_features.md`](eda_features.md)

## What the data supports

- **Inputs are intact and as described.** All 10 raw files match their checksums, and all 12
  grounding facts reproduce exactly.
- **Compound side:** 972 / 972 DSS3 compounds have structures. They are 961 distinct molecules
  by full InChIKey and 958 by connectivity block. compoundinfo SMILES often lose the stereo that
  their InChIKeys carry.
- **Cell-line side:** 985 trainable lines (CCLE ∩ DSS3). 734 more CCLE lines can be scored at
  inference. No workspace file maps any of them to a disease.
- **Target:** DSS3 mixes scales in 22 compounds. On the other 950, compound identity explains
  ~67–73% of variance and cell-line identity ~4–5%, leaving ~25–30% for selectivity plus noise.
- **Features:** all 978 LINCS landmark genes are present in CCLE. Expression variance is spread
  out (225 PCs for 80%).

## Decision log

| # | Decision | Choice | Rationale | Affects |
|---|---|---|---|---|
| D-01 | Split group key | **Connectivity block of the standardized parent** (`std_inchi_key[:14]`; refined in spec 03 after featurization showed 3,945 salt-form groups) | Stereoisomers and salt forms have identical features; splitting them would leak | 02, 04, 05, 06 |
| D-02 | `pert_id` duplicates in compoundinfo | **Collapse to one row per `pert_id`**; targets, MoAs and names as lists; raise if structures ever conflict | Duplicates are one row per (target, MoA); structure never conflicts | 02, 07 |
| D-03 | Mixed-scale DSS3 compounds (22) | **Exclude from headline training and evaluation**; list them in the model card; one sensitivity analysis with values ≥ 1 rescaled by 1/100 | A row's scale cannot be recovered; including them corrupts labels | 02, 06, 09 |
| D-04 | Regression target | **Rank-transformed DSS3**; headline metric Spearman ρ | Robust to the zero spike and any residual scale differences | 06 |
| D-05 | Sensitive/resistant label | **Within-compound top 10% of cell lines**; global p90 (DSS3 ≥ 0.359 on the single-scale set) reported for comparison | Measures selectivity, the product question, instead of re-labeling potent compounds | 04, 06 |
| D-06 | Cell-line features (MVP) | **978 LINCS landmark genes**, standardized on training folds; top-2,000-variance and 50–100 PCs as STRETCH comparisons | Fixed list (no selection leakage), interpretable, compact | 03, 06 |
| D-07 | Cell line → disease (A4) | **Add DepMap `Model.csv`** (same release as the expression file) at `data/raw/depmap/Model.csv` | The only direct `ModelID` → Oncotree disease/lineage mapping | 06 (Req 4), 08 |
| D-08 | Sensitivity evaluation framing | Report per-cell-line ρ (potency) and per-compound ρ (selectivity) separately; compound-only ablation; explicit selectivity verdict | Compound effects dominate; avoid presenting potency as selectivity | 06, 08, 09 |
| D-09 | L1000 Level 4 signatures (A2 revised, PR #8) | **Evidence-only**: never a feature of any model that must serve unseen compounds; optionally attached as labeled evidence in the interpretation layer | A new SMILES has no measured signature at inference (FR-10) | 06, 07 |

## Open items

- ~~Spec 01 task 5.3: profile the L1000 Level 4 GCTX.~~ **Resolved:** profiled and validated in
  [`eda_signatures.md`](eda_signatures.md). 951 / 960 DSS3 compounds have a measured signature,
  extracted to a 75 MB consensus table. D-09 holds: signatures are evidence in the bundle
  (`annotations.measured_signature`), never a model feature.
- **A4 / D-07: waiting for `Model.csv`.** Until it is added, spec 06 Requirement 4 (cancer-type
  ranking) is blocked. Everything else can proceed. When it arrives: add it to `SHA256SUMS`,
  extend the cell-line bridge, and re-run `oncora.data_prep.bridges`.
- **ChEMBL restore (spec 02, task 2.2)** needs separate approval, because it installs PostgreSQL.
