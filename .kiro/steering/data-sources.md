---
inclusion: always
---

# Oncora — Data Sources (verified from the files in this workspace)

All raw inputs live under `data/raw/` and are **read-only**. Checksums are in
`data/raw/SHA256SUMS` (verify with `cd data/raw && shasum -a 256 -c SHA256SUMS`).
Never commit raw data or the ChEMBL dump.

## Inventory

| Source | Path | Shape / format | Key columns |
|---|---|---|---|
| CCLE expression | `data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | 305 MB; 1,775 rows × 19,221 cols; values = log1p(TPM) | unnamed index, `SequencingID`, `ModelConditionID`, `ModelID` (`ACH-…`), `IsDefaultEntryForMC`, `IsDefaultEntryForModel`, then genes as `SYMBOL (ENTREZID)` |
| Sensitivity (DSS3) | `data/raw/pharmacodb/L1000_cp_pertid_Cellosaurus_celllineid_PharmacoDB_dss3.csv` | 21 MB; 512,950 rows | `pert_id` (`BRD-…`), `cell_line` (`ACH-…`), `sensitivity_score` |
| ChEMBL 37 | `data/raw/chembl/chembl_37_postgresql.tar` | 2.16 GB tar → `chembl_37/chembl_37_postgresql/chembl_37_postgresql.dmp` (PG custom-format) + `INSTALL_postgresql` | needs `pg_restore` |
| Compound annotations | `data/raw/lincs/StandardAnnotations/compoundinfo.txt` | TSV, 39,321 rows | `pert_id`, `cmap_name`, `target`, `moa`, `canonical_smiles`, `inchi_key`, `compound_aliases` |
| Perturbagen info | `…/pertinfo.txt` | TSV, 92,207 rows | `pert_id`, `pert_mfc_id`, `cmap_name`, `pert_type` |
| Cell info | `…/cellinfo.txt` | TSV, 240 rows | `cell_iname`, `cellosaurus_id`, `primary_disease`, `cell_lineage`, `ccle_name`, … |
| Gene info | `…/geneinfo.txt` | TSV, 12,328 rows | `gene_id` (Entrez), `gene_symbol`, `ensembl_id`, `feature_space` |
| Small-molecule metadata | `…/Small_Molecule_Metadata.txt` | TSV, 41,508 rows | `sm_name`, `sm_lincs_id`, `sm_smiles_parent`, `sm_inchi_key_parent`, … |
| Instance info | `…/instinfo.txt` | **655 MB** TSV, 2.92 M rows | `sample_id`, `pert_id`, `cell_iname`, dose/time, QC fields |
| Field definitions | `…/DataFieldDefinitions.xlsx` | XLSX | LINCS column dictionary |
| L1000 Level 4 signatures | `data/raw/lincs/level4_beta_trt_cp_n1805898x12328.gctx` | **~80 GB** GCTX (HDF5); **1,805,898 profiles × 12,328 genes**; `trt_cp` (compound perturbation), plate-normalized z-scores | matrix: rows = `rid` (Entrez gene ids, join via `geneinfo.txt`); cols = `cid` (sample/instance ids, join via `instinfo.sample_id` → `pert_id`) |

Annotation files use `""` (a literal pair of quotes) for missing values. Treat it as null.

### Quirks found by the profiler (`reports/eda_profile.md`)

- `Small_Molecule_Metadata.txt` is **not valid UTF-8**: 4 `sm_alternative_name` cells contain a
  stray `0xBF` byte, and the rest of the file is ASCII. Read it as Latin-1.
- `compoundinfo.txt` repeats `pert_id`s (34,419 distinct in 39,321 rows). It also uses the literal
  string **`restricted`** in place of SMILES and InChIKeys for some compounds. Treat it as null.
  `target`/`moa` are 79.5% null.
- `Small_Molecule_Metadata.sm_name` contains an Excel `#ERROR` value.
- `instinfo.pert_time` uses `-666` as a placeholder for missing values.
- DSS3 has no duplicate `(pert_id, cell_line)` pairs and no nulls. 19.9% of scores are exactly 0.
- **DSS3 mixes scales.** All 9,711 trainable rows with DSS3 ≥ 1 (up to 96) come from **22 compounds**,
  mostly CCLE-screen reference drugs measured in several merged studies. The other 950 compounds lie in
  [0, 0.995]. A row's scale cannot be recovered within those 22 compounds (`reports/eda_target.md`).
- **Compound identity explains ~67–73% of DSS3 variance; cell-line identity only ~4–5%.**
- All 978 LINCS landmark genes are present in CCLE (by Entrez ID) (`reports/eda_features.md`).
- CCLE values are log1p(TPM) in [0, 17.36] with no nulls. 5 genes have zero variance.
- DuckDB takes over 60 s to parse the 19k-column CCLE CSV, while Polars takes about 3 s. Use the
  interim Parquet (`data/interim/ccle_expression.parquet`, default entries only).

## Verified facts

- Sensitivity covers **972 compounds × 1,204 cell lines**; DSS3 median ≈ 0.087, p90 ≈ 0.41,
  max ≈ 96, ~20% ≈ 0 (heavily right-skewed).
- **972 / 972** sensitivity compounds resolve to SMILES + InChIKey via `compoundinfo.txt`
  (none are `restricted`). They are **961 distinct molecules** by full InChIKey (11 molecules
  appear under 2 `pert_id`s) and **958 by connectivity block**: three stereo groups, namely
  dexamethasone/betamethasone, doxorubicin/epirubicin, and PD-0325901 with and without stereo.
- compoundinfo `canonical_smiles` often **lose stereochemistry** that `inchi_key` carries: the
  RDKit-recomputed InChIKey differs in the stereo block for 93 of the 972 DSS3 compounds (4% overall).
  Group splits by `inchi_key[:14]` (see `reports/eda_bridges.md`).
- **985** `ACH-` IDs appear in both CCLE and the sensitivity file.
- **CCLE has duplicate models:** 1,775 rows but **1,719 distinct `ModelID`s**. Filter to
  `IsDefaultEntryForModel = Yes` (exactly 1,719 rows) to get one profile per cell line.
- L1000 Level-4 `trt_cp` signatures are **present** (`level4_beta_trt_cp_n1805898x12328.gctx`,
  1,805,898 × 12,328). **A2 revised:** signatures are available as an **optional evidence/interpretation
  source only**. They are **never a required inference input** — a new SMILES pasted by a user has no
  signature, so the sensitivity model's compound features stay structure-only (FR-10). Do not add any
  signature-derived column to a model whose inference path must serve unseen compounds.

## ChEMBL 37 export (spec 02, task 2)

Exported once to `data/interim/chembl/*.parquet` (~0.5 GB, gitignored). Row counts are in
`export_manifest.csv` with the tar's SHA256. `molfile` and protein `sequence` columns are dropped.

| Table | Rows |
|---|---|
| activities | 24,527,044 |
| assays | 1,970,438 |
| molecule_dictionary | 2,921,148 |
| compound_structures | 2,897,819 |
| component_synonyms | 121,157 |
| target_dictionary | 18,552 |
| target_components | 17,284 |
| component_sequences | 12,986 |

Spot check: imatinib (CHEMBL941) ranks ABL1/ABL2, PDGFRα/β, DDR1/2, CSF1R, LCK and NQO2 (a known
off-target) among its most potent human single-protein targets.

## Identity bridges

```
COMPOUND:  pert_id (BRD-…) ──compoundinfo──▶ canonical_smiles, inchi_key, target, moa
           ChEMBL molregno ──compound_structures──▶ SMILES ──RDKit──▶ InChIKey
           Cross-source identity = InChIKey (structure), never name.

CELL LINE: CCLE ModelID (ACH-…) == sensitivity cell_line (ACH-…)
           ACH-… ──?──▶ primary_disease / lineage      ← OPEN (A4)
```

## Checkpoint decisions (spec 01 task 6, signed off 2026-10-17)

Full log with rationale: `reports/eda_summary.md`. Values live in `src/oncora/config.py`.

| # | Decision |
|---|---|
| D-01 | Split group key = InChIKey connectivity block (`inchi_key[:14]`) |
| D-02 | compoundinfo duplicates collapse to one row per `pert_id` (lists of targets, MoAs, names) |
| D-03 | The 22 mixed-scale DSS3 compounds are excluded from headline training/eval; one rescaled (×0.01) sensitivity analysis |
| D-04 | Regression target = rank-transformed DSS3; headline Spearman ρ |
| D-05 | "Sensitive" = top 10% of a compound's cell lines; global p90 (≥ 0.359) as comparison |
| D-06 | Cell-line features = 978 LINCS landmark genes, standardized on training folds |
| D-07 | Cell line → disease via DepMap `Model.csv` at `data/raw/depmap/Model.csv` (**pending: the developer is adding it**) |
| D-08 | Report potency (per-cell-line ρ) and selectivity (per-compound ρ) separately; compound-only ablation |
| D-09 | L1000 Level 4 signatures are evidence-only, never a model feature for unseen compounds |

## Open item A4: cell line → disease mapping

`cellinfo.txt` has **no `ACH-` IDs** and covers only 240 LINCS cell lines (138 with a
`ccle_name`). The CCLE expression file has no lineage or disease columns. **No file in the
workspace maps the 985 trainable `ACH-` lines to a disease**, which FR-11 needs.

Confirmed by `reports/eda_bridges.md`: **0 / 985** trainable lines can be mapped from workspace files.

**Decided (D-07): option 1.** Candidates considered:
1. Add DepMap `Model.csv` from the same release as the expression file (`ModelID` →
   `OncotreeLineage`, `OncotreePrimaryDisease`, `CCLEName`, `RRID`). This is the most direct fix.
2. Map via Cellosaurus (`ACH-` → RRID → disease) using a Cellosaurus export.
3. Fall back to `cellinfo.txt` for the subset matched by CCLE name. Coverage would be poor.

## Memory discipline

- Never load `instinfo.txt` whole; query it with DuckDB or scan it lazily with Polars.
- 32 GB allows the CCLE matrix in memory, but read it once, convert to Parquet in `data/interim/`,
  and use the Parquet from then on.
- **Never materialize the GCTX whole.** `level4_beta_trt_cp_n1805898x12328.gctx` is an HDF5 matrix;
  full load ≈ 1.81M × 12,328 floats ≈ ~90 GB in RAM — it will not fit. Read it with `cmapPy`
  (`cmapPy.pandasGEXpress.parse` with explicit `rid`/`cid` subsetting) or `h5py`, slicing only the
  genes (`rid`) and profiles (`cid`) you need. Resolve the `cid` → `pert_id` link via
  `instinfo.sample_id`, and `rid` → symbol via `geneinfo.txt`. Confirm free disk space before the
  download completes (≥ ~100 GB headroom including derived slabs); gitignore it (`*.gctx`).
