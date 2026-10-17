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

Annotation files use `""` (a literal pair of quotes) for missing values. Treat it as null.

## Verified facts

- Sensitivity covers **972 compounds × 1,204 cell lines**; DSS3 median ≈ 0.087, p90 ≈ 0.41,
  max ≈ 96, ~20% ≈ 0 (heavily right-skewed).
- **972 / 972** sensitivity compounds resolve to SMILES + InChIKey via `compoundinfo.txt`.
- **985** `ACH-` IDs appear in both CCLE and the sensitivity file.
- **CCLE has duplicate models:** 1,775 rows but **1,719 distinct `ModelID`s**. Filter to
  `IsDefaultEntryForModel = Yes` (exactly 1,719 rows) to get one profile per cell line.
- L1000 Level-4 signature matrices are **absent** (annotations only). Signatures are never a
  required inference input (A2).

## Identity bridges

```
COMPOUND:  pert_id (BRD-…) ──compoundinfo──▶ canonical_smiles, inchi_key, target, moa
           ChEMBL molregno ──compound_structures──▶ SMILES ──RDKit──▶ InChIKey
           Cross-source identity = InChIKey (structure), never name.

CELL LINE: CCLE ModelID (ACH-…) == sensitivity cell_line (ACH-…)
           ACH-… ──?──▶ primary_disease / lineage      ← OPEN (A4)
```

## Open item A4: cell line → disease mapping

`cellinfo.txt` has **no `ACH-` IDs** and covers only 240 LINCS cell lines (138 with a
`ccle_name`). The CCLE expression file has no lineage or disease columns. **No file in the
workspace maps the 985 trainable `ACH-` lines to a disease**, which FR-11 needs.

Candidate resolutions (decide in spec 01, task 5):
1. Add DepMap `Model.csv` from the same release as the expression file (`ModelID` →
   `OncotreeLineage`, `OncotreePrimaryDisease`, `CCLEName`, `RRID`). This is the most direct fix.
2. Map via Cellosaurus (`ACH-` → RRID → disease) using a Cellosaurus export.
3. Fall back to `cellinfo.txt` for the subset matched by CCLE name. Coverage would be poor.

## Memory discipline

- Never load `instinfo.txt` whole; query it with DuckDB or scan it lazily with Polars.
- 32 GB allows the CCLE matrix in memory, but read it once, convert to Parquet in `data/interim/`,
  and use the Parquet from then on.
