# Design Document — 01 Environment & Exploratory Data Analysis

## Overview

Phase 0 creates the repository and answers one question: *what do the data actually support?*
Nothing here trains a model. Every output is a report or an interim Parquet file that later
specs consume. Principle: **inspect before committing to an architecture.**

## Architecture

```
data/raw/*  ──verify SHA256──▶  io/profile.py ──▶ reports/eda_profile.md
     │                                │
     │                                └──▶ data/interim/ccle_expression.parquet   (default entries only)
     │                                     data/interim/sensitivity.parquet
     │                                     data/interim/compoundinfo.parquet
     ▼
data_prep/bridges.py ──▶ data/interim/compound_bridge.parquet
                         data/interim/cellline_bridge.parquet
                         reports/eda_bridges.md
     ▼
notebooks/01_dss3_target.ipynb, 02_ccle_features.ipynb ──▶ reports/figures/*, §Target, §Features
     ▼
reports/eda_summary.md  (decision log) ──▶ copied into steering/data-sources.md
```

Converting the CCLE CSV to Parquet once removes the slow 305 MB CSV parse from every later stage.

## Components and Interfaces

### `src/oncora/config.py`
```python
ROOT: Path; DATA_RAW: Path; DATA_INTERIM: Path; DATA_PROCESSED: Path; REPORTS: Path; MODELS: Path
RAW = {  # logical name -> path under data/raw
    "ccle": "ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
    "dss3": "pharmacodb/L1000_cp_pertid_Cellosaurus_celllineid_PharmacoDB_dss3.csv",
    "chembl_tar": "chembl/chembl_37_postgresql.tar",
    "compoundinfo": "lincs/StandardAnnotations/compoundinfo.txt",
    ...  # cellinfo, geneinfo, pertinfo, instinfo, sm_metadata
}
SEED: int = 42
NULL_TOKENS = ("", '""')
```

### `src/oncora/io/checksums.py`
- `verify_raw(manifest=DATA_RAW/"SHA256SUMS") -> None`: raises `RawDataError` naming any missing or altered file.

### `src/oncora/io/profile.py`
- `profile_table(path, sep, null_tokens) -> TableProfile`: uses DuckDB `read_csv(..., sample_size=-1)`
  for exact types. For `instinfo.txt` it computes aggregates in SQL only.
- `TableProfile`: `name, n_rows, n_cols, columns: list[ColumnProfile]`. Each `ColumnProfile` has
  `name, dtype, null_rate, n_distinct, min, max, example_values, id_pattern` (regex such as `^ACH-\d{6}$`, `^BRD-`).
- `render_profile_md(profiles) -> str`.
- CLI: `python -m oncora.io.profile` writes `reports/eda_profile.md`.

### `src/oncora/io/convert.py`
- `ccle_to_parquet()`: keeps rows with `IsDefaultEntryForModel == "Yes"`, drops the unnamed index
  column, and writes long-format column metadata (`gene_symbol`, `entrez_id`, parsed from `SYMBOL (ENTREZID)`) to a sidecar Parquet.

### `src/oncora/data_prep/bridges.py`
- `build_compound_bridge() -> pl.DataFrame` with columns `pert_id, inchi_key, canonical_smiles, cmap_name, target, moa, n_source_rows`.
- `build_cellline_bridge() -> pl.DataFrame` with columns `model_id, in_ccle, in_dss3, primary_disease, lineage, disease_source`.
- `bridge_report(...) -> str`.

## Data Models

| File | Grain | Key |
|---|---|---|
| `interim/ccle_expression.parquet` | one row per ModelID (1,719) | `model_id` |
| `interim/ccle_genes.parquet` | one row per gene column | `gene_col` → `gene_symbol`, `entrez_id` |
| `interim/sensitivity.parquet` | one row per (pert_id, cell_line) | composite |
| `interim/compound_bridge.parquet` | one row per pert_id | `pert_id` |
| `interim/cellline_bridge.parquet` | one row per ACH id in CCLE ∪ DSS3 | `model_id` |

Every Parquet carries key-value metadata: `source_files` (with SHA256), `created_by` (module), `created_at`, `filters`.

## Error Handling

- Checksum mismatch → `RawDataError`, raised before any work starts.
- Grounding-number drift (for example, the overlap ≠ 985) → a warning in the report, not a crash.
  The point of the EDA is to discover such drift. The value is recorded.
- Duplicate `(pert_id, cell_line)` rows in DSS3 → counted and reported. The dedup rule is decided at the checkpoint.

## Testing Strategy

- `tests/test_checksums.py`: a tampered temp file is detected.
- `tests/test_bridges.py`: on small fixture TSVs, `""` is parsed as null, the CCLE default-entry
  filter keeps one row per model, and bridge coverage is computed correctly.
- The profile itself is validated by the developer review in Requirement 5.
