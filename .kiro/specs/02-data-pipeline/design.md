# Design Document — 02 Data Pipeline

## Overview

Three stages, each independently runnable, each writing provenance-tagged Parquet:

1. **ChEMBL export:** dump → local Postgres (once) → Parquet.
2. **ChEMBL filter:** quality filter, pChEMBL standardization, censoring policy, dedup.
3. **Modeling tables:** `target_pairs.parquet` and `sensitivity.parquet`.

## Architecture

```
chembl_37_postgresql.tar ──tar -x──▶ .dmp ──pg_restore──▶ local PG (chembl_37)
                                                          │  COPY … TO parquet (via DuckDB postgres extension)
                                                          ▼
                                     data/interim/chembl/{activities,assays,…}.parquet
                                                          │  data_prep/chembl_filter.py
                                                          ▼
                                     data/interim/chembl_activities.parquet
                                                          │  data_prep/target_pairs.py
                                                          ▼
                                     data/processed/target_pairs.parquet

interim/sensitivity.parquet + compound_bridge + ccle_expression(model_id)
                                                          │  data_prep/sensitivity_table.py
                                                          ▼
                                     data/processed/sensitivity.parquet
```

**Why the DuckDB `postgres` extension for the export:** a single `ATTACH` followed by
`COPY (SELECT …) TO 'x.parquet'` per table. No pandas round-trip, and it handles the
multi-GB `activities` table in streaming mode.

**Postgres install:** `brew install postgresql@16` (pg_restore is backward-compatible with the
PG 10.15 dump format). It is needed only for the export and can be uninstalled afterwards.

## Components and Interfaces

### `src/oncora/io/chembl_export.py`
- `extract_dump(tar_path, out_dir) -> Path`
- `restore(dump_path, dbname="chembl_37") -> None`: shells out to `createdb` and `pg_restore --no-owner -j 8`.
- `export_tables(tables: list[str], out_dir) -> dict[str, int]`: returns row counts.
- `spot_check(chembl_id="CHEMBL941") -> pl.DataFrame`
- Env: `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`.

### `src/oncora/data_prep/chembl_filter.py`
- `filter_activities(cfg) -> pl.LazyFrame`. Joins `activities ⋈ assays ⋈ target_dictionary ⋈
  molecule_dictionary ⋈ compound_structures` and applies Requirement 2's filters.
- Censoring policy (the active threshold is T; for `>` rows, implied pChEMBL = −log10(value in M)):
  | relation | rule |
  |---|---|
  | `=` | use `pchembl_value` |
  | `>`, `>=` | keep as inactive if implied pChEMBL < T, else drop |
  | `<`, `<=` | drop (ambiguous) |
- `dedup(df) -> pl.DataFrame`: group by `(inchi_key, target_chembl_id)` and take the median pChEMBL plus `n_measurements`.

### `src/oncora/data_prep/target_pairs.py`
- `build_target_pairs(cfg) -> pl.DataFrame`. Applies the threshold, measured negatives, seeded
  sampled negatives, and the minimum-positives filter.
- Target gene symbol from `component_synonyms` (`syn_type = 'GENE_SYMBOL'`). Spec 07 uses it to look up CCLE expression.

### `src/oncora/data_prep/sensitivity_table.py`
- `build_sensitivity(cfg) -> pl.DataFrame`

### `src/oncora/data_prep/provenance.py`
- `write_parquet(df, path, *, sources, filters, config) -> None`: stores the metadata in the Parquet key-value footer.
- `read_provenance(path) -> dict`
- `assert_table(df, *, required_cols, max_null_rate, min_rows, name) -> None`: raises `DataIntegrityError`.

## Data Models

**`processed/target_pairs.parquet`**
| column | type | notes |
|---|---|---|
| inchi_key | str | compound identity |
| canonical_smiles | str | RDKit-canonical |
| target_chembl_id | str | |
| target_gene_symbol | str? | for expression evidence |
| uniprot_accession | str? | |
| pchembl_median | float? | null for sampled negatives |
| label | int8 | 1 active / 0 inactive |
| negative_source | str? | `measured` \| `sampled` \| null for positives |

**`processed/sensitivity.parquet`**
| column | type |
|---|---|
| inchi_key | str |
| canonical_smiles | str |
| pert_ids | list[str] |
| model_id | str (ACH-) |
| dss3 | float |

Config (in `config.py`): `CHEMBL_MIN_CONFIDENCE=8`, `ACTIVE_PCHEMBL=6.0`, `NEG_RATIO=3`,
`MIN_POSITIVES_PER_TARGET=30`.

## Error Handling

- `pg_restore` not on PATH → an error with install instructions; nothing partial is written.
- Every `assert_table` failure names the stage, the file, and the violated bound.
- A sampled negative that collides with a measured activity for the same pair is a bug. It is asserted against.

## Testing Strategy

- `tests/test_chembl_filter.py`: fixture rows for each relation type check the censoring policy and the dedup median.
- `tests/test_target_pairs.py`: seeded sampling is deterministic, no sampled negative is a measured pair, targets below the minimum are dropped.
- `tests/test_sensitivity_table.py`: InChIKey merge, restriction to CCLE cell lines.
- `tests/test_provenance.py`: metadata round-trips.
