# Oncora

Speeding cancer drug discovery with explainable AI (eMerge Hackathon Challenge 6).

Given a compound as a SMILES string, Oncora predicts likely protein targets, sensitive cancer
cell lines, and cancer types worth investigating, and explains the evidence behind each.

> **These are computational research predictions, not validated clinical conclusions.**

## Setup

```bash
brew install uv libomp   # once; libomp is required by LightGBM on macOS
uv sync                  # creates .venv with pinned deps (Python 3.11)
uv run pytest
```

Raw data goes in `data/raw/` (not committed). Verify it with:

```bash
uv run python -m oncora.io.checksums
```

## One-time: ChEMBL export

ChEMBL ships as a PostgreSQL dump. It is restored once (8 tables, no indexes), exported to
`data/interim/chembl/*.parquet`, and the database is dropped. Later stages need no Postgres.

```bash
brew install postgresql@16
pg_ctl -D /opt/homebrew/var/postgresql@16 -l data/interim/chembl/postgres.log start
uv run python -m oncora.io.chembl_export all      # extract, restore, export, spot-check, cleanup (~3 min)
pg_ctl -D /opt/homebrew/var/postgresql@16 stop
```

## Rebuild the data

```bash
./scripts/build_data.sh   # ~1.5 min: checksums → profile → Parquet → bridges → EDA → ChEMBL filter → modeling tables
```

Every report in `reports/` is regenerated and should come out byte-identical; a diff means the
data or the code changed.

## Project docs

- `.kiro/steering/`: product, tech, structure, data sources, rigor rules
- `.kiro/specs/01-…09-*/`: requirements, design, and tasks per feature, in build order
