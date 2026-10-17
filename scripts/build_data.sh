#!/usr/bin/env bash
# Rebuild every derived data table and EDA report from data/raw, in dependency order.
# Prerequisite: the one-time ChEMBL export (see README) has produced data/interim/chembl/*.parquet.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! ls data/interim/chembl/activities.parquet >/dev/null 2>&1; then
  echo "Missing data/interim/chembl/*.parquet: run the one-time ChEMBL export first (README)." >&2
  exit 1
fi

run() { echo "==> $1"; uv run python -m "$1"; }

run oncora.io.checksums                    # raw files match SHA256SUMS
run oncora.io.profile                      # reports/eda_profile.md
run oncora.io.convert                      # interim CCLE + DSS3 Parquet
run oncora.data_prep.bridges               # compound + cell-line bridges, reports/eda_bridges.md
run oncora.eda.target                      # reports/eda_target.md
run oncora.eda.features                    # reports/eda_features.md
run oncora.data_prep.chembl_filter         # interim/chembl_activities.parquet
run oncora.data_prep.target_pairs          # processed/target_pairs.parquet
run oncora.data_prep.sensitivity_table     # processed/sensitivity.parquet

echo "Done. Modeling tables are in data/processed/."
