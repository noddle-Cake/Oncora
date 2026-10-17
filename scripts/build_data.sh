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
run oncora.features.build                  # processed/compound_features.parquet (~3.5 min first time, cached after)

run oncora.evaluation.floor                # reports/eval/baselines_*.md

# Optional evidence (D-09): L1000 signatures. Extract when the 89 GB GCTX is present (~1.5 min);
# once extracted the GCTX may be deleted, and the report is rebuilt from the 75 MB extract.
if [ -f data/raw/lincs/level4_beta_trt_cp_n1805898x12328.gctx ]; then
  run oncora.data_prep.signatures          # interim/l1000_consensus.parquet
fi
if [ -f data/interim/l1000_consensus.parquet ]; then
  run oncora.eda.signatures                # reports/eda_signatures.md
else
  echo "==> skipping L1000 signatures (optional; no GCTX or extract)"
fi

echo "Done. Modeling tables are in data/processed/."
