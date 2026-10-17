#!/usr/bin/env bash
# Full reproducibility pass (spec 09, Req 2; NFR-2, AC-8).
#
#   ./scripts/reproduce.sh
#
# Rebuilds every derived table and report from data/raw, retrains both models, and checks the
# headline metrics against reports/eval/reference_metrics.json. Runs offline and needs no
# ANTHROPIC_API_KEY: the evidence summary falls back to the deterministic template (NFR-9).
#
# One-time prerequisite: the ChEMBL export (see README) has produced data/interim/chembl/*.parquet.
set -euo pipefail
cd "$(dirname "$0")/.."

export MLFLOW_DISABLE_AGENT_HINT=1
START=$(date +%s)

if [ ! -f data/interim/chembl/activities.parquet ]; then
  echo "Missing data/interim/chembl/*.parquet: run the one-time ChEMBL export first (README)." >&2
  exit 1
fi

echo "=== 1/3  Data pipeline"
./scripts/build_data.sh

echo "=== 2/3  Models"
uv run python -m oncora.models.target.train
uv run python -m oncora.models.sensitivity.train

echo "=== 3/3  Metric comparison"
uv run python scripts/compare_metrics.py

echo "Reproduced in $(( ($(date +%s) - START) / 60 )) min."
