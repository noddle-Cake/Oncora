# Design Document — 09 Evaluation Hardening & Submission

## Overview

No new modeling. This spec packages evidence. Each artifact here answers a specific judging criterion.

| Artifact | Answers |
|---|---|
| `reports/demo/<inchi_key>.json` + `.md` commentary | Validation on unseen compounds, biological insight |
| `scripts/reproduce.sh` + README run guide | Reproducibility |
| `reports/eval/{target,sensitivity}/*` (from 05/06) | Scientific soundness |
| `reports/submission.md` | All criteria, in one page |

## Demo compound selection

1. Prefer an approved drug with well-known targets that is in `compoundinfo.txt` (so annotations
   exist) and, ideally, in the DSS3 set. Its sensitivity profile is ground truth if it is held out.
2. Hold it out **before** training: add its InChIKey to `config.DEMO_HOLDOUT` and exclude it from
   both modeling tables at build time (spec 02). This makes it unseen for both models while keeping its labels for comparison.
3. `demo_unseen.py` asserts the InChIKey is absent from both `train_index.parquet` files before running.

## Components

### `scripts/demo_unseen.py`
- Args: `--smiles` or `--inchi-key`; `--via-api` (default: in-process).
- Writes the full `PredictResponse` JSON. If held-out DSS3 labels exist, it also writes a
  side-by-side of predicted versus actual cell-line ranks, with Spearman ρ for this one compound.

### `scripts/reproduce.sh`
```bash
set -euo pipefail
uv sync --frozen
uv run python -m oncora.io.checksums
uv run python -m oncora.io.profile
uv run python -m oncora.io.convert
uv run python -m oncora.data_prep.bridges
uv run python -m oncora.data_prep.chembl_filter      # assumes interim/chembl/*.parquet exported once (spec 02)
uv run python -m oncora.data_prep.target_pairs
uv run python -m oncora.data_prep.sensitivity_table
uv run python -m oncora.features.build
uv run python -m oncora.models.target.train
uv run python -m oncora.models.sensitivity.train
uv run python scripts/compare_metrics.py              # diff vs reports/eval/*/reference.json
```
The ChEMBL Postgres export is a documented one-time prerequisite and is not repeated. The script
checks that its Parquet files exist and match the recorded row counts.

### `scripts/compare_metrics.py`
- Compares fresh metrics with the committed reference metrics within a tolerance (default 1e-6). Exits non-zero on drift.

## Testing Strategy

- `reproduce.sh` is the test: run it once on a fresh clone with the network disabled and `ANTHROPIC_API_KEY` unset.
- The demo script's unseen-assertion is unit-tested with a fake training index.
