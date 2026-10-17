---
inclusion: always
---

# Oncora — Technology

**Platform:** Apple M1 Max, 32 GB RAM, macOS/arm64, **CPU-only** (no CUDA). Every dependency
must have native arm64 wheels.

## Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Cheminformatics + ML ecosystem; clean arm64 support |
| Env / deps | `uv` + `pyproject.toml` + lockfile | Fast, reproducible |
| Large table I/O & joins | **DuckDB** (+ **Polars** for in-memory transforms) | Larger-than-RAM CSV/Parquet, SQL joins without full-DataFrame loads. pandas only for small final frames |
| ChEMBL | One-time `pg_restore` → export needed tables to Parquet → DuckDB | No Postgres in the runtime loop |
| Cheminformatics | RDKit | Validation, canonical SMILES, Morgan/ECFP, Bemis–Murcko, descriptors, rendering |
| Models | LightGBM (primary), scikit-learn (baselines, pipelines, metrics) | Strong on fingerprint/tabular features, fast on CPU, feature importances |
| Experiment tracking | MLflow, local file backend (`mlruns/`) | Zero-server run logging |
| Artifacts | `models/<run_id>/` + `model_card.json` | Simple, inspectable |
| Inference API | FastAPI | Typed contract, OpenAPI docs |
| UI | Streamlit | Python-native, fast to build |
| LLM | Anthropic Claude behind an interface, template fallback | Grounded prose only (see `llm-grounding.md`) |
| Tests | pytest | Unit, anti-leakage, join/shape, API contract |

## Deliberately NOT used

- **PostgreSQL at runtime:** only needed once to unpack the ChEMBL dump.
- **PyTorch / GNNs / ChemBERTa:** STRETCH/FUTURE benchmarks only.
- **React:** FUTURE.
- **Cloud services, W&B, model registries:** unneeded for a hackathon.

Add a technology only when a requirement demands it.

## Conventions

- Seeds, paths, thresholds and feature config live in `src/oncora/config.py`, the single source of truth.
  Do not hard-code them elsewhere.
- Secrets (DB creds, `ANTHROPIC_API_KEY`) come from the environment or a gitignored `.env`, never source.
- Each pipeline stage is a runnable module (`python -m oncora.<stage>`) that reads its inputs
  from `data/` and writes provenance-tagged Parquet.
- Prefer DuckDB SQL over `pd.read_csv` for anything in `data/raw/`.

## Common commands (once scaffolded, spec 01 task 1)

```bash
uv sync                      # install pinned deps
uv run pytest                # run tests
uv run mlflow ui             # browse experiment runs
uv run uvicorn oncora.serve.app:app --reload   # API
uv run streamlit run app/main.py               # UI
```
