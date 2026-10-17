---
inclusion: always
---

# Oncora — Repository Structure

```
Oncora/
├─ .kiro/
│  ├─ steering/          # always-on project context (this file, product, tech, data, rigor, llm)
│  └─ specs/             # one folder per feature spec: requirements.md, design.md, tasks.md
├─ data/
│  ├─ raw/               # provided inputs: read-only, never committed (see data-sources.md)
│  │  ├─ ccle/           #   CCLE expression matrix
│  │  ├─ pharmacodb/     #   DSS3 sensitivity labels
│  │  ├─ chembl/         #   ChEMBL 37 PostgreSQL dump (.tar)
│  │  ├─ lincs/StandardAnnotations/   # LINCS annotation tables
│  │  └─ SHA256SUMS      #   checksums of every raw file (provenance)
│  ├─ interim/           # cleaned per-source Parquet (provenance-tagged)
│  └─ processed/         # final modeling tables
├─ src/oncora/           # (spec 01 task 1 creates this)
│  ├─ config.py          # paths, seeds, thresholds, feature config
│  ├─ io/                # DuckDB/Polars loaders, ChEMBL export, profilers
│  ├─ chem/              # RDKit: validate, canonicalize, fingerprints, scaffolds, descriptors
│  ├─ data_prep/         # identity bridges, filters, joins → processed tables
│  ├─ features/          # compound + cell-line feature builders
│  ├─ evaluation/        # splitters, metrics, baselines
│  ├─ models/target/     # compound → target
│  ├─ models/sensitivity/# compound × cell line → sensitivity; cancer-type aggregation
│  ├─ interpret/         # evidence bundle + renderers (template | Claude)
│  └─ serve/             # FastAPI app
├─ app/                  # Streamlit UI
├─ notebooks/            # numbered EDA notebooks
├─ reports/              # EDA, evaluation, submission reports + figures
├─ models/               # versioned artifacts + model_card.json (gitignored except cards)
├─ mlruns/               # MLflow tracking (gitignored)
├─ tests/
├─ docs/archive/         # superseded documents (e.g. the original monolithic spec)
└─ pyproject.toml
```

## Spec map (build order)

| # | Spec | Covers (original roadmap) | Depends on |
|---|---|---|---|
| 01 | `01-environment-eda` | Phase 0: scaffold, profiling, bridges, target/feature EDA, architecture checkpoint | — |
| 02 | `02-data-pipeline` | Phase 1: ChEMBL restore/export, activity filtering, modeling tables | 01 |
| 03 | `03-feature-engineering` | Phase 2: RDKit compound featurizer, CCLE feature builder | 02 |
| 04 | `04-evaluation-framework` | Phase 3.1: leakage-safe splitters, metrics, trivial baselines | 03 |
| 05 | `05-target-model` | Phase 3.2: Compound → Target | 04 |
| 06 | `06-sensitivity-model` | Phase 3.3–3.4: Compound → Sensitivity, cancer-type aggregation | 04 |
| 07 | `07-interpretation` | Phase 4: evidence bundle, template + Claude renderers | 05, 06 |
| 08 | `08-serving-ui` | Phase 5: FastAPI, Streamlit, batch | 07 |
| 09 | `09-submission` | Phase 6: unseen-compound demo, reproducibility pass, writeup | 08 |

Specs 05 and 06 can proceed in parallel once 04 is done.

## Requirement IDs

The global IDs `FR-x`, `NFR-x`, `DR-x`, `AC-x` from the original spec (archived at
`docs/archive/oncora-spec-v0/`) are kept stable. Each spec's requirements.md lists its local
requirements (`Requirement N`, criteria `N.M`) with the global IDs they implement, and its tasks
reference local criteria (`_Requirements: N.M_`). Grep for a global ID to find every spec that touches it.
