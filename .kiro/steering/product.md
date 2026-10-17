---
inclusion: always
---

# Oncora — Product

**Mission:** Speeding cancer drug discovery with explainable AI.
**Challenge:** eMerge Hackathon Challenge 6 — Compound → Target → Cancer.

Oncora accepts a chemical compound as a **SMILES string** and returns:

1. likely **protein targets** (ranked, scored),
2. **sensitive cancer cell lines** (ranked),
3. **cancer types / lineages** warranting investigation (ranked),
4. a **plausible biological explanation** that cites its evidence.

It must work for compounds **not seen during training** — this is the judged capability.

## Non-negotiable disclaimer

> **These are computational research predictions, not validated clinical conclusions.**

The disclaimer is a product requirement, not boilerplate. It appears on every results surface:
every UI view and every API response (as a field).

## Locked scope decisions

| ID | Decision |
|---|---|
| D1 | Both models (Compound→Target via ChEMBL; Compound→Sensitivity via PharmacoDB + CCLE) ship as end-to-end **baselines** in the MVP. The bar is honest validation, not state-of-the-art. |
| D2 | UI = **Streamlit** demo + thin **FastAPI** inference service. React is FUTURE. |
| D3 | **CPU-only**, single machine (M1 Max, 32 GB). RDKit + LightGBM / scikit-learn. PyTorch never on the MVP critical path. |
| D4 | The two models are trained and validated **independently**; combined only in the interpretation layer. One model's output is never evidence the other is correct. |
| D5 | Claude is a **grounded explanation renderer only** — never a predictor, never fed raw data rows. A deterministic template fallback keeps everything reproducible with no API key. Budget ≈ $100. |
| DR-6 | Target prediction = **binary classification with constructed negatives**. |
| FR-11 | Cancer-type ranking uses **primary disease + lineage** (see open item A4 in `data-sources.md`). |

## Priority tiers

- **MVP** — required for a defensible, judgeable submission.
- **STRETCH** — valuable if time allows; never blocks MVP.
- **FUTURE** — out of scope: generative design (Challenge 7), React, GPU/deep models, user
  accounts, any clinical or patient-facing interpretation.

## Judging criteria → where the evidence lives

| Criterion | Evidence | Specs |
|---|---|---|
| Scientific soundness | pChEMBL standardization, documented label/negative policy, trivial baselines, imbalance/skew-aware metrics | 02, 04, 05, 06 |
| Validation on unseen compounds | InChIKey-disjoint + scaffold-disjoint headline splits, anti-leakage tests, unseen-compound demo | 04, 06, 09 |
| Biological insight | Evidence bundle linking targets ↔ lineages ↔ annotations ↔ expression | 07 |
| Originality | Dual-model pipeline unified by a grounded interpretation layer; structure-only inference | 06, 07 |
| Reproducibility | Pinned env, seeds, provenance, MLflow, Postgres-free Parquet, offline template summary | 01, 02, 09 |

## Project character (NFR-5)

Each pipeline stage is independently runnable and testable. This is also an **applied
data-science learning artifact** for the developer. Code and docs
explain *why* a technique is used (short rationale comments, "Learn" notes in tasks). Prefer
clarity over cleverness.
