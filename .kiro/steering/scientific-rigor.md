---
inclusion: always
---

# Oncora — Scientific Rigor Rules

These rules apply to every modeling, evaluation, and data-prep change. A breach is a bug.

## Leakage (the central risk)

- **Never use a random row split for a headline metric.** The same compound would sit in train
  and test, and the model would look good by memorizing it.
- Headline splits are **compound-disjoint by InChIKey** for both models. The sensitivity model
  also reports a **Bemis–Murcko scaffold-disjoint** split.
- Sensitivity CV is **grouped by compound** (`GroupKFold`-style), never by row.
- Fit anything learned from data (gene variance filter, scalers, descriptor normalization,
  imputation, thresholds tuned on data) **on the training fold only**.
- Every split has an automated test that asserts zero InChIKey (and scaffold) overlap.
- **No LINCS signature may be a required feature.** Compound features are derived from structure only (FR-10).

## Baselines and metrics

- Every model reports against a **trivial baseline** on the same leakage-safe split, and must beat it:
  - target: class prior / most-frequent;
  - sensitivity: global mean **and** cell-line mean. The cell-line-mean baseline tests whether
    the model has learned anything about the compound.
- Choose metrics for imbalance and skew:
  - target: PR-AUC (primary), ROC-AUC, top-k retrieval;
  - sensitivity: Spearman ρ, RMSE on the transformed target, and thresholded sensitive/resistant AUC.

## Reproducibility and provenance

- Fixed seeds from `config.py`. Derived tables record their source files (with SHA256), filters, and code version.
- Each model artifact ships a `model_card.json`: data snapshot hash, feature config, split
  strategy, label/threshold policy, negative-construction rule, metrics, seed.
- Each training run is logged to MLflow.

## Fail loudly

Data stages assert non-empty outputs, expected columns, and null-rate bounds. On a violation they
raise an error that names the file and filter. They never write an empty or silently truncated table.

## Honest communication

- Separate **model prediction** from **database annotation** everywhere.
- A target prediction plus a sensitivity prediction is **correlational evidence, not proof** (D4).
- Flag near-seen compounds (Tanimoto ≥ 0.99 to training data) so they are not presented as unseen-compound results.
