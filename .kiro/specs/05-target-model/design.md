# Design Document — 05 Compound → Target Model

## Overview

**Per-target binary LightGBM** over ECFP4 and descriptors. Per-target models (rather than one
multi-label model) let each target have its own negatives, as constructed in spec 02, and are
easy to calibrate and inspect. Cost: about one model per qualifying target. LightGBM on ~2k
sparse-ish features trains in seconds per target on CPU. `MAX_TARGETS` (config) caps the count if
the target list after the minimum-positives filter is too large.

## Architecture

```
target_pairs.parquet ─┐
compound_features ────┴─▶ global compound-disjoint holdout (spec 04)
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        for each target t:          LogReg-on-ECFP (comparison)
          LightGBM_t.fit(train_t)
          calibrator_t.fit(val_t)
                 │
                 ▼
     eval on test (PR/ROC-AUC per target, top-k per compound) + baselines ──▶ EvalReport
                 │
                 ▼
     models/target/<run_id>/  { lgbm/*.txt, calibrators.pkl, targets.parquet,
                                train_index.parquet, model_card.json }
```

The global split is done once on InChIKeys and then applied per target. This keeps top-k
evaluation per compound honest: no test compound was seen for any target.

## Components and Interfaces

### `src/oncora/models/target/train.py`
- `train(cfg) -> RunResult`. CLI: `python -m oncora.models.target.train`.
- Hyperparameters (config): `num_leaves=31, learning_rate=0.05, n_estimators≤500 with early
  stopping on the val fold, feature_fraction=0.5, min_child_samples=10, seed`. `scale_pos_weight` is
  left at the default, because calibration handles probability scale and the PR-AUC metric handles imbalance.

### `src/oncora/models/target/predict.py`
```python
@dataclass
class TargetPrediction:
    target_chembl_id: str
    pref_name: str
    gene_symbol: str | None
    uniprot: str | None
    score: float           # calibrated probability
    rank: int

class TargetModel:
    def predict(self, features: np.ndarray, top_n: int = 20) -> list[TargetPrediction]
    def explain(self, mol, target_chembl_id, k=5) -> list[BitAttribution]   # STRETCH
def load_target_model(run_id: str | None = None) -> TargetModel  # None → latest
```

### Attribution (STRETCH)
`featurize` exposes `bitInfoMap` from `rdFingerprintGenerator` `AdditionalOutput`. LightGBM
`pred_contrib=True` gives per-feature SHAP. The top positive bits are mapped to `(atom_idx, radius)`
environments for highlighting in the UI.

## Data Models

- `targets.parquet`: `target_chembl_id, pref_name, gene_symbol, uniprot, n_pos, n_neg_measured, n_neg_sampled, test_pr_auc`
- `train_index.parquet`: `inchi_key, fp (packed bits)`, used by `TrainingIndex` (spec 03).
- `model_card.json`: the schema shared with spec 06 (see `evaluation/report.py`).

## Error Handling

- A target with fewer than 2 classes in the train or val fold is skipped and recorded in the model card.
- A missing artifact → `load_target_model` raises with the expected path and the available run IDs.

## Testing Strategy

- A synthetic mini-dataset (3 targets, 200 compounds) trains end-to-end in under 10 s in CI.
- Round trip: train, save, load, predict gives identical scores.
- A guard test that the global split has no test InChIKey in any target's train set.
