# Design Document — 04 Evaluation Framework

## Overview

**Why this comes before models.** The naive trap is a random row split. Each compound appears in
hundreds of sensitivity rows (or several target pairs), so it ends up in both train and test and
the model scores well by memorizing it. Building and testing the splitters first means no model
can ever be judged on a leaky split.

## Architecture

```
modeling table ──▶ Splitter(strategy, seed) ──▶ SplitManifest + fold indices
                                                     │
            model.fit/predict ◀──────────────────────┤
            baseline.fit/predict ◀───────────────────┘
                     │
                     ▼
               metrics.* ──▶ bootstrap CIs ──▶ EvalReport (md + json) ──▶ MLflow artifact
```

## Components and Interfaces

### `src/oncora/evaluation/splits.py`
```python
Strategy = Literal["compound", "scaffold"]

@dataclass(frozen=True)
class SplitManifest:
    strategy: Strategy
    seed: int
    n_folds: int
    group_counts: list[int]          # groups per fold
    partition_hashes: list[str]      # sha256 of sorted connectivity blocks per fold

def compound_folds(df, *, n_folds, seed, key="connectivity") -> tuple[list[np.ndarray], SplitManifest]
def scaffold_folds(df, *, n_folds, seed, scaffold_col="scaffold", key="connectivity") -> ...
def holdout(df, *, strategy, test_frac, seed) -> (train_idx, test_idx, SplitManifest)
def assert_disjoint(df, folds, col) -> None   # raises LeakageError
```
- Compound folds: `GroupKFold`-style over unique standardized connectivity blocks (`std_inchi_key[:14]`, joined from
  `compound_features.parquet`) after a seeded shuffle. Standardization also merges salt forms.
  The connectivity block, not the full InChIKey, is the group: stereoisomers (e.g. dexamethasone/betamethasone)
  have identical achiral ECFP features, and compoundinfo SMILES often lack the stereo their InChIKeys carry.
- Scaffold folds: largest-first greedy assignment of scaffold groups to the currently smallest
  fold (by compound count), after a seeded shuffle within size ties. Empty scaffold → group = InChIKey.

### `src/oncora/evaluation/baselines.py`
```python
class PriorBaseline:          # target: per-target positive rate
class PopularityRanker:       # target: rank targets by training positive count
class GlobalMeanBaseline:     # sensitivity
class CellLineMeanBaseline:   # sensitivity: mean transformed DSS3 per model_id (train only); unseen line → global mean
```
They share a `fit(train_df)` / `predict(df)` interface with the models.

### `src/oncora/evaluation/metrics.py`
- `target_metrics(y_true, y_score, target_ids, compound_ids) -> dict`
- `sensitivity_metrics(y_true_raw, y_pred_transformed, compound_ids, transform, threshold) -> dict`
- `bootstrap_ci(metric_fn, df, group_col, n=1000, seed) -> (lo, hi)`: resamples **compounds**, not rows.

### `src/oncora/evaluation/report.py`
- `EvalReport.from_results(model_results, baseline_results, manifests) -> EvalReport`
- `.to_markdown()`, `.to_json()`, `.beats_baselines() -> dict[str, bool]`

## Data Models

`reports/eval/<model>/<run_id>.{md,json}`. The JSON contains metrics by split, CIs, the
baseline comparison, and the manifests.

## Error Handling

- `LeakageError` from `assert_disjoint` is raised inside every training entry point, not only in tests.
- A fold with zero positives for a target → that target is excluded from per-target AUC for that
  fold, and the exclusion is counted.

## Testing Strategy

- `tests/test_splits.py` (the anti-leakage suite):
  - property-style: for random synthetic tables, every strategy has zero key overlap;
  - scaffold strategy has zero scaffold overlap;
  - negative control: `sklearn.model_selection.KFold` on rows is detected by `assert_disjoint`;
  - same seed → identical manifests.
- `tests/test_baselines.py`: cell-line-mean is fit on train rows only (a test-row value change does not change predictions).
- `tests/test_metrics.py`: hand-computed small examples; bootstrap resamples by group.
