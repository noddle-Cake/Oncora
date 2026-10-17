# Design Document — 06 Compound → Sensitivity Model & Cancer-Type Aggregation

## Overview

One pairwise model: **(compound, cell line) → sensitivity**. Compound features come from structure
(spec 03). Cell-line features are a fixed vector per CCLE line. Because both halves can be computed
for any new SMILES and any CCLE line, inference for unseen compounds needs no measured data.

## Why two target framings

DSS3 is heavily right-skewed (median ≈ 0.087, max ≈ 96, ~20% near zero).
- **Regression on a transform** (log1p or rank), measured mainly by Spearman ρ: ranking lines is the product goal, and ρ ignores the skew.
- **Classification at a threshold**, measured by ROC/PR-AUC: gives an interpretable, imbalance-aware "sensitive vs. resistant" metric.

## The baseline trap and how the metrics expose it

Much of DSS3 variance is driven by the cell line (some lines are sensitive to almost everything).
A cell-line-mean predictor therefore scores well on overall ρ and even on per-compound ρ. It gives
every compound the same cell-line ranking, which happens to correlate with the truth. It scores
**≈ 0 on per-cell-line ρ** (ranking compounds within a line), because it is constant across
compounds. Per-cell-line ρ and thresholded PR-AUC are therefore the metrics that show whether the
model learned anything from structure. The acceptance gate (Requirement 2.5) is tied to them.

## Architecture

```
sensitivity.parquet (inchi_key, model_id, dss3)
        │  split by compound | scaffold (spec 04)
        ▼
for each fold:
  fit cell-line gene subset on train lines ─▶ cellline matrix
  X = [compound_feats[inchi_key] ‖ cellline_feats[model_id]]   (float32)
  y = transform(dss3);  y_cls = dss3 ≥ threshold
  LightGBM regressor + classifier  (+ GlobalMean, CellLineMean baselines)
        ▼
EvalReport (compound split, scaffold split) ─▶ final fit on all data ─▶ artifact
```

**Memory:** ~400k rows × (2,058 compound + G gene) features in float32. With G = 1,000 that is
≈ 4.9 GB, which fits in 32 GB. Build the matrix by indexing two small per-entity arrays
(`compound_feats[c_idx]`, `cell_feats[l_idx]`) instead of materializing a joined DataFrame. If
memory gets tight, use LightGBM `Dataset` with `free_raw_data=True`, or reduce G.

## Components and Interfaces

### `src/oncora/models/sensitivity/train.py`
- `train(cfg) -> RunResult`. CLI: `python -m oncora.models.sensitivity.train`.
- `build_design_matrix(pairs, compound_feats, cell_feats) -> (X, y, groups)`
- `assert_structure_only(feature_names, compound_cfg, cell_cfg)` (Requirement 1.4)

### `src/oncora/models/sensitivity/predict.py`
```python
@dataclass
class CellLinePrediction:
    model_id: str
    cell_line_name: str | None
    pred_dss3: float
    p_sensitive: float
    rank: int

class SensitivityModel:
    def predict(self, compound_vec: np.ndarray, top_n: int = 25) -> list[CellLinePrediction]
def load_sensitivity_model(run_id: str | None = None) -> SensitivityModel
```
Inference = tile the compound vector against the stored cell-line matrix (≈ 1–2k rows), then run
one batched predict.

### `src/oncora/models/sensitivity/aggregate.py`
```python
@dataclass
class CancerTypePrediction:
    primary_disease: str
    lineage: str | None
    score: float              # mean of top-k predicted (transformed) sensitivity
    frac_sensitive: float
    n_cell_lines: int
    low_support: bool
    rank: int
def aggregate(preds: list[CellLinePrediction], disease_map, k=3, min_lines=3) -> list[CancerTypePrediction]
```

## Data Models

- `disease_map.parquet`: `model_id, cell_line_name, primary_disease, lineage, source`. The source comes from the A4 resolution.
- `cellline_matrix.parquet`: the fitted, final cell-line feature matrix for every CCLE default-entry line.

## Error Handling

- The disease map is missing (A4 unresolved) → `aggregate` raises `DiseaseMapMissing` and the API
  returns cell-line rankings with `cancer_types = []` and a note. The rest of the system works without it.
- A non-finite prediction → an assertion failure in eval; it is never silently clipped.

## Testing Strategy

- `assert_structure_only` rejects a feature named like an L1000 signature column.
- Fold-wise gene-subset fitting: a test cell line's values do not affect the selected genes.
- Aggregation on fixtures: top-k mean, low-support flag, unmapped lines counted.
- Round trip: save, load, predict gives identical output.
