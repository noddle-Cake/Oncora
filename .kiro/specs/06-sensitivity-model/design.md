# Design Document — 06 Compound → Sensitivity Model & Cancer-Type Aggregation

## Overview

One pairwise model: **(compound, cell line) → sensitivity**. Compound features come from structure
(spec 03). Cell-line features are a fixed vector per CCLE line. Because both halves can be computed
for any new SMILES and any CCLE line, inference for unseen compounds needs no measured data.

## Why two target framings

DSS3 is heavily right-skewed (median ≈ 0.087, ~20% exactly zero). It is also **not on a single
scale**: every value ≥ 1 (up to 96) comes from 22 mixed-scale compounds, mostly CCLE-screen
reference drugs. The other 950 compounds lie in [0, 0.995]. Handling follows the spec 01
checkpoint decision (recommended: exclude from headline training, plus one rescaled
sensitivity analysis). See `reports/eda_target.md`.
- **Regression on a transform** (log1p or rank), measured mainly by Spearman ρ: ranking lines is the product goal, and ρ ignores the skew.
- **Classification at a threshold**, measured by ROC/PR-AUC: gives an interpretable, imbalance-aware "sensitive vs. resistant" metric.

## What the variance decomposition says about metrics and baselines

Measured in spec 01 (`reports/eda_target.md`, single-scale compounds): **compound identity
explains ~67–73% of DSS3 variance, cell-line identity only ~4–5%**, and an additive model ~70–76%.
The remaining ~25–30% is compound × cell-line interaction plus noise, i.e. selectivity.

- **Per-cell-line ρ** (ranking compounds within a line) is dominated by overall compound potency,
  which must be predicted from structure. The cell-line-mean baseline scores ≈ 0 here by
  construction, so it isolates whether the model learned anything from structure. The
  acceptance gate (Requirement 2.5) is tied to it.
- **Per-compound ρ** (ranking lines for one compound) measures selectivity, which is smaller and
  harder. The cell-line-mean baseline is the real competitor here. It gives every compound the
  same line ranking, and that ranking correlates with the truth to the extent that some lines
  are generally sensitive.
- The cell-line-mean baseline is **weak overall** (R² ≈ 0.04). Beating it is necessary, not impressive.
- **Compound-only ablation** (Requirement 2.6): the same model without cell-line features. It
  predicts potency only, so its per-compound ρ is undefined/constant. If the full model's
  per-compound ρ does not beat the cell-line-mean baseline, the cell-line rankings shown to users
  are just potency plus line average. Say so in the UI and the writeup.

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

**Memory:** ~475k rows × (2,058 compound + G gene) features in float32. With the recommended
G = 978 landmark genes that is ≈ 5.7 GB, which fits in 32 GB. Build the matrix by indexing two small per-entity arrays
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
