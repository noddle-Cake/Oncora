# Design Document — 08 Serving & UI

## Architecture

```
Streamlit (app/main.py) ──HTTP JSON──▶ FastAPI (src/oncora/serve/app.py)
                                          │ lifespan: load artifacts once
                                          ▼
                     parse_smiles → featurize → TargetModel.predict
                                              → SensitivityModel.predict → aggregate
                                              → build_bundle → explain
                                          ▼
                                   PredictResponse (+ DISCLAIMER, timings)
```

## Components and Interfaces

### `src/oncora/serve/app.py`
- FastAPI `lifespan` loads `load_target_model()`, `load_sensitivity_model()`, and the annotation
  tables into `app.state`. Run IDs come from env (`ONCORA_TARGET_RUN`, `ONCORA_SENS_RUN`) or default to the latest.
- Endpoints: `POST /predict`, `POST /predict/batch` (STRETCH), `GET /health`.

### `src/oncora/serve/schemas.py` (pydantic)
```python
class PredictRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)
    explain_with_llm: bool = True

class PredictResponse(BaseModel):
    valid: bool
    input_smiles: str
    canonical_smiles: str | None
    inchi_key: str | None
    targets: list[TargetPrediction]
    sensitive_cell_lines: list[CellLinePrediction]
    cancer_types: list[CancerTypePrediction]
    cancer_types_note: str | None
    evidence_summary: EvidenceSummary | None
    near_seen: NearSeen          # {target: {flag, max_tanimoto, nearest}, sensitivity: {...}}
    model_versions: dict[str, str]
    timings_ms: dict[str, float]
    disclaimer: str = DISCLAIMER

class ValidationErrorResponse(BaseModel):
    valid: Literal[False] = False
    input_smiles: str
    error: str
    disclaimer: str = DISCLAIMER
```
`DISCLAIMER` is defined once in `oncora/config.py`. The API, template, and UI all import it.

### `app/main.py` (Streamlit)
- Layout: input and examples in the sidebar; main area holds the **disclaimer banner first**, then
  the molecule image and a near-seen notice, then three tabs or columns (Targets | Cell lines |
  Cancer types), then the evidence summary with "Model prediction" and "Database annotation" blocks styled differently.
- The molecule is rendered with `rdkit.Chem.Draw.MolToImage` from `canonical_smiles`. The UI keeps
  RDKit for drawing only and never predicts locally.
- The API URL comes from `ONCORA_API_URL` (default `http://localhost:8000`).

## Error Handling

- Invalid SMILES → 422 `ValidationErrorResponse`. The UI shows `error` inline.
- A model artifact missing at startup → the service fails to start with the expected path
  (fail-fast beats serving half a system).
- A Claude failure never surfaces as an error. The summary simply has `renderer = "template"`.

## Testing Strategy

- `tests/test_api.py` (FastAPI `TestClient`, small fixture artifacts from specs 05/06):
  - a valid SMILES returns 200 with every field, and `disclaimer` matches `DISCLAIMER`;
  - an invalid SMILES returns 422 with a reason and the disclaimer;
  - `/health` lists the run IDs;
  - a missing disease map gives an empty `cancer_types` and a note;
  - with no API key, the summary renderer is `template`.
- The UI is checked manually against the spec 09 demo compound, plus a screenshot for the writeup.
