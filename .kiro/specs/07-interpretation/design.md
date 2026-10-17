# Design Document — 07 Biological Interpretation Layer

## Overview

```
TargetModel.predict ─┐
SensitivityModel ────┼─▶ EvidenceBundle (deterministic, typed, path-addressable)
aggregate() ─────────┤        │
annotations (exact ──┘        ├─▶ TemplateRenderer ─────────────┐
 InChIKey match)              │                                 ├─▶ EvidenceSummary + DISCLAIMER
                              └─▶ ClaudeRenderer ─validate─▶────┘
                                   (optional; fallback to template on any failure)
```

The template renderer exists to prove that the same output can be produced with no LLM. Claude
only improves the prose.

## Components and Interfaces

### `src/oncora/interpret/bundle.py`
```python
class Evidence(BaseModel):          # pydantic, so FastAPI can reuse the schema
    kind: Literal["prediction", "annotation", "derived"]
    source: str                     # e.g. "models/target/<run_id>", "compoundinfo.txt:moa"
    ...

class EvidenceBundle(BaseModel):
    query: QueryInfo                          # input_smiles, canonical_smiles, inchi_key
    predictions: Predictions                  # targets[], cell_lines[], cancer_types[], near_seen{}
    annotations: Annotations                  # compoundinfo{target[], moa[]}, chembl_actives[], matched: bool
    derived: Derived                          # target_annotation_overlap[], target_expression[]
    provenance: dict[str, str]                # model run ids, data hashes
    def get(self, path: str) -> Any           # resolves "predictions.targets[0].score"
    def to_canonical_json(self) -> str        # sorted keys, fixed float precision → deterministic

def build_bundle(parsed, target_preds, cell_preds, cancer_preds, *, annotations_db, ccle) -> EvidenceBundle
```
- Annotation lookup: InChIKey → `compound_bridge` (compoundinfo) and the ChEMBL activities Parquet.
  The `target` field is split on its delimiter, which is verified in spec 01.
- Floats are rounded to 3 significant figures in the bundle, so the numbers Claude may quote are fixed.

### `src/oncora/interpret/render_template.py`
- `render(bundle) -> EvidenceSummary`: Jinja2-free f-string templates. Each sentence carries its `cites: list[path]`.

### `src/oncora/interpret/render_claude.py`
```python
def render(bundle, *, client=None, model=cfg.LLM_MODEL, max_tokens=cfg.LLM_MAX_TOKENS,
           timeout=cfg.LLM_TIMEOUT_S) -> EvidenceSummary
```
- Uses the Anthropic Messages API with a **structured-output / tool schema**
  `{"claims": [{"text": str, "cites": [str]}], "section": "prediction" | "annotation" | "cross"}`.
- The system prompt holds the grounding rules from `steering/llm-grounding.md`. The user content is
  `bundle.to_canonical_json()` only.
- `validate_claims(claims, bundle)`: path existence, and a numeric-literal check (regex extracts
  numbers from the text and each must equal a bundle value at 3 s.f.). Failing claims are dropped.
- Usage logging: `input_tokens`, `output_tokens`, and model are appended to `reports/llm_usage.jsonl`.
- Config: `LLM_ENABLED`, `LLM_MODEL` (default the latest capable Claude model, e.g.
  `claude-opus-5-5`; choose a cheaper model to stretch the budget), `LLM_MAX_TOKENS=800`, `LLM_TIMEOUT_S=20`.

### `src/oncora/interpret/__init__.py`
- `explain(bundle) -> EvidenceSummary`: tries Claude if it is enabled and a key is present,
  otherwise uses the template. Appends `DISCLAIMER` in all cases.

### `EvidenceSummary`
`renderer: "template" | "claude"`, `sections: {prediction, annotation, cross}: list[Claim]`,
`disclaimer: str`, `dropped_claims: int`.

## Error Handling

- Any exception, timeout, or empty validated result from Claude → template, with the reason logged (never shown as an error to the user).
- A bundle with no predictions (impossible for a valid SMILES) → an assertion failure.

## Testing Strategy

- Bundle determinism: build twice and compare the JSON byte for byte.
- `get(path)` resolves every path that the template cites.
- Template output: every claim has at least one cite, sections are separated, disclaimer present; runs with the network disabled.
- `validate_claims`: a fabricated number and a bad path are dropped; valid claims pass.
- The Claude renderer uses a stubbed client (no real API calls in tests). Fallback is tested for a missing key, an exception, and a timeout.
