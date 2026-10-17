"""Typed API contract (spec 08, Req 1; FR-19, AC-9). The disclaimer is a field on every response."""

from typing import Literal

from pydantic import BaseModel, Field

from oncora.config import DISCLAIMER
from oncora.interpret.summary import EvidenceSummary

MAX_BATCH = 20


class PredictRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)
    explain_with_llm: bool = True


class BatchRequest(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class TargetOut(BaseModel):
    target_chembl_id: str
    pref_name: str | None
    gene_symbol: str | None
    uniprot: str | None
    score: float
    rank: int


class CellLineOut(BaseModel):
    model_id: str
    cell_line_name: str | None
    predicted_percentile: float
    predicted_dss3: float
    p_sensitive: float
    rank: int


class CancerTypeOut(BaseModel):
    primary_disease: str
    lineage: str | None
    score: float
    frac_sensitive: float
    n_cell_lines: int
    low_support: bool
    rank: int


class NearSeenOut(BaseModel):
    flag: bool
    max_tanimoto: float
    nearest_inchi_key: str | None


class PredictResponse(BaseModel):
    valid: Literal[True] = True
    input_smiles: str
    canonical_smiles: str
    inchi_key: str
    targets: list[TargetOut]
    sensitive_cell_lines: list[CellLineOut]
    cancer_types: list[CancerTypeOut]
    cancer_types_note: str | None
    evidence_summary: EvidenceSummary
    evidence_bundle: dict
    near_seen: dict[str, NearSeenOut]
    selectivity_learned: bool | None
    model_versions: dict[str, str]
    timings_ms: dict[str, float]
    disclaimer: str = DISCLAIMER


class ValidationErrorResponse(BaseModel):
    valid: Literal[False] = False
    input_smiles: str
    error: str
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_versions: dict[str, str]
    disease_map: bool
    disclaimer: str = DISCLAIMER


class BatchResponse(BaseModel):
    results: list[PredictResponse | ValidationErrorResponse]
    disclaimer: str = DISCLAIMER
