"""The evidence bundle: one deterministic, path-addressable object per query (spec 07, Req 1–2).

Every explanation, whether template or LLM, is generated *from* this object and checked *against*
it. Each item says what kind of evidence it is (`prediction` from a model, `annotation` from a
database, `derived` cross-evidence) and where it came from. Floats are rounded to 3 significant
figures, so serialization is byte-stable and any number in a rendered sentence can be matched
exactly against the bundle.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from oncora.config import DISCLAIMER


def r3(x: float | None) -> float | None:
    return None if x is None else float(f"{x:.3g}")


class QueryInfo(BaseModel):
    input_smiles: str
    canonical_smiles: str
    inchi_key: str


class TargetEvidence(BaseModel):
    kind: Literal["prediction"] = "prediction"
    rank: int
    target_chembl_id: str
    name: str | None
    gene_symbol: str | None
    score: float = Field(description="calibrated probability of pChEMBL >= 6")


class CellLineEvidence(BaseModel):
    kind: Literal["prediction"] = "prediction"
    rank: int
    model_id: str
    name: str | None
    predicted_percentile: float
    p_sensitive: float


class CancerTypeEvidence(BaseModel):
    kind: Literal["prediction"] = "prediction"
    rank: int
    primary_disease: str
    lineage: str | None
    score: float
    n_cell_lines: int
    low_support: bool


class NearSeen(BaseModel):
    flag: bool
    max_tanimoto: float
    nearest_inchi_key: str | None


class Applicability(BaseModel):
    kind: Literal["derived"] = "derived"
    heavy_atoms: int
    in_domain: bool = Field(description="False when the molecule is too small for reliable target ranking")
    note: str


class Predictions(BaseModel):
    targets: list[TargetEvidence]
    cell_lines: list[CellLineEvidence]
    cancer_types: list[CancerTypeEvidence]
    cancer_types_note: str | None = None
    near_seen: dict[str, NearSeen]
    model_runs: dict[str, str]
    applicability: Applicability | None = None
    selectivity_learned: bool | None = Field(
        None, description="spec 06 verdict: do per-compound cell-line rankings beat the line-average ranking?")


class ChemblActive(BaseModel):
    kind: Literal["annotation"] = "annotation"
    target_chembl_id: str
    name: str | None
    gene_symbol: str | None
    pchembl: float


class GeneResponse(BaseModel):
    gene_symbol: str
    z: float = Field(description="consensus z-score of the landmark gene (positive = up-regulated)")


class SignatureNeighbor(BaseModel):
    inchi_key: str
    names: list[str]
    moa: list[str]
    targets: list[str]
    correlation: float = Field(description="Pearson r between the two measured consensus signatures")


class MeasuredSignature(BaseModel):
    """What LINCS L1000 *measured* when cells were treated with this compound (D-09: never a prediction)."""
    kind: Literal["annotation"] = "annotation"
    source: str = "LINCS L1000 Level 4 trt_cp, 978 landmark genes, consensus over cell lines"
    match_level: Literal["exact", "connectivity"]
    n_profiles: int
    n_cell_lines: int
    reproducibility: float | None = Field(
        description="correlation between signatures from two disjoint halves of the cell lines; null with one line")
    reproducible: bool
    up: list[GeneResponse]
    down: list[GeneResponse]
    neighbors: list[SignatureNeighbor] = Field(
        default_factory=list, description="other profiled compounds whose measured response is most similar")


class Annotations(BaseModel):
    matched: bool
    match_level: Literal["exact", "connectivity", "none"]
    source_compoundinfo: str = "LINCS compoundinfo.txt (cmap_name, target, moa)"
    source_chembl: str = "ChEMBL 37 activities (median pChEMBL >= 6, single human protein)"
    names: list[str] = []
    moa: list[str] = []
    targets: list[str] = []
    chembl_actives: list[ChemblActive] = []
    measured_signature: MeasuredSignature | None = None


class TargetExpression(BaseModel):
    kind: Literal["derived"] = "derived"
    gene_symbol: str
    mean_percentile_in_top_lines: float = Field(description="mean expression percentile (0-1) of the gene in the top predicted lines")
    n_top_lines: int


class Derived(BaseModel):
    target_annotation_overlap: list[str] = Field(
        default_factory=list, description="predicted target genes that also appear in database annotations")
    target_expression: list[TargetExpression] = []
    signature_target_overlap: list[str] = Field(
        default_factory=list,
        description="predicted target genes that are annotated targets of compounds with a similar measured response")
    relationship: str = ("The target and sensitivity predictions come from independently trained models; "
                         "their co-occurrence is correlational evidence, not proof.")


class EvidenceBundle(BaseModel):
    query: QueryInfo
    predictions: Predictions
    annotations: Annotations
    derived: Derived
    disclaimer: str = DISCLAIMER

    def to_canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def get(self, path: str) -> Any:
        """Resolve a path like 'predictions.targets[0].score'; raises KeyError if absent."""
        node: Any = self.model_dump(mode="json")
        for name, index in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", path):
            try:
                node = node[name] if name else node[int(index)]
            except (KeyError, IndexError, TypeError):
                raise KeyError(path) from None
        return node

    def has(self, path: str) -> bool:
        try:
            self.get(path)
            return True
        except KeyError:
            return False
