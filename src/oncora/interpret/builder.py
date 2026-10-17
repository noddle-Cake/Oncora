"""Assemble an EvidenceBundle from model outputs and database annotations (spec 07, Req 1–2)."""

from oncora.chem.similarity import NearestNeighbor
from oncora.config import EVIDENCE_TOP_LINES, EVIDENCE_TOP_TARGETS, MIN_RELIABLE_HEAVY_ATOMS
from oncora.interpret.annotations import AnnotationIndex, ExpressionLookup, target_annotation_overlap
from oncora.interpret.bundle import (
    Applicability,
    CancerTypeEvidence,
    CellLineEvidence,
    Derived,
    EvidenceBundle,
    NearSeen,
    Predictions,
    QueryInfo,
    TargetEvidence,
    r3,
)

EXPRESSION_TARGETS = 3   # top predicted targets checked for expression in the top predicted lines


def _near(nn: NearestNeighbor | None) -> NearSeen | None:
    return None if nn is None else NearSeen(flag=nn.near_seen, max_tanimoto=r3(nn.max_tanimoto),
                                            nearest_inchi_key=nn.inchi_key)


def build_bundle(*, input_smiles: str, canonical_smiles: str, inchi_key: str,
                 targets: list, cell_lines: list, cancer_types: list, cancer_types_note: str | None,
                 near_seen: dict[str, NearestNeighbor], model_runs: dict[str, str],
                 selectivity_learned: bool | None, annotations: AnnotationIndex,
                 expression: ExpressionLookup | None = None, heavy_atoms: int | None = None) -> EvidenceBundle:
    t_ev = [TargetEvidence(rank=t.rank, target_chembl_id=t.target_chembl_id, name=t.pref_name,
                           gene_symbol=t.gene_symbol, score=r3(t.score)) for t in targets[:EVIDENCE_TOP_TARGETS]]
    c_ev = [CellLineEvidence(rank=c.rank, model_id=c.model_id, name=c.cell_line_name,
                             predicted_percentile=r3(c.predicted_percentile), p_sensitive=r3(c.p_sensitive))
            for c in cell_lines[:EVIDENCE_TOP_LINES]]
    k_ev = [CancerTypeEvidence(rank=k.rank, primary_disease=k.primary_disease, lineage=k.lineage, score=r3(k.score),
                               n_cell_lines=k.n_cell_lines, low_support=k.low_support) for k in cancer_types]
    preds = Predictions(targets=t_ev, cell_lines=c_ev, cancer_types=k_ev, cancer_types_note=cancer_types_note,
                        near_seen={k: v for k, v in ((k, _near(n)) for k, n in near_seen.items()) if v is not None},
                        model_runs=model_runs, selectivity_learned=selectivity_learned)
    if heavy_atoms is not None:
        in_domain = heavy_atoms >= MIN_RELIABLE_HEAVY_ATOMS
        preds.applicability = Applicability(
            heavy_atoms=heavy_atoms, in_domain=in_domain,
            note=("Molecule size is within the range where target ranking was validated."
                  if in_domain else
                  f"This molecule has only {heavy_atoms} heavy atoms. Below "
                  f"{MIN_RELIABLE_HEAVY_ATOMS}, fingerprint features are too sparse to place targets "
                  "reliably: known targets of small compounds such as aspirin and paracetamol rank in "
                  "the hundreds. Treat the target ranking as unreliable for this input."))
    ann = annotations.lookup(inchi_key)
    derived = Derived(target_annotation_overlap=target_annotation_overlap([t.gene_symbol for t in t_ev], ann))
    if expression is not None and c_ev:
        genes = [t.gene_symbol for t in t_ev[:EXPRESSION_TARGETS] if t.gene_symbol]
        derived.target_expression = expression.target_expression(genes, [c.model_id for c in c_ev])
    return EvidenceBundle(query=QueryInfo(input_smiles=input_smiles, canonical_smiles=canonical_smiles,
                                          inchi_key=inchi_key),
                          predictions=preds, annotations=ann, derived=derived)
