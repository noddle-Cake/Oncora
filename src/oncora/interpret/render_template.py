"""Deterministic template renderer: works offline, with no API key (spec 07, Req 3; NFR-9).

Every sentence is built from bundle fields and cites the paths it used, so the template proves the
same evidence can be explained with no LLM at all.
"""

from oncora.interpret.bundle import EvidenceBundle
from oncora.interpret.summary import Claim, EvidenceSummary

TOP_TARGETS, TOP_LINES, TOP_TYPES = 5, 5, 3


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def render(bundle: EvidenceBundle) -> EvidenceSummary:
    p, a, d = bundle.predictions, bundle.annotations, bundle.derived
    claims: list[Claim] = []

    if p.targets:
        items = p.targets[:TOP_TARGETS]
        named = ", ".join(f"{t.gene_symbol or t.target_chembl_id} ({t.name}; score {_fmt(t.score)})" for t in items)
        claims.append(Claim(section="prediction",
                            text=f"The target model ranks these proteins highest: {named}.",
                            cites=[f"predictions.targets[{i}].score" for i in range(len(items))]))
    ns = p.near_seen.get("target")
    if ns is not None:
        text = (f"This structure is nearly identical to a training compound (Tanimoto {_fmt(ns.max_tanimoto)}), "
                "so these predictions partly reflect memory rather than generalization." if ns.flag else
                f"The closest training compound has Tanimoto similarity {_fmt(ns.max_tanimoto)}, so this is an "
                "unseen-compound prediction.")
        claims.append(Claim(section="prediction", text=text, cites=["predictions.near_seen.target.max_tanimoto"]))
    if p.cell_lines:
        items = p.cell_lines[:TOP_LINES]
        named = ", ".join(f"{c.name or c.model_id} (percentile {_fmt(c.predicted_percentile)})" for c in items)
        claims.append(Claim(section="prediction",
                            text=f"The sensitivity model predicts the strongest response in: {named}.",
                            cites=[f"predictions.cell_lines[{i}].predicted_percentile" for i in range(len(items))]))
        if p.selectivity_learned is False:
            claims.append(Claim(section="prediction",
                                text="In held-out evaluation the sensitivity model did not rank cell lines better "
                                     "than their average sensitivity, so this ordering mostly reflects which lines "
                                     "respond to many compounds.",
                                cites=["predictions.selectivity_learned"]))
    if p.cancer_types:
        items = p.cancer_types[:TOP_TYPES]
        named = ", ".join(f"{c.primary_disease} ({c.n_cell_lines} lines{', low support' if c.low_support else ''})"
                          for c in items)
        claims.append(Claim(section="prediction", text=f"Aggregated by disease, the top cancer types are: {named}.",
                            cites=[f"predictions.cancer_types[{i}].score" for i in range(len(items))]))
    elif p.cancer_types_note:
        claims.append(Claim(section="prediction", text=p.cancer_types_note, cites=["predictions.cancer_types_note"]))

    if a.matched:
        prefix = "This structure" if a.match_level == "exact" else "A compound with the same skeleton (stereochemistry may differ)"
        if a.names or a.moa:
            parts = []
            if a.names:
                parts.append(f"is listed as {', '.join(a.names[:3])}")
            if a.moa:
                parts.append(f"with mechanism of action: {'; '.join(a.moa)}")
            claims.append(Claim(section="annotation", text=f"{prefix} {' '.join(parts)} (LINCS compoundinfo).",
                                cites=["annotations.names", "annotations.moa"]))
        if a.targets:
            claims.append(Claim(section="annotation",
                                text=f"LINCS annotates its targets as: {', '.join(a.targets)}.",
                                cites=["annotations.targets"]))
        if a.chembl_actives:
            items = a.chembl_actives[:5]
            named = ", ".join(f"{c.gene_symbol or c.target_chembl_id} (pChEMBL {_fmt(c.pchembl)})" for c in items)
            claims.append(Claim(section="annotation", text=f"ChEMBL records potent activity on: {named}.",
                                cites=[f"annotations.chembl_actives[{i}].pchembl" for i in range(len(items))]))
    else:
        claims.append(Claim(section="annotation",
                            text="No LINCS or ChEMBL annotation matches this structure; it is novel to these sources.",
                            cites=["annotations.matched"]))

    if d.target_annotation_overlap:
        claims.append(Claim(section="cross",
                            text=f"Predicted targets that databases also report: {', '.join(d.target_annotation_overlap)}.",
                            cites=["derived.target_annotation_overlap"]))
    for i, e in enumerate(d.target_expression):
        claims.append(Claim(section="cross",
                            text=f"{e.gene_symbol} expression in the {e.n_top_lines} top predicted lines averages the "
                                 f"{_fmt(e.mean_percentile_in_top_lines)} percentile of all CCLE lines.",
                            cites=[f"derived.target_expression[{i}].mean_percentile_in_top_lines"]))
    claims.append(Claim(section="cross", text=d.relationship, cites=["derived.relationship"]))
    return EvidenceSummary(renderer="template", claims=claims, disclaimer=bundle.disclaimer)
