"""Oncora demo UI (spec 08, Req 2; FR-17, FR-18).

    uv run streamlit run app/main.py      # expects the API at ONCORA_API_URL (default http://localhost:8000)

The UI only draws: every prediction comes from the API. RDKit is used here solely to render the
molecule.

Layout: one white card per idea on an off-white page, a row of stat tiles with a single filled
"hero" figure, and **linked entities**. Every target, cell line and compound on the page is a
link that opens a sheet: a side panel holding everything this response says about that entity,
with links onward to the entities it relates to (a target's measured activity, the compounds
whose measured response resembles this one and are annotated to it, …). The sheet sits over the
results instead of replacing them, so reading one entity never costs the place in the list. Each
sheet also links out to the entity's canonical page (ChEMBL, UniProt, DepMap, PubChem).

The response is kept in session state, because opening a sheet reruns the script. Styling lives in
app/style.css (University of Miami green and orange; orange is fill-only, see there).
"""

import html
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

API_URL = os.environ.get("ONCORA_API_URL", "http://localhost:8000")
DISCLAIMER = "These are computational research predictions, not validated clinical conclusions."
EXAMPLES = {
    "Gefitinib (held out of all training: unseen-compound demo)": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Vorinostat (strong measured response)": "ONC(=O)CCCCCCC(=O)Nc1ccccc1",
    "Aspirin (outside the validated domain)": "CC(=O)Oc1ccccc1C(=O)O",
    "Invalid example": "C1CC(",
}
SHOW_TARGETS, SHOW_LINES = 20, 25


def call_api(smiles: str, use_llm: bool) -> tuple[int, dict]:
    req = urllib.request.Request(f"{API_URL}/predict", method="POST",
                                 data=json.dumps({"smiles": smiles, "explain_with_llm": use_llm}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# --- small rendering helpers ------------------------------------------------------------------

def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def pill(text: str, kind: str) -> str:
    return f'<span class="pill pill--{kind}">{esc(text)}</span>'


def tile(label: str, value: str, detail: str = "", tone: str = "default") -> str:
    d = f'<span class="stat-tile__detail">{esc(detail)}</span>' if detail else ""
    return (f'<div class="stat-tile stat-tile--{tone}"><span class="stat-tile__value">{esc(value)}</span>'
            f'<span class="stat-tile__label">{esc(label)}</span>{d}</div>')


def bar(fraction: float, label: str, down: bool = False) -> str:
    pct = max(0.0, min(1.0, fraction)) * 100
    return (f'<div class="barwrap"><div class="bar{" bar--down" if down else ""}"><span style="width:{pct:.1f}%">'
            f'</span></div><span class="bar-label">{esc(label)}</span></div>')


def links(pairs: list[tuple[str, str]]) -> None:
    md('<div class="sheet-links">' + "".join(
        f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(text)} ↗</a>' for text, url in pairs) + "</div>")


# --- linked entities --------------------------------------------------------------------------

class Index:
    """Every entity in one response, keyed the way the chips refer to them."""

    def __init__(self, data: dict):
        self.data = data
        b = data["evidence_bundle"]
        self.ann = b["annotations"]
        self.derived = b["derived"]
        self.sig = self.ann.get("measured_signature")
        self.targets = {t["target_chembl_id"]: t for t in data["targets"]}
        self.by_gene = {t["gene_symbol"]: t for t in data["targets"] if t.get("gene_symbol")}
        self.lines = {c["model_id"]: c for c in data["sensitive_cell_lines"]}
        self.actives = {a["target_chembl_id"]: a for a in self.ann.get("chembl_actives", [])}
        self.neighbors = {n["inchi_key"]: n for n in (self.sig or {}).get("neighbors", [])}
        self.expression = {e["gene_symbol"]: e for e in self.derived.get("target_expression", [])}

    def neighbors_for_gene(self, gene: str | None) -> list[dict]:
        return [n for n in self.neighbors.values() if gene and gene in n.get("targets", [])]

    def target_badges(self, t: dict) -> str:
        out = []
        if t["target_chembl_id"] in self.actives:
            out.append(pill("ChEMBL measured", "annotation"))
        if t.get("gene_symbol") in self.ann.get("targets", []):
            out.append(pill("LINCS annotated", "annotation"))
        if t.get("gene_symbol") in self.derived.get("signature_target_overlap", []):
            out.append(pill("similar response", "cross"))
        return "".join(out)


def chip(label: str, kind: str, key: str, ctx: str) -> None:
    """A link to an entity: opens its sheet. Works from the page and from inside another sheet."""
    if st.button(label, key=f"chip-{ctx}-{kind}-{key}", type="tertiary"):
        st.query_params["sheet"] = f"{kind}:{key}"
        st.rerun()


def line_name(c: dict) -> str:
    return c.get("cell_line_name") or c["model_id"]


def target_label(t: dict) -> str:
    return t.get("gene_symbol") or t.get("target_chembl_id")


# --- sheets -----------------------------------------------------------------------------------

def target_sheet(ix: Index, chembl_id: str) -> None:
    t = ix.targets.get(chembl_id)
    a = ix.actives.get(chembl_id)
    gene = (t or a or {}).get("gene_symbol")
    name = (t or {}).get("pref_name") or (a or {}).get("name") or chembl_id
    md(f'<div class="sheet-kicker">Protein target</div><div class="sheet-title">{esc(gene or chembl_id)}</div>'
       f'<div class="sheet-sub">{esc(name)} · <span class="onc-mono">{esc(chembl_id)}</span></div>')
    c1, c2 = st.columns(2)
    if t:
        c1.markdown(tile("Model score: P(pChEMBL ≥ 6)", f"{t['score']:.3f}", tone="hero"), unsafe_allow_html=True)
        c2.markdown(tile("Rank among 1,360 targets", f"#{t['rank']}"), unsafe_allow_html=True)
    else:
        c1.markdown(tile("Model prediction", "Not in top list", "the model did not rank this target highly"),
                    unsafe_allow_html=True)

    md('<div class="sheet-section">What databases record for this compound</div>')
    if a:
        md(f'<div class="sheet-fact">{pill("ChEMBL", "annotation")} Measured activity on {esc(gene or chembl_id)}: '
           f'<b>pChEMBL {a["pchembl"]}</b> (≥ 6 means ≤ 1 µM).</div>')
    else:
        md(f'<div class="sheet-fact">{pill("ChEMBL", "annotation")} No measured activity ≥ pChEMBL 6 for this compound '
           "on this target (it may simply be untested).</div>")
    if gene and gene in ix.ann.get("targets", []):
        md(f'<div class="sheet-fact">{pill("LINCS", "annotation")} LINCS lists {esc(gene)} as an annotated target.</div>')

    related = ix.neighbors_for_gene(gene)
    md('<div class="sheet-section">Compounds that respond alike and are annotated to it</div>')
    if related:
        for n in related:
            cols = st.columns([3, 2], vertical_alignment="center")
            with cols[0]:
                chip((n["names"] or [n["inchi_key"]])[0], "compound", n["inchi_key"], f"t{chembl_id}")
            cols[1].markdown(bar(max(n["correlation"], 0), f"r {n['correlation']}"), unsafe_allow_html=True)
        md('<div class="onc-muted">Measured L1000 responses, independent of the structure-based prediction. '
           "Agreement is correlational evidence, not proof.</div>")
    else:
        md('<div class="onc-muted">None among the most similar measured responses.</div>')

    if gene in ix.expression:
        e = ix.expression[gene]
        md('<div class="sheet-section">Expression in the top predicted cell lines</div>')
        md(bar(e["mean_percentile_in_top_lines"], f"mean percentile {e['mean_percentile_in_top_lines']} "
               f"across {e['n_top_lines']} lines (CCLE)"))

    md('<div class="sheet-section">Open</div>')
    pairs = [("ChEMBL target", f"https://www.ebi.ac.uk/chembl/explore/target/{chembl_id}")]
    if t and t.get("uniprot"):
        pairs.append(("UniProt", f"https://www.uniprot.org/uniprotkb/{t['uniprot']}/entry"))
    links(pairs)


def line_sheet(ix: Index, model_id: str) -> None:
    c = ix.lines[model_id]
    md(f'<div class="sheet-kicker">Cancer cell line</div><div class="sheet-title">{esc(line_name(c))}</div>'
       f'<div class="sheet-sub"><span class="onc-mono">{esc(model_id)}</span> · rank #{c["rank"]} of every CCLE line scored</div>')
    c1, c2, c3 = st.columns(3)
    c1.markdown(tile("Predicted sensitivity percentile", f"{c['predicted_percentile']:.2f}", tone="hero"),
                unsafe_allow_html=True)
    c2.markdown(tile("P(in this compound's top 10%)", f"{c['p_sensitive']:.2f}"), unsafe_allow_html=True)
    c3.markdown(tile("Predicted DSS3", f"{c['predicted_dss3']:.3f}"), unsafe_allow_html=True)
    if ix.data.get("selectivity_learned") is False:
        md(f'<div class="sheet-fact">{pill("caveat", "warn")} In held-out evaluation, per-compound line rankings did '
           "not beat the line-average ranking; this mostly reflects potency plus the line's general sensitivity.</div>")
    else:
        md(f'<div class="sheet-fact">{pill("how to read", "prediction")} Validated for finding a compound\'s most '
           "responsive lines (top of the list), not for ordering the whole list.</div>")

    md('<div class="sheet-section">Top predicted targets of this compound</div>')
    for t in ix.data["targets"][:3]:
        cols = st.columns([3, 2], vertical_alignment="center")
        with cols[0]:
            chip(target_label(t), "target", t["target_chembl_id"], f"l{model_id}")
        cols[1].markdown(bar(t["score"], f"score {t['score']:.3f}"), unsafe_allow_html=True)
    md('<div class="onc-muted">Independent models: a predicted target and a sensitive line co-occur here; one does '
       "not explain the other.</div>")
    md('<div class="sheet-section">Open</div>')
    links([("DepMap cell line", f"https://depmap.org/portal/cell_line/{model_id}")])


def compound_sheet(ix: Index, inchi_key: str) -> None:
    n = ix.neighbors[inchi_key]
    name = (n["names"] or [inchi_key])[0]
    md(f'<div class="sheet-kicker">Compound with a similar measured response</div>'
       f'<div class="sheet-title">{esc(name)}</div><div class="sheet-sub onc-mono">{esc(inchi_key)}</div>')
    c1, c2 = st.columns(2)
    c1.markdown(tile("Correlation of measured responses", f"r {n['correlation']}", tone="hero"), unsafe_allow_html=True)
    c2.markdown(tile("Also known as", ", ".join(n["names"][1:3]) or "—"), unsafe_allow_html=True)
    md('<div class="sheet-section">Mechanism of action (LINCS)</div>')
    md("".join(pill(m, "annotation") for m in n["moa"]) or '<span class="onc-muted">No MoA annotated.</span>')
    md('<div class="sheet-section">Annotated targets</div>')
    if n["targets"]:
        for g in n["targets"]:
            t = ix.by_gene.get(g)
            cols = st.columns([3, 2], vertical_alignment="center")
            if t:
                with cols[0]:
                    chip(g, "target", t["target_chembl_id"], f"c{inchi_key}")
                cols[1].markdown(pill(f"also predicted for the query · #{t['rank']}", "cross"), unsafe_allow_html=True)
            else:
                cols[0].markdown(f"**{esc(g)}**")
                cols[1].markdown('<span class="onc-muted">not in the query\'s top predictions</span>',
                                 unsafe_allow_html=True)
    else:
        md('<span class="onc-muted">No targets annotated.</span>')
    md('<div class="onc-muted">Similar measured responses suggest, but do not prove, a shared mechanism.</div>')
    md('<div class="sheet-section">Open</div>')
    links([("PubChem", f"https://pubchem.ncbi.nlm.nih.gov/#query={inchi_key}")])


SHEETS = {"target": target_sheet, "line": line_sheet, "compound": compound_sheet}


def _close_sheet() -> None:
    st.query_params.pop("sheet", None)


def open_sheet(ix: Index) -> None:
    """Open the sheet named in the URL (?sheet=kind:key).

    The open sheet lives in the URL, not only in session state, so a sheet has an address: a link to
    "gefitinib, the EGFR sheet" can be pasted into a message. A chip sets the parameter and reruns,
    which is also how one sheet links to the next; dismissing a sheet clears it.
    """
    kind, _, key = st.query_params.get("sheet", "").partition(":")
    valid = {"target": key in ix.targets or key in ix.actives, "line": key in ix.lines,
             "compound": key in ix.neighbors}
    if not valid.get(kind):
        return
    # A full-height drawer from the right, beside the list it was opened from; Esc or a click
    # outside closes it. Streamlit's own drawer, so no layout is fought with CSS.
    st.dialog(" ", width="medium", position="right", on_dismiss=_close_sheet)(SHEETS[kind])(ix, key)


# --- page -------------------------------------------------------------------------------------

st.set_page_config(page_title="Oncora", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")
md(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>")
md('<div class="onc-header"><span class="onc-header__name">Oncora</span>'
   '<span class="onc-header__tag">Compound → protein targets → sensitive cancer cell lines, with cited, '
   "linked evidence</span></div>")

def predict(smiles: str, use_llm: bool) -> None:
    with st.spinner("Scoring 1,360 targets and every CCLE cell line…"):
        st.session_state.result = call_api(smiles, use_llm)
    st.session_state.result_smiles = smiles


# A shared link (?smiles=…) loads its prediction on arrival; it must land before the input renders.
linked = st.query_params.get("smiles")
if linked and st.session_state.get("result_smiles") != linked:
    st.session_state.smiles = linked
    predict(linked, st.session_state.get("use_llm", True))

with st.container(border=True):
    def _use_example():
        st.session_state.smiles = EXAMPLES[st.session_state.example]

    c1, c2 = st.columns([2, 5])
    c1.selectbox("Example compounds", list(EXAMPLES), key="example", on_change=_use_example)
    st.session_state.setdefault("smiles", EXAMPLES[next(iter(EXAMPLES))])
    c2.text_input("SMILES", key="smiles", placeholder="Paste a molecule as SMILES")
    c3, c4, _ = st.columns([2, 3, 4], vertical_alignment="center")
    go = c3.button("Predict", type="primary", width="stretch")
    use_llm = c4.toggle("Explain with Claude (falls back to template)", value=True, key="use_llm")

if go:
    predict(st.session_state.smiles, use_llm)
    st.query_params.pop("sheet", None)
    st.query_params["smiles"] = st.session_state.smiles     # the result now has a shareable address

if "result" not in st.session_state:
    md('<div class="onc-muted">Pick an example or paste a SMILES string, then press Predict. Every target, cell '
       "line and compound in the results opens a linked sheet.</div>")
    st.stop()

status, data = st.session_state.result
md(f'<div class="onc-disclaimer">⚠️ {esc(data.get("disclaimer", DISCLAIMER))}</div>')   # FR-18: first on every results view
if not data.get("valid"):
    st.error(f"Invalid SMILES: {data.get('error', 'unknown error')}")
    st.stop()

ix = Index(data)
b = data["evidence_bundle"]
ns = data["near_seen"]["target"]
ap = (b.get("predictions") or {}).get("applicability")

# Stat tiles: one hero.
top_t = data["targets"][0] if data["targets"] else None
top_l = data["sensitive_cell_lines"][0] if data["sensitive_cell_lines"] else None
cols = st.columns(4)
cols[0].markdown(tile("Top predicted target", target_label(top_t) if top_t else "—",
                      f"score {top_t['score']:.3f}" if top_t else "", tone="hero"), unsafe_allow_html=True)
cols[1].markdown(tile("Most sensitive cell line (predicted)", line_name(top_l) if top_l else "—",
                      f"percentile {top_l['predicted_percentile']:.2f}" if top_l else ""), unsafe_allow_html=True)
cols[2].markdown(tile("Nearest training compound (Tanimoto)", f"{ns['max_tanimoto']:.2f}",
                      "near-seen: recall, not generalization" if ns["flag"] else "unseen compound",
                      tone="warning" if ns["flag"] else "default"), unsafe_allow_html=True)
sig = ix.sig
cols[3].markdown(tile("Measured response (L1000)",
                      (f"r {sig['reproducibility']}" if sig.get("reproducibility") is not None else "1 cell line")
                      if sig else "Not profiled",
                      (f"{sig['n_cell_lines']} cell lines · {'reproducible' if sig['reproducible'] else 'inconclusive'}"
                       if sig else "no lab measurement for this structure")), unsafe_allow_html=True)
st.write("")

left, right = st.columns([2, 3])
with left:
    with st.container(border=True):
        md('<div class="onc-card-title">Query compound</div>')
        st.image(Draw.MolToImage(Chem.MolFromSmiles(data["canonical_smiles"]), size=(420, 300)),
                 width="stretch")
        md(f'<div class="onc-mono">{esc(data["canonical_smiles"])}</div>')
        md(f'<div class="onc-muted">InChIKey <span class="onc-mono">{esc(data["inchi_key"])}</span></div>')
        flags = [pill("near-seen", "bad") if ns["flag"] else pill("unseen compound", "good")]
        if ap and not ap["in_domain"]:
            flags.append(pill("outside validated domain", "warn"))
        if ix.ann.get("matched"):
            flags.append(pill(f"database match: {ix.ann['match_level']}", "annotation"))
        md("".join(flags))
        if ap and not ap["in_domain"]:
            md(f'<div class="onc-muted">{esc(ap["note"])}</div>')
with right:
    with st.container(border=True):
        summary = data["evidence_summary"]
        md('<div class="onc-card-title">Evidence summary</div>')
        md(f'<div class="onc-muted">Renderer: {esc(summary["renderer"])}'
           + (f' (Claude unavailable: {esc(summary["fallback_reason"])})' if summary.get("fallback_reason") else "")
           + "</div>")
        for key, title, kind in (("prediction", "Model predictions", "prediction"),
                                 ("annotation", "Database annotations", "annotation"),
                                 ("cross", "How the evidence relates", "cross")):
            claims = [c["text"] for c in summary["claims"] if c["section"] == key]
            if claims:
                md(f"{pill(title, kind)}<ul class='claims claims--{key}'>"
                   + "".join(f"<li>{esc(t)}</li>" for t in claims) + "</ul>")

st.write("")
tabs = st.tabs(["Targets", "Cell lines", "Measured response", "Database annotations", "Cancer types", "Details"])

with tabs[0], st.container(border=True):
    md('<div class="onc-muted">Calibrated probability that the compound is active (pChEMBL ≥ 6) on each target. '
       "Select a target for its linked evidence.</div>")
    for t in data["targets"][:SHOW_TARGETS]:
        c = st.columns([0.5, 2.2, 2.6, 2.2, 3], vertical_alignment="center")
        c[0].markdown(f'<span class="rank">{t["rank"]}</span>', unsafe_allow_html=True)
        with c[1]:
            chip(target_label(t), "target", t["target_chembl_id"], "list")
        c[2].markdown(f'<span class="chip-sub">{esc(t.get("pref_name"))}</span>', unsafe_allow_html=True)
        c[3].markdown(bar(t["score"], f"{t['score']:.3f}"), unsafe_allow_html=True)
        c[4].markdown(ix.target_badges(t), unsafe_allow_html=True)

with tabs[1], st.container(border=True):
    if data.get("selectivity_learned") is False:
        md(f'{pill("caveat", "warn")} <span class="onc-muted">Per-compound line rankings did not beat the '
           "line-average ranking in held-out evaluation; read them as potency plus general sensitivity.</span>")
    else:
        md('<div class="onc-muted">Predicted sensitivity percentile per line. Validated for finding a compound\'s most '
           "responsive lines (the top of this list), not for ordering the whole list.</div>")
    for c in data["sensitive_cell_lines"][:SHOW_LINES]:
        r = st.columns([0.5, 2.4, 3, 2.2], vertical_alignment="center")
        r[0].markdown(f'<span class="rank rank--quiet">{c["rank"]}</span>', unsafe_allow_html=True)
        with r[1]:
            chip(line_name(c), "line", c["model_id"], "list")
        r[2].markdown(bar(c["predicted_percentile"], f"percentile {c['predicted_percentile']:.2f}"),
                      unsafe_allow_html=True)
        r[3].markdown(f'<span class="chip-sub">P(top 10%) {c["p_sensitive"]:.2f}</span>', unsafe_allow_html=True)

with tabs[2], st.container(border=True):
    if not sig:
        md('<div class="onc-muted">LINCS never profiled this structure, so there is no measured expression response. '
           "This is expected for a new compound: the models never use measured responses as input.</div>")
    else:
        md(f'{pill("measured, not predicted", "annotation")} <span class="onc-muted">LINCS L1000: {sig["n_profiles"]} '
           f'profiles in {sig["n_cell_lines"]} cell lines; reproducibility '
           f'{sig["reproducibility"] if sig["reproducibility"] is not None else "unknown"}'
           f'{"" if sig["reproducible"] else " (inconclusive)"}.</span>')
        g1, g2 = st.columns(2)
        zmax = max([abs(g["z"]) for g in sig["up"] + sig["down"]] or [1])
        with g1:
            md('<div class="sheet-section">Up-regulated genes</div>')
            for g in sig["up"]:
                md(f'<div class="sheet-fact"><b>{esc(g["gene_symbol"])}</b></div>' + bar(g["z"] / zmax, f"z {g['z']}"))
        with g2:
            md('<div class="sheet-section">Down-regulated genes</div>')
            for g in sig["down"]:
                md(f'<div class="sheet-fact"><b>{esc(g["gene_symbol"])}</b></div>'
                   + bar(abs(g["z"]) / zmax, f"z {g['z']}", down=True))
        md('<div class="sheet-section">Responds like</div>')
        if sig["neighbors"]:
            for n in sig["neighbors"]:
                r = st.columns([2.4, 3.6, 2], vertical_alignment="center")
                with r[0]:
                    chip((n["names"] or [n["inchi_key"]])[0], "compound", n["inchi_key"], "list")
                r[1].markdown("".join(pill(m, "annotation") for m in n["moa"][:3])
                              or '<span class="onc-muted">no MoA annotated</span>', unsafe_allow_html=True)
                r[2].markdown(bar(max(n["correlation"], 0), f"r {n['correlation']}"), unsafe_allow_html=True)
        else:
            md('<div class="onc-muted">Neighbors are only shown for a reproducible response.</div>')

with tabs[3], st.container(border=True):
    if not ix.ann.get("matched"):
        md('<div class="onc-muted">No LINCS or ChEMBL record matches this structure: it is novel to these sources.</div>')
    else:
        md(f'{pill("database annotation", "annotation")} <span class="onc-muted">Match level: '
           f'{esc(ix.ann["match_level"])}. These are records, not predictions.</span>')
        if ix.ann.get("names"):
            md(f'<div class="sheet-fact"><b>Names:</b> {esc(", ".join(ix.ann["names"][:6]))}</div>')
        if ix.ann.get("moa"):
            md('<div class="sheet-fact"><b>Mechanism of action:</b> '
               + "".join(pill(m, "annotation") for m in ix.ann["moa"]) + "</div>")
        if ix.ann.get("targets"):
            md(f'<div class="sheet-fact"><b>LINCS targets:</b> {esc(", ".join(ix.ann["targets"]))}</div>')
        if ix.ann.get("chembl_actives"):
            md('<div class="sheet-section">ChEMBL measured actives (pChEMBL ≥ 6)</div>')
            for a in ix.ann["chembl_actives"]:
                r = st.columns([2.4, 3.6, 2], vertical_alignment="center")
                with r[0]:
                    chip(a.get("gene_symbol") or a["target_chembl_id"], "target", a["target_chembl_id"], "ann")
                r[1].markdown(f'<span class="chip-sub">{esc(a.get("name"))}</span>', unsafe_allow_html=True)
                r[2].markdown(f'<span class="bar-label">pChEMBL {a["pchembl"]}</span>', unsafe_allow_html=True)

with tabs[4], st.container(border=True):
    if data["cancer_types"]:
        for k in data["cancer_types"]:
            md(f'<div class="sheet-fact"><span class="rank rank--quiet">{k["rank"]}</span> <b>{esc(k["primary_disease"])}</b> '
               f'<span class="chip-sub">{k["n_cell_lines"]} lines{" · low support" if k["low_support"] else ""}</span></div>')
    else:
        md(f'<div class="onc-muted">{esc(data.get("cancer_types_note") or "No cancer types available.")}</div>')

with tabs[5], st.container(border=True):
    st.json({"model_versions": data["model_versions"], "timings_ms": data["timings_ms"]})
    with st.expander("Raw evidence bundle"):
        st.json(b)

md(f'<div class="onc-disclaimer">⚠️ {esc(data["disclaimer"])}</div>')
open_sheet(ix)
