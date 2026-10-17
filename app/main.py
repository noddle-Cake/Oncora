"""Oncora demo UI (spec 08, Req 2; FR-17, FR-18).

    uv run streamlit run app/main.py      # expects the API at ONCORA_API_URL (default http://localhost:8000)

The UI only draws: every prediction comes from the API. RDKit is used here solely to render the
molecule.
"""

import json
import os
import urllib.error
import urllib.request

import polars as pl
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

API_URL = os.environ.get("ONCORA_API_URL", "http://localhost:8000")
DISCLAIMER = "These are computational research predictions, not validated clinical conclusions."
EXAMPLES = {
    "Gefitinib (held out of all training: unseen-compound demo)": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Invalid example": "C1CC(",
}


def call_api(smiles: str, use_llm: bool) -> tuple[int, dict]:
    req = urllib.request.Request(f"{API_URL}/predict", method="POST",
                                 data=json.dumps({"smiles": smiles, "explain_with_llm": use_llm}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


st.set_page_config(page_title="Oncora", page_icon="🧬", layout="wide")
st.title("Oncora")
st.caption("Compound → protein targets → sensitive cell lines → cancer types, with cited evidence.")

with st.sidebar:
    choice = st.selectbox("Example compounds", list(EXAMPLES))
    smiles = st.text_area("SMILES", value=EXAMPLES[choice], height=100)
    use_llm = st.toggle("Explain with Claude (falls back to template)", value=True)
    go = st.button("Predict", type="primary", use_container_width=True)

if go:
    with st.spinner("Scoring 1,360 targets and every CCLE cell line…"):
        status, data = call_api(smiles, use_llm)
    st.warning(f"⚠️ {data.get('disclaimer', DISCLAIMER)}")          # FR-18: first on every results view

    if not data.get("valid"):
        st.error(f"Invalid SMILES: {data.get('error', 'unknown error')}")
        st.stop()

    left, right = st.columns([1, 2])
    with left:
        st.image(Draw.MolToImage(Chem.MolFromSmiles(data["canonical_smiles"]), size=(360, 280)))
        st.code(data["canonical_smiles"], language=None)
        st.caption(f"InChIKey `{data['inchi_key']}`")
        ns = data["near_seen"]["target"]
        if ns["flag"]:
            st.error(f"Near-seen: Tanimoto {ns['max_tanimoto']} to training compound `{ns['nearest_inchi_key']}`. "
                     "Treat these predictions as recall, not generalization.")
        else:
            st.success(f"Unseen compound: nearest training compound has Tanimoto {ns['max_tanimoto']}.")
    with right:
        summary = data["evidence_summary"]
        st.subheader("Evidence summary")
        st.caption(f"Renderer: {summary['renderer']}"
                   + (f" (Claude unavailable: {summary['fallback_reason']})" if summary.get("fallback_reason") else ""))
        styles = {"prediction": (st.info, "🔮 Model predictions"),
                  "annotation": (st.success, "📚 Database annotations"),
                  "cross": (st.container, "🔗 How the evidence relates")}
        for key, (box, title) in styles.items():
            claims = [c["text"] for c in summary["claims"] if c["section"] == key]
            if claims:
                (box if box is not st.container else st.markdown)(f"**{title}**\n\n" + "\n".join(f"- {t}" for t in claims))

    t1, t2, t3 = st.tabs(["Targets", "Cell lines", "Cancer types"])
    with t1:
        st.dataframe(pl.DataFrame(data["targets"]).select("rank", "gene_symbol", "pref_name", "target_chembl_id", "score"),
                     hide_index=True, use_container_width=True)
    with t2:
        if data.get("selectivity_learned") is False:
            st.info("In held-out evaluation, the sensitivity model's per-compound cell-line ranking did not beat the "
                    "line-average ranking: these rankings mostly reflect potency plus each line's general sensitivity.")
        st.dataframe(pl.DataFrame(data["sensitive_cell_lines"]), hide_index=True, use_container_width=True)
    with t3:
        if data["cancer_types"]:
            st.dataframe(pl.DataFrame(data["cancer_types"]), hide_index=True, use_container_width=True)
        else:
            st.info(data.get("cancer_types_note") or "No cancer types available.")

    with st.expander("Model versions, timings, raw evidence bundle"):
        st.json({"model_versions": data["model_versions"], "timings_ms": data["timings_ms"]})
        st.json(data["evidence_bundle"])
    st.warning(f"⚠️ {data['disclaimer']}")
