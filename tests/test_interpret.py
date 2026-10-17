import json
from types import SimpleNamespace

import polars as pl
import pytest

from oncora.chem.similarity import NearestNeighbor
from oncora.interpret import explain, render_claude, render_template
from oncora.interpret.annotations import AnnotationIndex
from oncora.interpret.builder import build_bundle
from oncora.interpret.bundle import EvidenceBundle

IK = "XGALLCVXEZPNRQ-UHFFFAOYSA-N"


def _index():
    lincs = pl.DataFrame({"inchi_key": [IK], "connectivity": [IK[:14]], "names": [["gefitinib"]],
                          "moa": [["EGFR inhibitor"]], "targets": [["EGFR"]]})
    chembl = pl.DataFrame({"inchi_key": [IK, IK], "connectivity": [IK[:14]] * 2,
                           "target_chembl_id": ["CHEMBL203", "CHEMBL3009"], "target_name": ["EGFR", "ERBB4"],
                           "pchembl_median": [7.76, 6.59], "target_gene_symbol": ["EGFR", "ERBB4"]})
    return AnnotationIndex(lincs, chembl)


def _bundle(inchi_key=IK) -> EvidenceBundle:
    targets = [SimpleNamespace(rank=1, target_chembl_id="CHEMBL203", pref_name="Epidermal growth factor receptor",
                               gene_symbol="EGFR", score=0.98765),
               SimpleNamespace(rank=2, target_chembl_id="CHEMBL2363", pref_name="Ephrin A2", gene_symbol="EPHA2",
                               score=0.91234)]
    lines = [SimpleNamespace(rank=1, model_id="ACH-000001", cell_line_name=None, predicted_percentile=0.97321,
                             p_sensitive=0.61)]
    return build_bundle(input_smiles="smi", canonical_smiles="smi", inchi_key=inchi_key, targets=targets,
                        cell_lines=lines, cancer_types=[], cancer_types_note="Cancer-type ranking needs DepMap Model.csv.",
                        near_seen={"target": NearestNeighbor(0.9193, "OTHER", False)},
                        model_runs={"target": "r1", "sensitivity": "r2"}, selectivity_learned=False,
                        annotations=_index())


def test_bundle_is_deterministic_and_path_addressable():
    a, b = _bundle(), _bundle()
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.get("predictions.targets[0].score") == 0.988          # rounded to 3 s.f.
    assert a.get("annotations.chembl_actives[1].gene_symbol") == "ERBB4"
    assert a.derived.target_annotation_overlap == ["EGFR"]
    assert a.annotations.match_level == "exact"
    with pytest.raises(KeyError):
        a.get("predictions.targets[9].score")


def test_unmatched_compound_says_so():
    b = _bundle(inchi_key="ZZZZZZZZZZZZZZ-UHFFFAOYSA-N")
    assert not b.annotations.matched and b.annotations.match_level == "none"
    s = render_template.render(b)
    assert any("novel to these sources" in c.text for c in s.section("annotation"))


def test_connectivity_match_is_labeled():
    b = _bundle(inchi_key=IK[:14] + "-ABCDEFGHIJ-N")
    assert b.annotations.matched and b.annotations.match_level == "connectivity"


def test_template_cites_every_claim_and_separates_sections():
    b = _bundle()
    s = render_template.render(b)
    assert s.renderer == "template" and s.disclaimer == b.disclaimer
    for c in s.claims:
        assert c.cites and all(b.has(p) for p in c.cites), c
    assert s.section("prediction") and s.section("annotation") and s.section("cross")
    md = s.to_markdown()
    assert "Model predictions" in md and "Database annotations" in md and b.disclaimer in md
    assert "correlational" in md


def test_validate_claims_drops_fabrications():
    b = _bundle()
    raw = [
        {"section": "prediction", "text": "EGFR scores 0.988.", "cites": ["predictions.targets[0].score"]},
        {"section": "prediction", "text": "EGFR scores 0.75.", "cites": ["predictions.targets[0].score"]},   # bad number
        {"section": "annotation", "text": "It is an EGFR inhibitor.", "cites": ["annotations.nonexistent"]},  # bad path
        {"section": "cross", "text": "No cites."},                                                            # malformed
    ]
    kept, dropped = render_claude.validate_claims(raw, b)
    assert [c.text for c in kept] == ["EGFR scores 0.988."] and dropped == 3


class _StubClient:
    def __init__(self, payload=None, exc=None, stop_reason="end_turn"):
        self.payload, self.exc, self.stop_reason, self.calls = payload, exc, stop_reason, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def with_options(self, **kw):
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        text = json.dumps(self.payload)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=self.stop_reason,
                               model=kw["model"], usage=SimpleNamespace(input_tokens=100, output_tokens=50))


def test_claude_renderer_sends_only_the_bundle_and_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(render_claude, "LLM_USAGE_LOG", tmp_path / "usage.jsonl")
    b = _bundle()
    stub = _StubClient({"claims": [{"section": "prediction", "text": "EGFR ranks first (0.988).",
                                    "cites": ["predictions.targets[0].score"]}]})
    s = explain(b, client=stub)
    assert s.renderer == "claude" and len(s.claims) == 1 and s.disclaimer == b.disclaimer
    call = stub.calls[0]
    assert call["messages"][0]["content"].endswith(b.to_canonical_json())
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"
    assert "tool_choice" not in call
    assert (tmp_path / "usage.jsonl").read_text().count("\n") == 1


@pytest.mark.parametrize("stub", [
    _StubClient(exc=__import__("anthropic").APIConnectionError(request=None)),
    _StubClient({"claims": []}, stop_reason="refusal"),
    _StubClient({"claims": [{"section": "cross", "text": "Made up 42.", "cites": ["x.y"]}]}),
])
def test_claude_failures_fall_back_to_template(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(render_claude, "LLM_USAGE_LOG", tmp_path / "usage.jsonl")
    s = explain(_bundle(), client=stub)
    assert s.renderer == "template" and s.fallback_reason


def test_no_key_uses_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert explain(_bundle()).renderer == "template"


@pytest.mark.parametrize("heavy, flagged", [(13, True), (37, False)])
def test_applicability_domain_warns_for_small_molecules(heavy, flagged):
    from oncora.interpret.builder import build_bundle
    b = build_bundle(input_smiles="s", canonical_smiles="s", inchi_key=IK, targets=[], cell_lines=[],
                     cancer_types=[], cancer_types_note=None, near_seen={}, model_runs={},
                     selectivity_learned=None, annotations=_index(), heavy_atoms=heavy)
    ap = b.predictions.applicability
    assert ap.heavy_atoms == heavy and ap.in_domain is (not flagged)
    texts = [c.text for c in render_template.render(b).section("prediction")]
    assert any("unreliable" in t for t in texts) is flagged
