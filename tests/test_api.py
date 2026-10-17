"""API contract tests with a fake pipeline (spec 08 testing strategy)."""

import pytest
from fastapi.testclient import TestClient

from oncora.config import DISCLAIMER
from oncora.serve.app import create_app

SUMMARY = {"renderer": "template", "claims": [{"section": "prediction", "text": "t", "cites": ["a"]}],
           "disclaimer": DISCLAIMER, "dropped_claims": 0, "fallback_reason": None}


class FakePipeline:
    model_versions = {"target": "t1", "sensitivity": "s1"}
    disease_map = None

    def __init__(self):
        self.llm_flags = []

    def predict(self, smiles, *, explain_with_llm=True):
        self.llm_flags.append(explain_with_llm)
        if smiles == "bad":
            return {"valid": False, "input_smiles": smiles, "error": "Could not parse SMILES: syntax error",
                    "disclaimer": DISCLAIMER}
        return {
            "valid": True, "input_smiles": smiles, "canonical_smiles": "CCO", "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "targets": [{"target_chembl_id": "CHEMBL203", "pref_name": "EGFR", "gene_symbol": "EGFR",
                         "uniprot": "P00533", "score": 0.9, "rank": 1}],
            "sensitive_cell_lines": [{"model_id": "ACH-000001", "cell_line_name": None, "predicted_percentile": 0.9,
                                      "predicted_dss3": 0.3, "p_sensitive": 0.5, "rank": 1}],
            "cancer_types": [], "cancer_types_note": "needs Model.csv",
            "evidence_summary": SUMMARY, "evidence_bundle": {},
            "near_seen": {"target": {"flag": False, "max_tanimoto": 0.5, "nearest_inchi_key": "X"}},
            "selectivity_learned": False, "model_versions": self.model_versions,
            "timings_ms": {"targets": 1.0}, "disclaimer": DISCLAIMER,
        }


@pytest.fixture
def client():
    fake = FakePipeline()
    with TestClient(create_app(lambda: fake)) as c:
        c.fake = fake
        yield c


def test_predict_valid(client):
    r = client.post("/predict", json={"smiles": "CCO"})
    assert r.status_code == 200
    body = r.json()
    assert body["disclaimer"] == DISCLAIMER and body["valid"] is True
    for key in ("targets", "sensitive_cell_lines", "cancer_types", "evidence_summary", "near_seen",
                "model_versions", "timings_ms"):
        assert key in body
    assert body["cancer_types"] == [] and body["cancer_types_note"]


def test_predict_invalid_is_422_with_reason_and_disclaimer(client):
    r = client.post("/predict", json={"smiles": "bad"})
    assert r.status_code == 422
    assert r.json() == {"valid": False, "input_smiles": "bad", "error": "Could not parse SMILES: syntax error",
                        "disclaimer": DISCLAIMER}


def test_empty_smiles_rejected_by_schema(client):
    assert client.post("/predict", json={"smiles": ""}).status_code == 422


def test_health(client):
    body = client.get("/health").json()
    assert body["model_versions"] == {"target": "t1", "sensitivity": "s1"} and body["disease_map"] is False
    assert body["disclaimer"] == DISCLAIMER


def test_batch_uses_template_only_and_reports_each(client):
    r = client.post("/predict/batch", json={"smiles": ["CCO", "bad"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["valid"] for x in res] == [True, False]
    assert client.fake.llm_flags == [False, False]
    assert client.post("/predict/batch", json={"smiles": ["C"] * 21}).status_code == 422
