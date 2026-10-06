import pytest
from fastapi.testclient import TestClient

from claimverifier.api import create_app
from claimverifier.labels import LABELS


@pytest.fixture(scope="module")
def client(lite_verifier):
    return TestClient(create_app(verifier=lite_verifier))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 10
    assert body["components"]["nli"]["method"] == "lite"


def test_verify(client):
    r = client.post("/verify", json={"claim": "Smoking increases the risk of COPD.", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in LABELS and len(body["documents"]) == 2
    assert body["documents"][0]["doc_id"] == 104 and body["explanation"]


def test_verify_validation(client):
    assert client.post("/verify", json={"claim": "x"}).status_code == 422
    assert client.post("/verify", json={"claim": "valid claim here", "top_k": 99}).status_code == 422


def test_verify_batch(client):
    r = client.post("/verify/batch", json={"claims": ["Coffee lowers Parkinson risk.", "Statins prevent stroke."]})
    assert r.status_code == 200 and len(r.json()) == 2


def test_verify_custom(client):
    r = client.post("/verify/custom", json={
        "claim": "Green tea lowers blood pressure.",
        "documents": [{"title": "Tea trial", "abstract": "We randomised adults to green tea. "
                                                          "Green tea lowered systolic blood pressure."}]})
    assert r.status_code == 200
    doc = r.json()["documents"][0]
    assert doc["title"] == "Tea trial" and len(doc["sentences"]) == 2
    bad = client.post("/verify/custom", json={"claim": "abc def", "documents": [{"abstract": "  "}]})
    assert bad.status_code == 422


def test_documents_and_examples(client):
    assert client.get("/documents/101").json()["title"].startswith("Aspirin")
    assert client.get("/documents/999").status_code == 404
    examples = client.get("/examples", params={"n": 3, "seed": 1}).json()
    assert len(examples) == 3 and all(e["label"] in LABELS for e in examples)
