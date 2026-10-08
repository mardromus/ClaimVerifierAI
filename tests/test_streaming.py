import json

import pytest
from fastapi.testclient import TestClient

from claimverifier.api import create_app
from claimverifier.data import Document
from claimverifier.labels import LABELS


def _events(response):
    events, current = [], None
    for line in response.iter_lines():
        if line.startswith("event:"):
            current = line[6:].strip()
        elif line.startswith("data:"):
            events.append((current, json.loads(line[5:])))
    return events


class FakeSource:
    name, label, description = "europepmc", "Europe PMC", "fake"

    def __init__(self, docs):
        self.docs, self.calls = docs, 0

    def search(self, claim, limit):
        self.calls += 1
        return self.docs[:limit]

    def describe(self):
        return {"name": self.name, "label": self.label, "description": self.description}


def test_verify_stream_event_order(lite_verifier):
    events = list(lite_verifier.verify_stream("Regular aspirin use reduces the risk of colorectal cancer.", top_k=2))
    names = [e for e, _ in events]
    assert names[:3] == ["stage", "stage", "candidates"]
    assert names.index("result") < names.index("explanation") < names.index("done")
    assert "token" in names
    result = dict(events)["result"]
    assert result["verdict"] in LABELS and result["source"] == "corpus" and len(result["documents"]) == 2
    stages = [(d["stage"], d["status"]) for e, d in events if e == "stage"]
    assert stages == [("retrieval", "start"), ("retrieval", "done"), ("rationale", "start"), ("rationale", "done"),
                      ("nli", "start"), ("nli", "done"), ("explanation", "start"), ("explanation", "done")]
    explanation = dict(events)["explanation"]
    assert "".join(d["text"] for e, d in events if e == "token") == explanation["text"]


def test_rerank_keeps_strongest_evidence(lite_verifier):
    v = lite_verifier
    old = v.cfg.retrieval.rerank_depth
    v.cfg.retrieval.rerank_depth = 6
    try:
        retrieved, scores, timings = v.retrieve_and_score(["Smoking increases the risk of COPD."], top_k=2)
    finally:
        v.cfg.retrieval.rerank_depth = old
    docs = retrieved[0]
    assert [d.rank for d in docs] == [1, 2] and len(scores) == 2
    assert all(d.retrieval_rank >= 1 for d in docs)
    assert max(scores[0]) + 0.1 >= max(scores[1])  # sorted by evidence strength (+ small rank prior)
    assert set(timings) == {"retrieval", "rationale"}


def test_live_source_and_custom_documents(lite_verifier):
    docs = [Document(doc_id=900 + i, title=f"Paper {i}", sentences=[s], meta={"source": "Europe PMC", "url": f"https://x/{i}"})
            for i, s in enumerate(["Unrelated cell biology finding.", "Green tea lowered systolic blood pressure in adults.",
                                   "Coffee intake was recorded."])]
    src = FakeSource(docs)
    events = dict(lite_verifier.verify_stream("Green tea lowers blood pressure.", top_k=2, explain=False, source=src))
    assert src.calls == 1 and len(events["candidates"]["documents"]) == 3
    assert events["result"]["source"] == "europepmc" and events["result"]["candidates_scanned"] == 3
    assert len(events["result"]["documents"]) == 2 and events["result"]["documents"][0]["meta"]["source"] == "Europe PMC"
    assert "explanation" not in events
    with pytest.raises(LookupError):
        list(lite_verifier.verify_stream("x y z", source=FakeSource([])))


@pytest.fixture()
def client(lite_verifier, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setattr("claimverifier.api.WEB_DIST", dist)
    reports = tmp_path / "reports" / "demo_dev"
    reports.mkdir(parents=True)
    (reports / "metrics.json").write_text(json.dumps({
        "num_claims": 2,
        "verdict_classification": {"accuracy": 0.5, "macro_f1": 0.4, "macro_precision": 0.4, "macro_recall": 0.45},
        "evidence_retrieval": {"recall@5": 0.9, "mrr": 0.8},
        "scifact_abstract_level": {"label_only": {"f1": 0.3}}}))
    monkeypatch.setattr(lite_verifier.cfg, "reports_dir", str(tmp_path / "reports"))
    return TestClient(create_app(verifier=lite_verifier))


def test_api_stream_cache_and_errors(client):
    body = {"claim": "Statins reduce cardiovascular events.", "top_k": 2}
    with client.stream("POST", "/api/verify/stream", json=body) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        first = _events(r)
    assert first[-1][0] == "done" and any(e == "result" for e, _ in first)
    with client.stream("POST", "/api/verify/stream", json=body) as r:
        second = _events(r)
    assert second == first  # replayed from the result cache
    full = client.post("/api/verify", json=body).json()
    assert full["verdict"] in LABELS and full["explanation"]
    custom = client.post("/api/verify/stream", json={"claim": "Tea lowers blood pressure.", "source": "custom",
                                                       "documents": [{"title": "T", "abstract": "Tea lowered blood pressure."}]})
    assert ("result" in [e for e, _ in _events(custom)])
    assert client.post("/api/verify/stream", json={"claim": "abc def", "source": "custom"}).status_code == 422
    assert client.post("/api/verify", json={"claim": "abc def", "source": "wikipedia"}).status_code == 422


def test_api_disabled_source(client, lite_verifier, monkeypatch):
    monkeypatch.setattr(lite_verifier.cfg.sources, "enabled", [])
    assert client.post("/api/verify/stream", json={"claim": "abc def", "source": "pubmed"}).status_code == 400


def test_api_info_search_reports(client):
    info = client.get("/api/info").json()
    assert info["stats"]["documents"] == 10 and {s["name"] for s in info["sources"]} >= {"corpus", "custom"}
    hits = client.get("/api/search", params={"q": "coffee parkinson", "k": 3}).json()
    assert hits[0]["doc_id"] == 108 and "snippet" in hits[0]
    reports = client.get("/api/reports").json()
    assert reports[0]["name"] == "demo_dev" and reports[0]["macro_f1"] == 0.4
    assert client.get("/api/reports/demo_dev").json()["metrics"]["num_claims"] == 2
    assert client.get("/api/reports/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/reports/missing").status_code == 404


def test_spa_serving(client):
    assert "id=root" in client.get("/").text
    assert "id=root" in client.get("/evaluation").text  # client-side route
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/api/nope").status_code == 404
    assert client.get("/health").json()["status"] == "ok"  # legacy route still works
