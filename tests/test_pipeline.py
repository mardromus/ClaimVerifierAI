import json

from claimverifier.data import Document, load_claims
from claimverifier.evaluation.calibrate import calibrate, load_calibration, save_calibration
from claimverifier.evaluation.evaluate import evaluate, markdown_report, save_report
from claimverifier.labels import LABELS
from claimverifier.rationale import select_sentences


def test_verify_structure(lite_verifier):
    r = lite_verifier.verify("Regular aspirin use reduces the risk of colorectal cancer.", top_k=3)
    assert r.verdict in LABELS
    assert abs(sum(r.scores.values()) - 1.0) < 1e-6
    assert 0.0 <= r.confidence <= 1.0 and r.confidence == max(r.scores.values())
    assert len(r.documents) == 3 and r.documents[0].doc_id == 101
    d = r.documents[0]
    assert len(d.sentence_scores) == len(d.sentences) and d.evidence
    assert all(0 <= e.index < len(d.sentences) for e in d.evidence)
    assert r.explanation and r.explanation_backend == "template"
    assert set(r.timings_ms) >= {"retrieval", "rationale", "nli", "explanation"}
    json.dumps(r.to_dict())  # JSON-serialisable


def test_verify_batch_without_explanation(lite_verifier):
    results = lite_verifier.verify_batch(["Statins reduce cardiovascular events.", "Coffee lowers Parkinson risk."])
    assert len(results) == 2 and all(r.explanation == "" for r in results)
    assert results[0].documents[0].doc_id == 106


def test_verify_against_custom_documents(lite_verifier):
    doc = Document(doc_id=-1, title="My abstract", sentences=[
        "We studied tea drinkers.", "Green tea consumption reduced blood pressure in adults."])
    r = lite_verifier.verify_against("Green tea reduces blood pressure.", [doc])
    assert r.documents[0].title == "My abstract" and r.documents[0].url == ""
    assert r.documents[0].evidence[0].index in (0, 1)


def test_select_sentences():
    import numpy as np

    assert select_sentences(np.array([0.1, 0.9, 0.6, 0.7]), 0.5, 2) == ([1, 3], True)
    assert select_sentences(np.array([0.1, 0.3]), 0.5, 3) == ([1], False)
    assert select_sentences(np.array([]), 0.5, 3) == ([], False)


def test_evaluate_and_report(lite_verifier, lite_cfg, tmp_path):
    claims = load_claims(lite_cfg.data.data_dir, "dev")
    result = evaluate(lite_verifier, claims, ks=(1, 3, 5), depth=10, batch_size=4)
    m = result["metrics"]
    assert m["num_claims"] == 6
    assert 0 <= m["verdict_classification"]["accuracy"] <= 1
    assert m["evidence_retrieval"]["num_queries"] == 4
    assert m["evidence_retrieval"]["recall@5"] >= m["evidence_retrieval"]["recall@1"]
    assert 0 < m["evidence_retrieval"]["mrr"] <= 1
    assert len(result["predictions"]) == 6
    out = save_report(result, tmp_path / "rep", "Test report")
    assert (out / "metrics.json").exists() and (out / "predictions.jsonl").exists()
    assert "Macro F1" in markdown_report(m, "x") and "MRR" in (out / "report.md").read_text()


def test_calibration_round_trip(lite_verifier, lite_cfg):
    claims = load_claims(lite_cfg.data.data_dir, "train")
    result = calibrate(lite_verifier, claims, thresholds=(0.4, 0.6), nei_weights=(1.0, 2.0), batch_size=8)
    assert len(result["grid"]) == 4 and result["best"] in result["grid"]
    components = {"rationale": lite_verifier.rationale.describe(), "nli": lite_verifier.nli.describe()}
    save_calibration(lite_cfg.artifacts_dir, result, "train", components)
    assert load_calibration(lite_cfg.artifacts_dir)["threshold"] == result["best"]["threshold"]
    cal = lite_verifier.apply_calibration()
    assert cal is not None and lite_verifier.cfg.rationale.threshold == result["best"]["threshold"]
    assert lite_verifier.components()["decision"]["calibrated"]
