import numpy as np
import pytest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from claimverifier.data.scifact import Claim, EvidenceSet
from claimverifier.evaluation.metrics import (
    best_threshold,
    binary_prf,
    classification_metrics,
    retrieval_metrics,
    scifact_abstract_metrics,
    scifact_sentence_metrics,
)
from claimverifier.labels import CONTRADICTED, INSUFFICIENT, LABELS, SUPPORTED


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = list(rng.choice(LABELS, size=200))
    y_pred = list(rng.choice(LABELS, size=200))
    m = classification_metrics(y_true, y_pred, LABELS)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["macro_precision"] == pytest.approx(p.mean())
    assert m["macro_recall"] == pytest.approx(r.mean())
    assert m["macro_f1"] == pytest.approx(f.mean())
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    assert m["weighted_f1"] == pytest.approx(wf)
    for i, lab in enumerate(LABELS):
        assert m["per_class"][lab]["f1"] == pytest.approx(f[i])
        assert m["per_class"][lab]["support"] == s[i]
    assert np.array(m["confusion_matrix"]["matrix"]).sum() == 200


def test_classification_handles_missing_class():
    m = classification_metrics([SUPPORTED, SUPPORTED], [SUPPORTED, INSUFFICIENT], LABELS)
    assert m["accuracy"] == 0.5
    assert m["per_class"][CONTRADICTED] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def test_retrieval_metrics_hand_computed():
    ranked = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    gold = [{2}, {8, 99}, set()]  # third query has no relevant docs and is ignored
    m = retrieval_metrics(ranked, gold, ks=(1, 2, 4))
    assert m["num_queries"] == 2
    assert m["recall@1"] == 0.0
    assert m["recall@2"] == pytest.approx(0.5)            # (1 + 0) / 2
    assert m["recall@4"] == pytest.approx((1 + 0.5) / 2)
    assert m["hit@4"] == 1.0
    assert m["precision@2"] == pytest.approx((0.5 + 0) / 2)
    assert m["mrr"] == pytest.approx((1 / 2 + 1 / 4) / 2)


def _claim(evidence):
    return Claim(id=1, claim="c", evidence={d: [EvidenceSet(s, lab)] for d, (s, lab) in evidence.items()})


def test_scifact_abstract_metrics():
    claims = [_claim({10: ([1, 2], SUPPORTED), 11: ([0], SUPPORTED)}), _claim({})]
    preds = [
        {10: {"label": SUPPORTED, "sentences": [1, 2, 5]},   # correct label + rationale
         11: {"label": CONTRADICTED, "sentences": [0]},      # wrong label
         12: {"label": SUPPORTED, "sentences": [0]}},        # not an evidence abstract
        {13: {"label": SUPPORTED, "sentences": [1]}},        # NEI claim -> false positive
    ]
    m = scifact_abstract_metrics(preds, claims)
    assert m["gold_abstracts"] == 2 and m["predicted_abstracts"] == 4
    assert m["label_only"]["precision"] == pytest.approx(1 / 4)
    assert m["label_only"]["recall"] == pytest.approx(1 / 2)
    assert m["rationalized"]["recall"] == pytest.approx(1 / 2)
    # rationale must contain a full gold set within the first 3 predicted sentences
    preds[0][10]["sentences"] = [1, 5, 6, 2]
    assert scifact_abstract_metrics(preds, claims)["rationalized"]["recall"] == 0.0
    s = scifact_sentence_metrics(preds, claims)
    assert s["recall"] == pytest.approx(3 / 3)  # sentences 1, 2 of doc 10 and 0 of doc 11


def test_binary_and_threshold():
    y = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.4, 0.3, 0.8, 0.6])
    assert binary_prf(y, scores >= 0.5) == pytest.approx({"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3})
    best = best_threshold(y, scores)
    assert best["f1"] >= 2 / 3
