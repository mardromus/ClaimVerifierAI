import pytest

from claimverifier.aggregation import aggregate
from claimverifier.labels import CONTRADICTED, INSUFFICIENT, SUPPORTED


def _sum(v):
    return sum(v.scores.values())


def test_no_documents_is_insufficient():
    v = aggregate([], [])
    assert v.label == INSUFFICIENT and v.confidence == pytest.approx(1.0)


def test_strong_support():
    v = aggregate([0.9, 0.2], [[0.95, 0.02, 0.03], [0.1, 0.1, 0.8]])
    assert v.label == SUPPORTED
    assert v.scores[SUPPORTED] > 0.85 and _sum(v) == pytest.approx(1.0)


def test_strong_contradiction():
    v = aggregate([0.8], [[0.05, 0.9, 0.05]])
    assert v.label == CONTRADICTED and _sum(v) == pytest.approx(1.0)


def test_irrelevant_evidence_is_down_weighted():
    # confident NLI on a barely relevant abstract should not produce a verdict
    v = aggregate([0.05], [[0.95, 0.0, 0.05]], threshold=0.5)
    assert v.label == INSUFFICIENT
    assert v.support_strength == pytest.approx(0.1 * 0.95)


def test_mixed_evidence_flag_and_split():
    v = aggregate([0.9, 0.9], [[0.9, 0.05, 0.05], [0.05, 0.85, 0.1]])
    assert v.mixed
    assert v.scores[SUPPORTED] > v.scores[CONTRADICTED]
    assert _sum(v) == pytest.approx(1.0)


def test_nei_weight_makes_verdict_more_cautious():
    probs = [[0.6, 0.1, 0.3]]
    assert aggregate([0.6], probs).label == SUPPORTED
    assert aggregate([0.6], probs, nei_weight=3.0).label == INSUFFICIENT


def test_min_relevance_filters_documents():
    v = aggregate([0.3], [[0.99, 0.0, 0.01]], threshold=0.3, min_relevance=0.5)
    assert v.label == INSUFFICIENT and v.support_strength == 0.0
