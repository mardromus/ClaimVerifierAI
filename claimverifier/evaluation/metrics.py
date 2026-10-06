"""Evaluation metrics.

* Classification: Accuracy, Precision, Recall, F1 (per class, macro and weighted), confusion matrix.
* Retrieval: Recall@K, Precision@K, Hit@K and Mean Reciprocal Rank (MRR).
* Rationale (evidence sentence) selection: sentence-level Precision / Recall / F1.
* SciFact abstract-level evaluation (label-only and rationalized), following Wadden et al. (2020).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def f1_from_pr(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    labels = list(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    per_class = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        precision = _safe_div(tp, cm[:, i].sum())
        recall = _safe_div(tp, cm[i, :].sum())
        per_class[lab] = {"precision": precision, "recall": recall, "f1": f1_from_pr(precision, recall),
                          "support": int(cm[i, :].sum())}
    supports = np.array([per_class[lab]["support"] for lab in labels], dtype=float)
    total = supports.sum()

    def avg(key: str, weighted: bool) -> float:
        values = np.array([per_class[lab][key] for lab in labels])
        if weighted:
            return float((values * supports).sum() / total) if total else 0.0
        return float(values.mean())

    return {
        "accuracy": _safe_div(np.trace(cm), cm.sum()),
        "macro_precision": avg("precision", False),
        "macro_recall": avg("recall", False),
        "macro_f1": avg("f1", False),
        "weighted_precision": avg("precision", True),
        "weighted_recall": avg("recall", True),
        "weighted_f1": avg("f1", True),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "num_examples": int(cm.sum()),
    }


def binary_prf(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true, y_pred = np.asarray(list(y_true)), np.asarray(list(y_pred))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    precision = _safe_div(tp, (y_pred == 1).sum())
    recall = _safe_div(tp, (y_true == 1).sum())
    return {"precision": precision, "recall": recall, "f1": f1_from_pr(precision, recall)}


# ---------------------------------------------------------------------------- retrieval


def retrieval_metrics(ranked: Sequence[Sequence[int]], gold: Sequence[Set[int]],
                      ks: Sequence[int] = (1, 3, 5, 10, 20)) -> Dict[str, float]:
    """Recall@K, Precision@K, Hit@K and MRR over queries with at least one relevant document.

    Recall@K = |relevant in top-K| / |relevant|, averaged over queries.
    MRR      = mean of 1 / rank of the first relevant document (0 if none retrieved).
    """
    pairs = [(list(r), set(g)) for r, g in zip(ranked, gold) if g]
    n = len(pairs)
    out: Dict[str, float] = {"num_queries": n}
    if n == 0:
        return out
    for k in ks:
        recalls, precisions, hits = [], [], []
        for r, g in pairs:
            found = len(set(r[:k]) & g)
            recalls.append(found / len(g))
            precisions.append(found / k)
            hits.append(float(found > 0))
        out[f"recall@{k}"] = float(np.mean(recalls))
        out[f"precision@{k}"] = float(np.mean(precisions))
        out[f"hit@{k}"] = float(np.mean(hits))
    rr = []
    for r, g in pairs:
        rank = next((i + 1 for i, d in enumerate(r) if d in g), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr"] = float(np.mean(rr))
    out["depth"] = max((len(r) for r, _ in pairs), default=0)
    return out


# ---------------------------------------------------------------------------- SciFact abstract-level


def scifact_abstract_metrics(predictions: Sequence[Mapping], claims: Sequence, max_sentences: int = 3) -> Dict:
    """Abstract-level evaluation from the SciFact paper.

    ``predictions[i]`` maps doc_id -> {"label": canonical label, "sentences": [sentence indices]} for
    the abstracts predicted as SUPPORTED/CONTRADICTED for ``claims[i]``.

    * label-only: a predicted abstract is correct if it is a gold evidence abstract and the label matches.
    * rationalized: additionally, the predicted sentences (first ``max_sentences``) must contain all
      sentences of at least one gold rationale set.
    """
    n_pred = n_gold = correct_label = correct_rationalized = 0
    for pred, claim in zip(predictions, claims):
        n_gold += len(claim.evidence)
        for doc_id, p in pred.items():
            n_pred += 1
            doc_id = int(doc_id)
            if doc_id not in claim.evidence or p["label"] != claim.doc_label(doc_id):
                continue
            correct_label += 1
            sentences = set(list(p.get("sentences", []))[:max_sentences])
            if any(set(es.sentences) <= sentences for es in claim.evidence[doc_id]):
                correct_rationalized += 1

    def prf(correct: int) -> Dict[str, float]:
        p, r = _safe_div(correct, n_pred), _safe_div(correct, n_gold)
        return {"precision": p, "recall": r, "f1": f1_from_pr(p, r)}

    return {"label_only": prf(correct_label), "rationalized": prf(correct_rationalized),
            "predicted_abstracts": n_pred, "gold_abstracts": n_gold}


def scifact_sentence_metrics(predictions: Sequence[Mapping], claims: Sequence) -> Dict:
    """Sentence-level selection P/R/F1: a predicted (doc, sentence) is correct if it is a gold rationale
    sentence of that abstract (selection-only, ignoring the label)."""
    n_pred = n_gold = correct = 0
    for pred, claim in zip(predictions, claims):
        gold: Dict[int, Set[int]] = {d: claim.rationale_sentences(d) for d in claim.evidence}
        n_gold += sum(len(s) for s in gold.values())
        for doc_id, p in pred.items():
            sents = set(p.get("sentences", []))
            n_pred += len(sents)
            correct += len(sents & gold.get(int(doc_id), set()))
    p, r = _safe_div(correct, n_pred), _safe_div(correct, n_gold)
    return {"precision": p, "recall": r, "f1": f1_from_pr(p, r)}


def average_precision_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    return binary_prf(y_true, (scores >= threshold).astype(int))


def best_threshold(y_true: np.ndarray, scores: np.ndarray,
                   grid: Sequence[float] = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))) -> Dict[str, float]:
    best = {"threshold": 0.5, "f1": -1.0}
    for t in grid:
        m = binary_prf(y_true, (scores >= t).astype(int))
        if m["f1"] > best["f1"]:
            best = {"threshold": float(t), **m}
    return best


def summarize_for_table(metrics: Dict, keys: List[str]) -> Dict[str, float]:
    return {k: round(float(metrics[k]), 4) for k in keys if k in metrics}
