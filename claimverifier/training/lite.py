"""Training of the lightweight (CPU, scikit-learn) rationale selector and NLI model."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..data.scifact import Claim, Document
from ..evaluation.metrics import best_threshold, binary_prf, classification_metrics
from ..labels import CONTRADICTED, INSUFFICIENT, LABELS, SUPPORTED
from ..nli import LiteNLI
from ..rationale import FeatureRationaleSelector, shift_probabilities
from ..retrieval.embedders import Embedder, embedder_signature
from ..text import content_words
from .examples import build_nli_examples, build_rationale_examples

logger = logging.getLogger(__name__)


def sentence_idf(corpus: List[Document]) -> Dict[str, float]:
    """Inverse document frequency of content words, computed over abstract sentences."""
    df: Counter = Counter()
    n = 0
    for doc in corpus:
        for sent in doc.sentences:
            df.update(content_words(sent))
            n += 1
    return {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}


def _group_items(examples):
    """Group (claim, doc) rationale examples so features can use document context."""
    groups: Dict[tuple, list] = {}
    for e in examples:
        groups.setdefault((e.claim_id, e.doc_id), []).append(e)
    return list(groups.values())


def retrieved_doc_ids(retriever, claims: List[Claim], top_k: int) -> Dict[int, List[int]]:
    """claim id -> top-k retrieved doc ids (hard negatives for rationale training)."""
    ranked = retriever.rank_doc_ids([c.claim for c in claims], depth=top_k)
    return {c.id: r for c, r in zip(claims, ranked)}


def _rationale_xy(selector: FeatureRationaleSelector, claims: List[Claim], corpus: List[Document],
                  retrieved: Optional[Dict[int, List[int]]] = None):
    doc_by_id = {d.doc_id: d for d in corpus}
    examples = build_rationale_examples(claims, corpus, retrieved=retrieved)
    groups = _group_items(examples)
    items = [(g[0].claim, doc_by_id[g[0].doc_id].sentences) for g in groups]
    feats = selector.featurize(items)
    X, y, claim_ids = [], [], []
    for g, f in zip(groups, feats):
        labels = np.zeros(len(f), dtype=int)
        for e in g:
            labels[e.sentence_index] = e.label
        X.append(f)
        y.append(labels)
        claim_ids.append(np.full(len(f), g[0].claim_id))
    return np.vstack(X), np.concatenate(y), np.concatenate(claim_ids)


def train_lite_models(cfg: Config, retriever, corpus: List[Document], train_claims: List[Claim],
                      dev_claims: List[Claim]) -> Dict:
    """Train and save the feature-based rationale selector and the lite NLI model.

    The rationale selector is trained on the cited abstracts plus the retriever's top-k abstracts for
    each claim (hard negatives), i.e. on the same kind of input it sees inside the pipeline.
    """
    embedder: Embedder = retriever.embedder
    top_k = cfg.retrieval.top_k
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict

    out_dir = Path(cfg.artifacts_dir) / "lite"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {}

    # -- rationale selector
    selector = FeatureRationaleSelector(embedder, idf=sentence_idf(corpus), signature=embedder_signature(cfg.retrieval))
    X_train, y_train, groups = _rationale_xy(selector, train_claims, corpus,
                                             retrieved_doc_ids(retriever, train_claims, top_k))
    X_dev, y_dev, _ = _rationale_xy(selector, dev_claims, corpus, retrieved_doc_ids(retriever, dev_claims, top_k))
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=cfg.seed)
    # Choose the operating point on out-of-fold train predictions (grouped by claim, no dev peeking)
    # and shift the scores so that it corresponds to 0.5.
    oof = cross_val_predict(clf, X_train, y_train, groups=groups, cv=GroupKFold(n_splits=5),
                            method="predict_proba")[:, 1]
    cv_best = best_threshold(y_train, oof, grid=np.round(np.arange(0.05, 0.951, 0.01), 2))
    clf.fit(X_train, y_train)
    selector.model = clf
    selector.shift = float(np.log(cv_best["threshold"] / (1 - cv_best["threshold"])))
    selector.save(out_dir)
    dev_scores = shift_probabilities(clf.predict_proba(X_dev)[:, 1], selector.shift)
    report["rationale"] = {
        "train_examples": int(len(y_train)), "train_positive": int(y_train.sum()),
        "cv_operating_point": cv_best,
        "dev_examples": int(len(y_dev)), "dev_positive": int(y_dev.sum()),
        "dev_at_threshold": {"threshold": cfg.rationale.threshold,
                             **binary_prf(y_dev, (dev_scores >= cfg.rationale.threshold).astype(int))},
        "dev_best_threshold": best_threshold(y_dev, dev_scores),
        "coefficients": dict(zip(FeatureRationaleSelector.FEATURES, np.round(clf.coef_[0], 4).tolist())),
    }
    logger.info("Feature rationale selector dev: %s", report["rationale"]["dev_at_threshold"])

    # -- NLI (stance: support vs. contradict, given gold evidence)
    train_ex = build_nli_examples(train_claims, corpus, include_title=cfg.nli.include_title, seed=cfg.seed)
    dev_ex = build_nli_examples(dev_claims, corpus, include_title=cfg.nli.include_title, seed=cfg.seed)
    nli = LiteNLI(stance_only=True).fit([(e.premise, e.hypothesis) for e in train_ex], [e.label for e in train_ex],
                                        seed=cfg.seed)
    nli.save(out_dir)
    dev_ex = [e for e in dev_ex if e.label != INSUFFICIENT] if nli.stance_only else dev_ex
    labels = [SUPPORTED, CONTRADICTED] if nli.stance_only else LABELS
    probs = nli.predict([(e.premise, e.hypothesis) for e in dev_ex])
    preds = [LABELS[i] for i in probs.argmax(1)]
    m = classification_metrics([e.label for e in dev_ex], preds, labels)
    report["nli"] = {"stance_only": nli.stance_only, "train_examples": len(train_ex), "dev_examples": len(dev_ex),
                     "dev_accuracy": m["accuracy"], "dev_macro_f1": m["macro_f1"],
                     "dev_per_class": m["per_class"]}
    logger.info("Lite NLI dev (gold evidence): accuracy=%.3f macro-F1=%.3f", m["accuracy"], m["macro_f1"])

    with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
