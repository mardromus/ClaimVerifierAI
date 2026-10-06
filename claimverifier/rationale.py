"""Rationale (evidence sentence) selection.

Given a claim and the sentences of a retrieved abstract, score how likely each sentence is
to be evidence for or against the claim.

* ``scibert``    - SciBERT cross-encoder fine-tuned on SciFact rationales (``claimverifier train-rationale``).
* ``features``   - logistic regression over similarity / lexical / positional features
                   (``claimverifier train-lite``); fast, CPU-only.
* ``similarity`` - zero-shot cosine similarity between Sentence-BERT (or LSA) embeddings.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .text import content_words, numbers, tokenize

logger = logging.getLogger(__name__)

RESULT_CUES = {
    "significant", "significantly", "showed", "show", "shows", "found", "find", "associated", "association",
    "results", "result", "conclusion", "conclusions", "conclude", "demonstrate", "demonstrated", "demonstrates",
    "suggest", "suggests", "indicate", "indicates", "observed", "revealed", "reveal", "compared", "increased",
    "decreased", "reduced", "risk", "effect", "odds", "ratio", "confidence", "interval", "ci", "hr", "or",
}


class RationaleSelector:
    method = "base"

    def score_batch(self, items: Sequence[Tuple[str, Sequence[str]]]) -> List[np.ndarray]:
        """For each (claim, sentences) item, return an array of rationale probabilities."""
        raise NotImplementedError

    def score(self, claim: str, sentences: Sequence[str]) -> np.ndarray:
        return self.score_batch([(claim, sentences)])[0]

    def describe(self) -> Dict:
        return {"method": self.method}


# ---------------------------------------------------------------------------- SciBERT


class SciBertRationaleSelector(RationaleSelector):
    method = "scibert"

    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 32, max_length: int = 256):
        from .hf_utils import load_sequence_classifier

        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer, self.model = load_sequence_classifier(model_path, device)
        label2id = {k.upper(): v for k, v in (self.model.config.label2id or {}).items()}
        self.positive_id = int(label2id.get("RATIONALE", 1))

    def score_batch(self, items):
        from .hf_utils import predict_pair_probs

        pairs, sizes = [], []
        for claim, sentences in items:
            pairs.extend((claim, s) for s in sentences)
            sizes.append(len(sentences))
        probs = predict_pair_probs(self.model, self.tokenizer, pairs, self.batch_size, self.max_length, self.device)
        scores = probs[:, self.positive_id] if len(pairs) else np.zeros(0, dtype=np.float32)
        return _split(scores, sizes)

    def describe(self):
        return {"method": self.method, "model": self.model_path}


# ---------------------------------------------------------------------------- zero-shot similarity


class SimilarityRationaleSelector(RationaleSelector):
    method = "similarity"

    def __init__(self, embedder, scale: float = 12.0, center: float = 0.55):
        self.embedder = embedder
        self.scale = scale
        self.center = center

    def score_batch(self, items):
        claims = [c for c, _ in items]
        sentences = [s for _, sents in items for s in sents]
        sizes = [len(sents) for _, sents in items]
        claim_emb = self.embedder.encode(claims)
        sent_emb = self.embedder.encode(sentences)
        out, offset = [], 0
        for i, n in enumerate(sizes):
            cos = sent_emb[offset:offset + n] @ claim_emb[i]
            out.append(1.0 / (1.0 + np.exp(-self.scale * (cos - self.center))))
            offset += n
        return out

    def describe(self):
        return {"method": self.method, "embedder": self.embedder.describe()}


# ---------------------------------------------------------------------------- learned features


class FeatureRationaleSelector(RationaleSelector):
    """Logistic regression over semantic, lexical and positional features."""

    method = "features"
    FILE = "rationale_features.joblib"
    FEATURES = [
        "cosine", "cosine_minus_doc_max", "cosine_rank_frac", "idf_overlap", "claim_recall", "jaccard",
        "bigram_overlap", "number_overlap", "position", "is_first", "is_last", "log_length", "result_cues",
    ]

    def __init__(self, embedder, model=None, idf: Optional[Dict[str, float]] = None, signature: str = "",
                 shift: float = 0.0):
        self.embedder = embedder
        self.model = model
        self.idf = idf or {}
        self.signature = signature
        # Logit shift that maps the cross-validated F1-optimal operating point to a score of 0.5.
        self.shift = shift
        self._default_idf = max(self.idf.values()) if self.idf else 1.0  # unseen words are rare words

    # -- features
    def _idf(self, word: str) -> float:
        return self.idf.get(word, self._default_idf)

    def featurize(self, items: Sequence[Tuple[str, Sequence[str]]]) -> List[np.ndarray]:
        claims = [c for c, _ in items]
        sentences = [s for _, sents in items for s in sents]
        claim_emb = self.embedder.encode(claims)
        sent_emb = self.embedder.encode(sentences)
        feats, offset = [], 0
        for ci, (claim, sents) in enumerate(items):
            n = len(sents)
            if n == 0:
                feats.append(np.zeros((0, len(self.FEATURES)), dtype=np.float32))
                continue
            cos = sent_emb[offset:offset + n] @ claim_emb[ci]
            offset += n
            order = np.argsort(-cos, kind="stable")
            rank = np.empty(n)
            rank[order] = np.arange(n)
            c_words = content_words(claim)
            c_idf = sum(self._idf(w) for w in c_words) or 1.0
            c_tokens = tokenize(claim)
            c_bigrams = set(zip(c_tokens, c_tokens[1:]))
            c_nums = numbers(claim)
            rows = []
            for i, sent in enumerate(sents):
                s_words = content_words(sent)
                s_tokens = tokenize(sent)
                s_bigrams = set(zip(s_tokens, s_tokens[1:]))
                shared = c_words & s_words
                rows.append([
                    cos[i],
                    cos[i] - cos.max(),
                    1.0 - rank[i] / max(n - 1, 1),
                    sum(self._idf(w) for w in shared) / c_idf,
                    len(shared) / (len(c_words) or 1),
                    len(shared) / (len(c_words | s_words) or 1),
                    len(c_bigrams & s_bigrams) / (len(c_bigrams) or 1),
                    float(bool(c_nums & numbers(sent))),
                    i / max(n - 1, 1),
                    float(i == 0),
                    float(i == n - 1),
                    math.log1p(len(s_tokens)),
                    sum(t in RESULT_CUES for t in s_tokens) / (len(s_tokens) or 1),
                ])
            feats.append(np.asarray(rows, dtype=np.float32))
        return feats

    def score_batch(self, items):
        if self.model is None:
            raise RuntimeError("FeatureRationaleSelector is not trained")
        feats = self.featurize(items)
        sizes = [len(f) for f in feats]
        if sum(sizes) == 0:
            return [np.zeros(0, dtype=np.float32) for _ in items]
        probs = self.model.predict_proba(np.vstack([f for f in feats if len(f)]))[:, 1]
        return _split(shift_probabilities(probs, self.shift).astype(np.float32), sizes)

    # -- persistence
    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "idf": self.idf, "signature": self.signature, "shift": self.shift},
                    directory / self.FILE)

    @classmethod
    def load(cls, directory: Path, embedder, expected_signature: Optional[str] = None) -> "FeatureRationaleSelector":
        import joblib

        state = joblib.load(directory / cls.FILE)
        if expected_signature and state["signature"] != expected_signature:
            raise RuntimeError(f"Rationale features model in {directory} was trained with {state['signature']}, "
                               f"but the current embedder is {expected_signature}. Re-run `claimverifier train-lite`.")
        return cls(embedder, state["model"], state["idf"], state["signature"], state.get("shift", 0.0))

    def describe(self):
        return {"method": self.method, "embedder": self.embedder.describe()}


def shift_probabilities(probs: np.ndarray, shift: float) -> np.ndarray:
    """sigmoid(logit(p) - shift): moves the decision point p = sigmoid(shift) to 0.5."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) - shift)))


def _split(values: np.ndarray, sizes: Sequence[int]) -> List[np.ndarray]:
    out, offset = [], 0
    for n in sizes:
        out.append(np.asarray(values[offset:offset + n], dtype=np.float32))
        offset += n
    return out


def select_sentences(scores: np.ndarray, threshold: float, max_sentences: int) -> Tuple[List[int], bool]:
    """Indices of selected rationale sentences (in abstract order) and whether they passed the threshold.

    If no sentence reaches the threshold, the single best sentence is returned with ``passed=False``
    so that it can still be shown (and weighted down) as weak evidence.
    """
    if len(scores) == 0:
        return [], False
    order = np.argsort(-scores, kind="stable")
    chosen = [int(i) for i in order[:max_sentences] if scores[i] >= threshold]
    if chosen:
        return sorted(chosen), True
    return [int(order[0])], False


def create_rationale_selector(cfg, embedder, artifacts_dir: str | Path, device: str = "cpu") -> RationaleSelector:
    """Instantiate the configured selector, falling back along ``cfg.rationale.fallbacks``."""
    from .retrieval.embedders import embedder_signature

    rcfg = cfg.rationale
    methods = [rcfg.method] + [m for m in rcfg.fallbacks if m != rcfg.method]
    lite_dir = Path(artifacts_dir) / "lite"
    errors = []
    for method in methods:
        try:
            if method == "scibert":
                if not (Path(rcfg.model_path) / "config.json").exists():
                    raise FileNotFoundError(f"no fine-tuned SciBERT model at {rcfg.model_path} "
                                            f"(run `python -m claimverifier train-rationale`)")
                selector = SciBertRationaleSelector(rcfg.model_path, device, rcfg.batch_size, rcfg.max_length)
            elif method == "features":
                if not (lite_dir / FeatureRationaleSelector.FILE).exists():
                    raise FileNotFoundError(f"no feature model in {lite_dir} (run `python -m claimverifier train-lite`)")
                selector = FeatureRationaleSelector.load(lite_dir, embedder, embedder_signature(cfg.retrieval))
            elif method == "similarity":
                selector = SimilarityRationaleSelector(embedder, rcfg.similarity_scale, rcfg.similarity_center)
            else:
                raise ValueError(f"unknown rationale method {method!r}")
        except Exception as e:  # noqa: BLE001 - fall back to the next method
            errors.append(f"{method}: {e}")
            logger.warning("Rationale selector %r unavailable: %s", method, e)
            continue
        if method != rcfg.method:
            logger.warning("Using fallback rationale selector %r", method)
        return selector
    raise RuntimeError("No rationale selector available: " + "; ".join(errors))
