"""Calibrate the decision parameters of the pipeline on a labelled split.

The rationale threshold (which sentences count as evidence) and the weight of the
"insufficient evidence" outcome trade off Supported/Contradicted recall against
Insufficient-Evidence recall. ``calibrate`` grid-searches both to maximise claim-level
macro-F1 on a labelled split (the *train* split by default, so the dev split stays unseen)
and stores the result in ``<artifacts_dir>/calibration.json``, which the pipeline then uses.

Retrieval and rationale scores are computed once; NLI predictions are cached across grid points.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from ..data.scifact import Claim
from ..labels import LABELS
from ..nli import NLIModel
from .metrics import classification_metrics

logger = logging.getLogger(__name__)

CALIBRATION_FILE = "calibration.json"
DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.3, 0.951, 0.05), 2))
DEFAULT_NEI_WEIGHTS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class CachingNLI(NLIModel):
    def __init__(self, inner: NLIModel):
        self.inner = inner
        self.method = inner.method
        self.cache: Dict = {}

    def predict(self, pairs):
        missing = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        if missing:
            for pair, probs in zip(missing, self.inner.predict(missing)):
                self.cache[pair] = probs
        if not pairs:
            return np.zeros((0, len(LABELS)), dtype=np.float32)
        return np.stack([self.cache[p] for p in pairs])

    def describe(self):
        return self.inner.describe()


def calibrate(verifier, claims: List[Claim], thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
              nei_weights: Sequence[float] = DEFAULT_NEI_WEIGHTS, batch_size: int = 64,
              top_k: Optional[int] = None) -> Dict:
    claims = [c for c in claims if c.labelled]
    texts = [c.claim for c in claims]
    y_true = [c.label for c in claims]
    retrieved, scores = [], []
    for start in tqdm(range(0, len(texts), batch_size), desc="retrieval + rationales", unit="batch"):
        batch = texts[start:start + batch_size]
        r = verifier.retriever.search_batch(batch, top_k or verifier.cfg.retrieval.top_k)
        retrieved.extend(r)
        scores.extend(verifier.score_rationales(batch, r))

    original_nli = verifier.nli
    verifier.nli = CachingNLI(original_nli)
    grid = []
    try:
        for t in tqdm(thresholds, desc="grid search", unit="threshold"):
            for w in nei_weights:
                results = verifier._analyze(texts, retrieved, explain=False, sentence_stance=False,
                                            rationale_scores=scores, threshold=float(t), nei_weight=float(w))
                m = classification_metrics(y_true, [r.verdict for r in results], LABELS)
                grid.append({"threshold": float(t), "nei_weight": float(w), "macro_f1": m["macro_f1"],
                             "accuracy": m["accuracy"]})
    finally:
        verifier.nli = original_nli
    best = max(grid, key=lambda g: (round(g["macro_f1"], 6), g["accuracy"], -abs(g["nei_weight"] - 1.0)))
    logger.info("Best calibration: %s", best)
    return {"best": best, "num_claims": len(claims), "grid": grid}


def save_calibration(artifacts_dir: str | Path, result: Dict, split: str, components: Dict) -> Path:
    path = Path(artifacts_dir) / CALIBRATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"threshold": result["best"]["threshold"], "nei_weight": result["best"]["nei_weight"],
               "split": split, "num_claims": result["num_claims"], "macro_f1": result["best"]["macro_f1"],
               "accuracy": result["best"]["accuracy"], "components": components}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_calibration(artifacts_dir: str | Path) -> Optional[Dict]:
    path = Path(artifacts_dir) / CALIBRATION_FILE
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
