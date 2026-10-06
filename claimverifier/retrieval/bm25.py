"""Okapi BM25 lexical retrieval (used for hybrid dense + sparse search)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..text import tokenize


class BM25Index:
    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:  # pragma: no cover
            raise ImportError("rank-bm25 is required for hybrid retrieval: pip install rank-bm25") from e
        tokenized = [tokenize(t) or ["<empty>"] for t in texts]
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)
        self.size = len(tokenized)

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(self.size, dtype=np.float32)
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)

    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.scores(query)
        k = min(k, self.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return scores[top], top

    def search_batch(self, queries: List[str], k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [self.search(q, k) for q in queries]
