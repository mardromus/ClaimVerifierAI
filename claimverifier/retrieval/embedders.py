"""Text embedders producing L2-normalised float32 vectors (cosine similarity = inner product).

* :class:`SentenceBertEmbedder` - Sentence-BERT (``sentence-transformers``) semantic embeddings.
* :class:`LSAEmbedder` - TF-IDF + truncated SVD (latent semantic analysis); a dependency-light
  fallback that needs no model download.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class Embedder:
    name: str = "base"
    dim: int

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        raise NotImplementedError

    def fit(self, texts: Sequence[str]) -> "Embedder":  # only needed by corpus-fitted embedders
        return self

    def save(self, directory: Path) -> None:
        pass

    def describe(self) -> dict:
        return {"embedder": self.name, "dim": self.dim}


class SentenceBertEmbedder(Embedder):
    name = "sbert"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", max_seq_length: Optional[int] = 256):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise ImportError("sentence-transformers is required for the 'sbert' embedder. "
                              "Install it with `pip install -r requirements.txt` or use embedder: lsa.") from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        getter = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = int(getter())

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress,
                                convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def describe(self) -> dict:
        return {"embedder": self.name, "model_name": self.model_name, "dim": self.dim}


class LSAEmbedder(Embedder):
    """TF-IDF (word uni+bi-grams, sublinear tf) projected with truncated SVD."""

    name = "lsa"
    FILE = "lsa_embedder.joblib"

    def __init__(self, dim: int = 384, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.vectorizer = None
        self.svd = None
        self._projection = None

    @property
    def projection(self) -> np.ndarray:
        # (vocab, dim) C-contiguous copy of the SVD components: sparse @ dense without a per-call copy.
        if self._projection is None:
            self._projection = np.ascontiguousarray(self.svd.components_.T, dtype=np.float32)
        return self._projection

    def fit(self, texts: Sequence[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from ..text import STOP_WORDS

        # Rare terms are noise in a large corpus but carry the topic in a small one.
        min_df = 2 if len(texts) >= 1000 else 1
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.5, sublinear_tf=True,
                                          stop_words=list(STOP_WORDS), token_pattern=r"(?u)\b[\w-]{2,}\b")
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = max(2, min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed, n_iter=7)
        self.svd.fit(tfidf)
        self.dim = n_components
        logger.info("Fitted LSA embedder: vocab=%d, dim=%d", len(self.vectorizer.vocabulary_), self.dim)
        return self

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("LSAEmbedder must be fitted (or loaded) before encoding")
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        tfidf = self.vectorizer.transform(list(texts)).astype(np.float32)
        return l2_normalize(np.asarray(tfidf @ self.projection))

    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd, "dim": self.dim}, directory / self.FILE)

    @classmethod
    def load(cls, directory: Path) -> "LSAEmbedder":
        import joblib

        state = joblib.load(directory / cls.FILE)
        emb = cls(dim=state["dim"])
        emb.vectorizer, emb.svd = state["vectorizer"], state["svd"]
        return emb


def create_embedder(retrieval_cfg, device: str = "cpu", fit_texts: Optional[List[str]] = None,
                    directory: Optional[Path] = None, seed: int = 42) -> Embedder:
    """Create the embedder described by a :class:`RetrievalConfig`.

    For LSA, the model is loaded from ``directory`` when available, otherwise fitted on ``fit_texts``.
    """
    kind = retrieval_cfg.embedder
    if kind == "sbert":
        return SentenceBertEmbedder(retrieval_cfg.model_name, device=device,
                                    max_seq_length=retrieval_cfg.max_seq_length)
    if kind == "lsa":
        if directory is not None and (directory / LSAEmbedder.FILE).exists():
            return LSAEmbedder.load(directory)
        if fit_texts is None:
            raise RuntimeError("LSA embedder is not fitted; build the index first (`claimverifier index`).")
        emb = LSAEmbedder(dim=retrieval_cfg.lsa_dim, seed=seed).fit(fit_texts)
        if directory is not None:
            emb.save(directory)
        return emb
    raise ValueError(f"Unknown embedder {kind!r} (expected 'sbert' or 'lsa')")


def embedder_signature(retrieval_cfg) -> str:
    if retrieval_cfg.embedder == "sbert":
        return json.dumps({"embedder": "sbert", "model_name": retrieval_cfg.model_name})
    return json.dumps({"embedder": retrieval_cfg.embedder})
