"""FAISS vector index over L2-normalised embeddings (inner product == cosine similarity)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _faiss():
    try:
        import faiss
    except ImportError as e:  # pragma: no cover
        raise ImportError("faiss-cpu is required: pip install faiss-cpu") from e
    return faiss


class FaissIndex:
    def __init__(self, dim: int, index_type: str = "flat", hnsw_m: int = 32,
                 ivf_nlist: int = 64, ivf_nprobe: int = 16):
        self.dim = dim
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.index = None

    def _create(self, n_vectors: int):
        faiss = _faiss()
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dim)
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = max(64, self.hnsw_m * 2)
            return index
        if self.index_type == "ivf":
            nlist = max(1, min(self.ivf_nlist, n_vectors // 39))  # FAISS wants ~39 points per centroid
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = min(self.ivf_nprobe, nlist)
            return index
        raise ValueError(f"Unknown FAISS index type {self.index_type!r} (flat | hnsw | ivf)")

    def build(self, embeddings: np.ndarray) -> "FaissIndex":
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings of shape (n, {self.dim}), got {embeddings.shape}")
        self.index = self._create(len(embeddings))
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        logger.info("Built FAISS %s index with %d vectors (dim=%d)", self.index_type, self.index.ntotal, self.dim)
        return self

    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, row_ids), each of shape (n_queries, k). Missing results have id -1."""
        if self.index is None:
            raise RuntimeError("Index is empty: call build() or load() first")
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k = max(1, min(k, self.size))
        return self.index.search(queries, k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _faiss().write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, index_type: str = "flat", ivf_nprobe: int = 16) -> "FaissIndex":
        faiss = _faiss()
        index = faiss.read_index(str(path))
        obj = cls(dim=index.d, index_type=index_type, ivf_nprobe=ivf_nprobe)
        try:
            ivf = faiss.extract_index_ivf(index)
            ivf.nprobe = min(ivf_nprobe, ivf.nlist)
        except RuntimeError:  # not an IVF index
            pass
        obj.index = index
        return obj
