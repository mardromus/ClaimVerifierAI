"""Abstract retrieval: dense semantic search (embedder + FAISS), optionally fused with BM25."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.scifact import Document
from .bm25 import BM25Index
from .embedders import Embedder, create_embedder, embedder_signature
from .index import FaissIndex

logger = logging.getLogger(__name__)

INDEX_DIR = "index"


@dataclass
class RetrievedDoc:
    doc: Document
    rank: int
    score: float          # final ranking score (RRF score when hybrid, cosine otherwise)
    dense_score: float    # cosine similarity between claim and abstract embeddings
    bm25_score: float = 0.0


def corpus_fingerprint(corpus: Sequence[Document]) -> str:
    h = hashlib.sha1()
    for d in corpus:
        h.update(str(d.doc_id).encode())
        h.update(d.title.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


class Retriever:
    def __init__(self, cfg, corpus: List[Document], embedder: Embedder, index: FaissIndex,
                 doc_embeddings: np.ndarray, bm25: Optional[BM25Index] = None):
        self.cfg = cfg
        self.corpus = corpus
        self.embedder = embedder
        self.index = index
        self.doc_embeddings = doc_embeddings
        self.bm25 = bm25
        self.doc_by_id: Dict[int, Document] = {d.doc_id: d for d in corpus}

    # ------------------------------------------------------------------ build / load
    @staticmethod
    def index_dir(artifacts_dir: str | Path) -> Path:
        return Path(artifacts_dir) / INDEX_DIR

    @classmethod
    def build(cls, cfg, corpus: List[Document], artifacts_dir: str | Path, device: str = "cpu",
              seed: int = 42, show_progress: bool = True) -> "Retriever":
        directory = cls.index_dir(artifacts_dir)
        directory.mkdir(parents=True, exist_ok=True)
        texts = [d.full_text for d in corpus]
        lsa_file = directory / "lsa_embedder.joblib"
        if lsa_file.exists():
            lsa_file.unlink()  # always refit LSA when rebuilding
        embedder = create_embedder(cfg, device=device, fit_texts=texts, directory=directory, seed=seed)
        logger.info("Encoding %d abstracts with %s", len(texts), embedder.describe())
        embeddings = embedder.encode(texts, batch_size=cfg.batch_size, show_progress=show_progress)
        index = FaissIndex(embedder.dim, cfg.index_type, cfg.hnsw_m, cfg.ivf_nlist, cfg.ivf_nprobe)
        index.build(embeddings)
        index.save(directory / "faiss.index")
        np.save(directory / "doc_embeddings.npy", embeddings)
        meta = {
            "signature": embedder_signature(cfg),
            "embedder": embedder.describe(),
            "index_type": cfg.index_type,
            "num_documents": len(corpus),
            "doc_ids": [d.doc_id for d in corpus],
            "corpus_fingerprint": corpus_fingerprint(corpus),
        }
        with open(directory / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        bm25 = BM25Index(texts) if cfg.hybrid else None
        return cls(cfg, corpus, embedder, index, embeddings, bm25)

    @classmethod
    def exists(cls, artifacts_dir: str | Path) -> bool:
        d = cls.index_dir(artifacts_dir)
        return (d / "faiss.index").exists() and (d / "meta.json").exists()

    @classmethod
    def load(cls, cfg, corpus: List[Document], artifacts_dir: str | Path, device: str = "cpu",
             embedder: Optional[Embedder] = None) -> "Retriever":
        directory = cls.index_dir(artifacts_dir)
        if not cls.exists(artifacts_dir):
            raise FileNotFoundError(f"No index found in {directory}. Run `python -m claimverifier index` first.")
        with open(directory / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["signature"] != embedder_signature(cfg):
            raise RuntimeError(f"Index in {directory} was built with {meta['signature']} but the config asks for "
                               f"{embedder_signature(cfg)}. Rebuild it with `python -m claimverifier index`.")
        if meta["corpus_fingerprint"] != corpus_fingerprint(corpus):
            raise RuntimeError(f"Index in {directory} was built for a different corpus. Rebuild the index.")
        if embedder is None:
            embedder = create_embedder(cfg, device=device, directory=directory)
        index = FaissIndex.load(directory / "faiss.index", cfg.index_type, cfg.ivf_nprobe)
        embeddings = np.load(directory / "doc_embeddings.npy")
        bm25 = BM25Index([d.full_text for d in corpus]) if cfg.hybrid else None
        return cls(cfg, corpus, embedder, index, embeddings, bm25)

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDoc]:
        return self.search_batch([query], top_k)[0]

    def search_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[List[RetrievedDoc]]:
        top_k = top_k or self.cfg.top_k
        top_k = min(top_k, len(self.corpus))
        query_emb = self.embedder.encode(queries, batch_size=self.cfg.batch_size)
        pool = max(top_k, self.cfg.candidate_pool) if self.bm25 is not None else top_k
        dense_scores, dense_ids = self.index.search(query_emb, pool)
        results = []
        for qi, query in enumerate(queries):
            d_ids = [int(i) for i in dense_ids[qi] if i >= 0]
            if self.bm25 is None:
                ranking = d_ids[:top_k]
                fused = {i: float(s) for i, s in zip(d_ids, dense_scores[qi])}
                bm25_all = None
            else:
                bm25_all = self.bm25.scores(query)
                b_ids = np.argsort(-bm25_all, kind="stable")[:pool]
                fused: Dict[int, float] = {}
                for rank, i in enumerate(d_ids):
                    fused[i] = fused.get(i, 0.0) + self.cfg.dense_weight / (self.cfg.rrf_k + rank + 1)
                for rank, i in enumerate(b_ids):
                    i = int(i)
                    fused[i] = fused.get(i, 0.0) + self.cfg.bm25_weight / (self.cfg.rrf_k + rank + 1)
                ranking = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
            cosines = self.doc_embeddings[ranking] @ query_emb[qi]
            results.append([
                RetrievedDoc(doc=self.corpus[i], rank=r + 1, score=float(fused[i]), dense_score=float(cos),
                             bm25_score=float(bm25_all[i]) if bm25_all is not None else 0.0)
                for r, (i, cos) in enumerate(zip(ranking, cosines))
            ])
        return results

    def rank_doc_ids(self, queries: List[str], depth: int = 100) -> List[List[int]]:
        """Ranked doc-id lists (used to compute Recall@K / MRR)."""
        return [[r.doc.doc_id for r in res] for res in self.search_batch(queries, depth)]
