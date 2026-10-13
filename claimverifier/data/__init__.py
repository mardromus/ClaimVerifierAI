"""Dataset loading (SciFact and custom corpora in the same JSONL format)."""

from .scifact import (
    Claim,
    Document,
    EvidenceSet,
    dataset_statistics,
    download_scifact,
    load_claims,
    load_corpus,
)

__all__ = ["Claim", "Document", "EvidenceSet", "dataset_statistics", "download_scifact", "load_claims", "load_corpus"]
