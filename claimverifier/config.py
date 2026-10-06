"""Configuration objects, loaded from YAML files in ``configs/``.

Every field has a default, so a YAML file only needs to list what it changes.
Values can also be overridden from the command line with ``--set section.key=value``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data/scifact"
    # Optional custom corpus in SciFact JSONL format ({doc_id, title, abstract: [sentences]}).
    corpus_path: Optional[str] = None


@dataclass
class RetrievalConfig:
    # "sbert": Sentence-BERT embeddings; "lsa": TF-IDF + truncated SVD (offline fallback).
    embedder: str = "sbert"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 256
    lsa_dim: int = 384
    # FAISS index type: "flat" (exact), "hnsw" or "ivf" (approximate, for large corpora).
    index_type: str = "flat"
    hnsw_m: int = 32
    ivf_nlist: int = 64
    ivf_nprobe: int = 16
    top_k: int = 5
    # Hybrid retrieval: fuse dense (FAISS) and BM25 rankings with reciprocal-rank fusion.
    hybrid: bool = True
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    candidate_pool: int = 100
    batch_size: int = 64


@dataclass
class RationaleConfig:
    # "scibert": fine-tuned SciBERT cross-encoder; "features": learned lightweight
    # scorer (logistic regression over similarity features); "similarity": zero-shot
    # embedding cosine similarity.
    method: str = "scibert"
    model_path: str = "models/scibert-rationale"
    base_model: str = "allenai/scibert_scivocab_uncased"
    # Methods tried in order when the configured one is not available (e.g. not trained yet).
    fallbacks: List[str] = field(default_factory=lambda: ["features", "similarity"])
    threshold: float = 0.5
    max_sentences: int = 3
    max_length: int = 256
    batch_size: int = 32
    # Logistic calibration of cosine similarity for the zero-shot "similarity" method.
    similarity_scale: float = 12.0
    similarity_center: float = 0.55


@dataclass
class NLIConfig:
    # "transformer": any Hugging Face sequence-classification NLI model (DeBERTa-v3 by default);
    # "lite": scikit-learn classifier trained on SciFact (offline fallback).
    method: str = "transformer"
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    # If this directory exists (created by `claimverifier train-nli`) it is used instead of model_name.
    finetuned_path: str = "models/deberta-v3-scifact"
    # Checkpoint that `train-nli` starts from (default: model_name).
    base_model: Optional[str] = None
    fallbacks: List[str] = field(default_factory=lambda: ["lite"])
    max_length: int = 512
    batch_size: int = 16
    # Prepend the paper title to the premise.
    include_title: bool = False


@dataclass
class AggregationConfig:
    # Multiplier for the "insufficient evidence" mass before normalisation (>1 = more cautious).
    nei_weight: float = 1.0
    # Ignore documents whose best rationale score is below this value.
    min_relevance: float = 0.0
    # Use <artifacts_dir>/calibration.json (written by `claimverifier calibrate`) when it matches the
    # current rationale + NLI models; it overrides rationale.threshold and aggregation.nei_weight.
    use_calibration: bool = True


@dataclass
class ExplanationConfig:
    # "transformers": local Hugging Face chat model (Qwen 2.5 / Llama 3 Instruct);
    # "ollama": local Ollama server; "openai": any OpenAI-compatible server (vLLM, llama.cpp,
    # LM Studio, ...); "template": deterministic extractive explanation; "none": disabled.
    backend: str = "transformers"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    openai_api_key_env: str = "OPENAI_API_KEY"
    max_new_tokens: int = 256
    temperature: float = 0.2
    max_evidence: int = 6
    timeout: float = 120.0
    fallback_to_template: bool = True


@dataclass
class Config:
    name: str = "default"
    artifacts_dir: str = "artifacts/default"
    reports_dir: str = "reports"
    device: str = "auto"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rationale: RationaleConfig = field(default_factory=RationaleConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    explanation: ExplanationConfig = field(default_factory=ExplanationConfig)

    # ------------------------------------------------------------------ helpers
    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _build(cls, values: Dict[str, Any]):
    """Recursively construct a dataclass from a (possibly partial) dict, rejecting unknown keys."""
    if values is None:
        return cls()
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        f = known[name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value or {})
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(values: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides to a raw config dict (values parsed as YAML)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = values
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    return values


def load_config(path: Optional[str | Path] = None, overrides: Optional[List[str]] = None) -> Config:
    """Load a :class:`Config` from YAML (or defaults when ``path`` is None)."""
    values: Dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
    values = apply_overrides(values, overrides or [])
    return _build(Config, values)


def resolve_device(device: str = "auto") -> str:
    """Resolve ``auto`` to ``cuda`` / ``mps`` / ``cpu`` (without importing torch unless needed)."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
