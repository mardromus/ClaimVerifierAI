"""High-level training entry points used by the CLI."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..config import Config, resolve_device
from ..data.scifact import load_claims, load_corpus
from ..evaluation.metrics import best_threshold, binary_prf, classification_metrics
from ..labels import LABEL2ID, LABELS, normalize_model_label
from .examples import build_nli_examples, build_rationale_examples
from .transformer_trainer import TrainingArgs, finetune_pair_classifier

logger = logging.getLogger(__name__)


def _load_data(cfg: Config):
    corpus = load_corpus(cfg.data.data_dir, cfg.data.corpus_path)
    return corpus, load_claims(cfg.data.data_dir, "train"), load_claims(cfg.data.data_dir, "dev")


# ---------------------------------------------------------------------------- SciBERT rationale selector


def _retrieved_negatives(cfg: Config, corpus, claims_by_split, k: int):
    from ..retrieval.retriever import Retriever

    device = resolve_device(cfg.device)
    if Retriever.exists(cfg.artifacts_dir):
        retriever = Retriever.load(cfg.retrieval, corpus, cfg.artifacts_dir, device)
    else:
        retriever = Retriever.build(cfg.retrieval, corpus, cfg.artifacts_dir, device, cfg.seed)
    out = []
    for claims in claims_by_split:
        ranked = retriever.rank_doc_ids([c.claim for c in claims], depth=k)
        out.append({c.id: r for c, r in zip(claims, ranked)})
    return out


def train_rationale_model(cfg: Config, base_model: Optional[str] = None, output_dir: Optional[str] = None,
                          args: Optional[TrainingArgs] = None, neg_ratio: Optional[float] = None,
                          retrieval_negatives: int = 0) -> Dict:
    """Fine-tune SciBERT as a (claim, sentence) -> rationale cross-encoder.

    ``retrieval_negatives=k`` adds the retriever's top-k abstracts of each claim as hard negatives,
    matching what the selector sees inside the pipeline (combine with ``neg_ratio`` to bound cost).
    """
    corpus, train_claims, dev_claims = _load_data(cfg)
    base_model = base_model or cfg.rationale.base_model
    output_dir = output_dir or cfg.rationale.model_path
    args = args or TrainingArgs(max_length=128, learning_rate=2e-5, epochs=3)
    args.device = resolve_device(args.device)

    train_ret = dev_ret = None
    if retrieval_negatives > 0:
        train_ret, dev_ret = _retrieved_negatives(cfg, corpus, [train_claims, dev_claims], retrieval_negatives)
    train = build_rationale_examples(train_claims, corpus, neg_ratio=neg_ratio, seed=args.seed, retrieved=train_ret)
    dev = build_rationale_examples(dev_claims, corpus, retrieved=dev_ret)
    logger.info("Rationale examples: train=%d (%d positive), dev=%d (%d positive)", len(train),
                sum(e.label for e in train), len(dev), sum(e.label for e in dev))

    def eval_fn(y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        m = binary_prf(y, (probs[:, 1] >= 0.5).astype(int))
        best = best_threshold(y, probs[:, 1])
        return {"score": m["f1"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "best_threshold": best["threshold"], "best_threshold_f1": best["f1"]}

    return finetune_pair_classifier(
        base_model,
        [(e.claim, e.sentence) for e in train], [e.label for e in train],
        [(e.claim, e.sentence) for e in dev], [e.label for e in dev],
        output_dir, id2label={0: "NOT_RATIONALE", 1: "RATIONALE"}, args=args, eval_fn=eval_fn)


# ---------------------------------------------------------------------------- DeBERTa-v3 verifier


def nli_head_mapping(model_name: str) -> Optional[Dict[str, int]]:
    """Canonical label -> model label id if the model already has a usable 3-way NLI head, else None."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    try:
        mapping = {normalize_model_label(name): int(i) for i, name in config.id2label.items()}
    except ValueError:
        return None
    return mapping if set(mapping) == set(LABELS) else None


def train_nli_model(cfg: Config, base_model: Optional[str] = None, output_dir: Optional[str] = None,
                    args: Optional[TrainingArgs] = None, balanced: bool = True) -> Dict:
    """Fine-tune an NLI model (default: DeBERTa-v3 MNLI/FEVER/ANLI) on SciFact claim-evidence pairs.

    If the base model already has an entailment/neutral/contradiction head, it is kept (transfer
    from MNLI); otherwise a new SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE head is trained.
    """
    corpus, train_claims, dev_claims = _load_data(cfg)
    base_model = base_model or cfg.nli.model_name
    output_dir = output_dir or cfg.nli.finetuned_path
    args = args or TrainingArgs(max_length=256, learning_rate=1e-5, epochs=3, batch_size=8, grad_accum=2)
    args.device = resolve_device(args.device)

    train = build_nli_examples(train_claims, corpus, include_title=cfg.nli.include_title, seed=args.seed)
    dev = build_nli_examples(dev_claims, corpus, include_title=cfg.nli.include_title, seed=args.seed)
    logger.info("NLI examples: train=%d %s, dev=%d %s", len(train), dict(Counter(e.label for e in train)),
                len(dev), dict(Counter(e.label for e in dev)))

    mapping = nli_head_mapping(base_model)
    keep_head = mapping is not None
    if mapping is None:
        mapping = dict(LABEL2ID)
        id2label = {i: lab for lab, i in mapping.items()}
    else:
        id2label = None
        logger.info("Re-using the NLI head of %s with mapping %s", base_model, mapping)
    to_model = np.array([mapping[lab] for lab in LABELS])  # canonical index -> model index

    class_weights = None
    if balanced:
        counts = Counter(e.label for e in train)
        n_model_labels = len(mapping)
        class_weights = [1.0] * n_model_labels
        for lab in LABELS:
            class_weights[mapping[lab]] = len(train) / (len(LABELS) * counts[lab])

    def eval_fn(y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        canonical = probs[:, to_model]
        y_true = [LABELS[int(np.where(to_model == t)[0][0])] for t in y]
        y_pred = [LABELS[i] for i in canonical.argmax(1)]
        m = classification_metrics(y_true, y_pred, LABELS)
        return {"score": m["macro_f1"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    return finetune_pair_classifier(
        base_model,
        [(e.premise, e.hypothesis) for e in train], [mapping[e.label] for e in train],
        [(e.premise, e.hypothesis) for e in dev], [mapping[e.label] for e in dev],
        output_dir, id2label=id2label, args=args, eval_fn=eval_fn, class_weights=class_weights,
        keep_head=keep_head)


def model_dir_exists(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists()
