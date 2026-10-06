"""Fine-tuning of transformer sequence-pair classifiers.

Used for
* the SciBERT rationale selector (claim, sentence) -> {not rationale, rationale}, and
* the DeBERTa-v3 claim verifier (evidence, claim) -> {entailment, neutral, contradiction}.

A plain PyTorch loop (AdamW + linear warm-up/decay, length-grouped batches, best-checkpoint
selection on the dev split) keeps the dependency surface small and works on CPU, CUDA and MPS.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..hf_utils import length_grouped_batches, predict_pair_probs, require_transformers

logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    grad_accum: int = 1
    max_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_batch_size: int = 64
    seed: int = 42
    device: str = "cpu"
    log_every: int = 25
    max_steps: Optional[int] = None  # stop early (useful for smoke tests)


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finetune_pair_classifier(
    base_model: str,
    train_pairs: Sequence[Tuple[str, str]],
    train_labels: Sequence[int],
    dev_pairs: Sequence[Tuple[str, str]],
    dev_labels: Sequence[int],
    output_dir: str | Path,
    id2label: Optional[Dict[int, str]] = None,
    args: Optional[TrainingArgs] = None,
    eval_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    class_weights: Optional[List[float]] = None,
    keep_head: bool = False,
) -> Dict:
    """Fine-tune ``base_model`` and save the best checkpoint (by ``eval_fn(...)['score']``) to ``output_dir``.

    ``keep_head=True`` re-uses the model's existing classification head (e.g. an MNLI head) and its
    label ids; otherwise a new head with ``id2label`` is initialised.
    """
    require_transformers()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    args = args or TrainingArgs()
    _set_seed(args.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if keep_head:
        model = AutoModelForSequenceClassification.from_pretrained(base_model)
    else:
        assert id2label is not None, "id2label is required when training a new head"
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(id2label), id2label=id2label,
            label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True)
    model.to(args.device)

    lengths = [len(a) + len(b) for a, b in train_pairs]
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = math.ceil(math.ceil(len(train_pairs) / args.batch_size) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    weight = torch.tensor(class_weights, dtype=torch.float32, device=args.device) if class_weights else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    labels_t = np.asarray(train_labels, dtype=np.int64)

    history, best_score, best_epoch, step = [], -math.inf, -1, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        batches = length_grouped_batches(lengths, args.batch_size, rng)
        for b_idx, batch_idx in enumerate(batches):
            batch = [train_pairs[i] for i in batch_idx]
            enc = tokenizer([a for a, _ in batch], [b for _, b in batch], truncation=True,
                            max_length=args.max_length, padding=True, return_tensors="pt").to(args.device)
            logits = model(**enc).logits
            target = torch.as_tensor(labels_t[batch_idx], device=args.device)
            loss = loss_fn(logits.float(), target) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            n_batches += 1
            if (b_idx + 1) % args.grad_accum == 0 or b_idx == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    logger.info("epoch %d step %d/%d loss %.4f (%.0fs)", epoch, step, total_steps,
                                running / n_batches, time.time() - start)
                if args.max_steps and step >= args.max_steps:
                    break
        model.eval()
        probs = predict_pair_probs(model, tokenizer, dev_pairs, args.eval_batch_size, args.max_length, args.device)
        metrics = eval_fn(np.asarray(dev_labels), probs) if eval_fn else {}
        score = metrics.get("score", -running / max(n_batches, 1))
        record = {"epoch": epoch, "train_loss": running / max(n_batches, 1), **metrics,
                  "elapsed_sec": round(time.time() - start, 1)}
        history.append(record)
        logger.info("epoch %d done: %s", epoch, json.dumps(record))
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("saved new best checkpoint to %s (score %.4f)", output_dir, score)
        if args.max_steps and step >= args.max_steps:
            break

    info = {"base_model": base_model, "best_epoch": best_epoch, "best_score": best_score,
            "train_examples": len(train_pairs), "dev_examples": len(dev_pairs),
            "args": asdict(args), "history": history}
    with open(output_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info
