"""Helpers for Hugging Face sequence(-pair) classification models (SciBERT, DeBERTa-v3, ...)."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def require_transformers():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError("PyTorch and transformers are required for this component. "
                          "Install them with `pip install -r requirements.txt`, "
                          "or use the offline configuration `configs/lite.yaml`.") from e


def load_sequence_classifier(name_or_path: str, device: str = "cpu"):
    """Load (tokenizer, model) for inference, in eval mode on ``device``."""
    require_transformers()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_pair_probs(model, tokenizer, pairs: Sequence[Tuple[str, str]], batch_size: int = 16,
                       max_length: int = 512, device: str = "cpu") -> np.ndarray:
    """Softmax probabilities (n_pairs, num_labels) for (text_a, text_b) pairs.

    Pairs are processed in length-sorted batches (much faster with dynamic padding)
    and returned in the original order.
    """
    import torch

    num_labels = model.config.num_labels
    if len(pairs) == 0:
        return np.zeros((0, num_labels), dtype=np.float32)
    order = np.argsort([-(len(a) + len(b)) for a, b in pairs], kind="stable")
    out = np.zeros((len(pairs), num_labels), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            idx = order[start:start + batch_size]
            batch = [pairs[i] for i in idx]
            enc = tokenizer([a for a, _ in batch], [b for _, b in batch], truncation=True,
                            max_length=max_length, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            out[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
    return out


def length_grouped_batches(lengths: List[int], batch_size: int, rng: np.random.Generator,
                           mega_factor: int = 50) -> List[List[int]]:
    """Shuffle, then sort by length inside mega-batches so batches have similar lengths."""
    idx = rng.permutation(len(lengths))
    mega = batch_size * mega_factor
    batches = []
    for start in range(0, len(idx), mega):
        chunk = sorted(idx[start:start + mega], key=lambda i: -lengths[i])
        batches.extend(chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size))
    order = rng.permutation(len(batches))
    return [list(map(int, batches[i])) for i in order]
