"""Aggregate document-level NLI predictions into a claim-level verdict with confidence scores.

For every retrieved abstract *d* we have
  r_d  - relevance: the highest rationale probability among its selected evidence sentences
  p_d  - NLI probabilities (support s_d, contradict c_d, insufficient n_d) for its evidence

Each abstract is gated by its relevance: g_d = min(1, r_d / tau), where tau is the rationale
threshold, so abstracts with real evidence count fully and weaker ones proportionally less.
Evidence strength for each stance is the strongest gated signal across documents:

  S = max_d g_d * s_d          C = max_d g_d * c_d

These are combined as two independent "detectors":

  P(supported)    = S (1 - C) + S C * S / (S + C)
  P(contradicted) = C (1 - S) + S C * C / (S + C)
  P(insufficient) = w (1 - S)(1 - C)                (w = nei_weight, then renormalised)

so the three scores sum to one, a claim is only Supported / Contradicted when some abstract
provides relevant evidence that the NLI model is confident about, and conflicting strong
evidence is split between the two stances (and flagged as ``mixed``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .labels import CONTRADICTED, INSUFFICIENT, LABEL2ID, LABELS, SUPPORTED


@dataclass
class Verdict:
    label: str
    confidence: float
    scores: Dict[str, float]
    support_strength: float
    contradict_strength: float
    mixed: bool


def aggregate(relevances: Sequence[float], doc_probs: Sequence[Sequence[float]], threshold: float = 0.5,
              nei_weight: float = 1.0, min_relevance: float = 0.0) -> Verdict:
    rel = np.asarray(relevances, dtype=np.float64)
    probs = np.asarray(doc_probs, dtype=np.float64).reshape(-1, len(LABELS))
    mask = rel >= min_relevance
    if mask.any():
        gate = np.minimum(1.0, rel[mask] / max(threshold, 1e-6))
        probs = probs[mask]
        s = float(np.max(gate * probs[:, LABEL2ID[SUPPORTED]]))
        c = float(np.max(gate * probs[:, LABEL2ID[CONTRADICTED]]))
    else:
        s = c = 0.0
    both = s * c
    share_s = s / (s + c) if (s + c) > 0 else 0.5
    sup = s * (1 - c) + both * share_s
    con = c * (1 - s) + both * (1 - share_s)
    nei = nei_weight * (1 - s) * (1 - c)
    total = sup + con + nei
    if total <= 0:  # only possible with nei_weight == 0 and no evidence at all
        sup, con, nei, total = 0.0, 0.0, 1.0, 1.0
    scores = {SUPPORTED: sup / total, CONTRADICTED: con / total, INSUFFICIENT: nei / total}
    label = max(LABELS, key=lambda lab: (scores[lab], -LABELS.index(lab)))
    return Verdict(label=label, confidence=scores[label], scores=scores, support_strength=s,
                   contradict_strength=c, mixed=bool(s >= 0.5 and c >= 0.5))


def doc_stance(probs: Sequence[float]) -> str:
    return LABELS[int(np.argmax(probs))]


def stance_counts(stances: List[str]) -> Dict[str, int]:
    return {lab: sum(s == lab for s in stances) for lab in LABELS}
