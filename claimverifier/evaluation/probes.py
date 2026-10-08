"""Robustness probes for the stance (NLI) model.

24 hand-written premise/claim pairs in plain language (12 supported, 12 contradicted) that differ only in effect
direction, negation or paraphrase ("reduced the risk" vs "reduces the risk" / "increases the risk"). A verifier
that relies on dataset artifacts instead of meaning fails them even when its benchmark score looks good.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..labels import LABELS

PROBES_PATH = Path(__file__).resolve().parents[1] / "resources" / "stance_probes.jsonl"


def load_probes(path: Path = PROBES_PATH) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_probes(nli, probes: List[Dict] | None = None) -> Dict:
    """Accuracy of the support-vs-contradict decision (the INSUFFICIENT probability is ignored)."""
    probes = probes or load_probes()
    probs = nli.predict([(p["premise"], p["claim"]) for p in probes])
    pred = [LABELS[int(i)] for i in np.asarray(probs)[:, :2].argmax(1)]
    rows = [{**p, "predicted": y, "probs": {lab: round(float(v), 4) for lab, v in zip(LABELS, pr)}}
            for p, y, pr in zip(probes, pred, probs)]
    correct = sum(r["predicted"] == r["label"] for r in rows)
    return {"accuracy": correct / len(rows), "correct": correct, "total": len(rows), "rows": rows}
