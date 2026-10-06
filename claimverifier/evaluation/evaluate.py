"""End-to-end evaluation of the claim verification pipeline on SciFact."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from ..data.scifact import Claim
from ..labels import DISPLAY_NAMES, LABELS
from .metrics import classification_metrics, retrieval_metrics, scifact_abstract_metrics, scifact_sentence_metrics

logger = logging.getLogger(__name__)


def evaluate_retrieval(retriever, claims: List[Claim], ks: Sequence[int] = (1, 3, 5, 10, 20), depth: int = 100,
                       batch_size: int = 32) -> Dict:
    """Recall@K / MRR against gold evidence abstracts (and against all cited abstracts)."""
    texts = [c.claim for c in claims]
    ranked = []
    for start in range(0, len(texts), batch_size):
        ranked.extend(retriever.rank_doc_ids(texts[start:start + batch_size], depth=depth))
    return {"evidence_retrieval": retrieval_metrics(ranked, [set(c.gold_doc_ids) for c in claims], ks),
            "evidence_retrieval_cited_docs": retrieval_metrics(ranked, [set(c.cited_doc_ids) for c in claims], ks)}


def evaluate(verifier, claims: List[Claim], ks: Sequence[int] = (1, 3, 5, 10, 20), depth: int = 100,
             batch_size: int = 32, top_k: Optional[int] = None, explain: bool = False) -> Dict:
    """Evaluate retrieval (Recall@K, MRR), verdict classification (Acc/P/R/F1) and SciFact abstract-level F1."""
    claims = [c for c in claims if c.labelled]
    texts = [c.claim for c in claims]
    t0 = time.time()

    # Retrieval quality (independent of top_k used for verification).
    ret = evaluate_retrieval(verifier.retriever, claims, ks, depth, batch_size)
    retrieval, retrieval_cited = ret["evidence_retrieval"], ret["evidence_retrieval_cited_docs"]

    # Verification.
    results = []
    for start in tqdm(range(0, len(texts), batch_size), desc="verifying", unit="batch"):
        results.extend(verifier.verify_batch(texts[start:start + batch_size], top_k=top_k, explain=explain,
                                             sentence_stance=False))
    y_true = [c.label for c in claims]
    y_pred = [r.verdict for r in results]
    classification = classification_metrics(y_true, y_pred, LABELS)
    abstract_preds = [r.abstract_predictions() for r in results]
    elapsed = time.time() - t0

    metrics = {
        "num_claims": len(claims),
        "label_distribution": {lab: y_true.count(lab) for lab in LABELS},
        "verdict_classification": classification,
        "evidence_retrieval": retrieval,
        "evidence_retrieval_cited_docs": retrieval_cited,
        "scifact_abstract_level": scifact_abstract_metrics(abstract_preds, claims),
        "scifact_sentence_selection": scifact_sentence_metrics(abstract_preds, claims),
        "seconds_total": round(elapsed, 1),
        "ms_per_claim": round(1000 * elapsed / max(len(claims), 1), 1),
        "components": results[0].components if results else {},
    }
    predictions = [{
        "id": c.id, "claim": c.claim, "gold": c.label, "predicted": r.verdict, "confidence": round(r.confidence, 4),
        "scores": {k: round(v, 4) for k, v in r.scores.items()},
        "retrieved": [d.doc_id for d in r.documents], "gold_docs": c.gold_doc_ids,
        "evidence": {str(k): v for k, v in r.abstract_predictions().items()},
    } for c, r in zip(claims, results)]
    return {"metrics": metrics, "predictions": predictions}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def markdown_report(metrics: Dict, title: str) -> str:
    c = metrics["verdict_classification"]
    r = metrics["evidence_retrieval"]
    a = metrics["scifact_abstract_level"]
    s = metrics["scifact_sentence_selection"]
    comp = metrics.get("components", {})
    lines = [f"# {title}", ""]
    if comp:
        lines += ["## Components", ""]
        lines += [f"- **{k}**: `{json.dumps(v)}`" for k, v in comp.items()]
        lines.append("")
    lines += [
        f"Claims evaluated: **{metrics['num_claims']}** "
        f"({', '.join(f'{DISPLAY_NAMES[k]}: {v}' for k, v in metrics['label_distribution'].items())}); "
        f"{metrics['ms_per_claim']} ms/claim.",
        "",
        "## Verdict classification (claim level)",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | {_pct(c['accuracy'])} |",
        f"| Macro Precision | {_pct(c['macro_precision'])} |",
        f"| Macro Recall | {_pct(c['macro_recall'])} |",
        f"| Macro F1 | {_pct(c['macro_f1'])} |",
        f"| Weighted F1 | {_pct(c['weighted_f1'])} |",
        "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    for lab, m in c["per_class"].items():
        lines.append(f"| {DISPLAY_NAMES[lab]} | {_pct(m['precision'])} | {_pct(m['recall'])} | {_pct(m['f1'])} "
                     f"| {m['support']} |")
    cm = c["confusion_matrix"]
    lines += ["", "Confusion matrix (rows = gold, columns = predicted):", "",
              "| | " + " | ".join(DISPLAY_NAMES[lab] for lab in cm["labels"]) + " |",
              "|---" * (len(cm["labels"]) + 1) + "|"]
    for lab, row in zip(cm["labels"], cm["matrix"]):
        lines.append(f"| **{DISPLAY_NAMES[lab]}** | " + " | ".join(str(v) for v in row) + " |")
    ks = sorted(int(k.split("@")[1]) for k in r if k.startswith("recall@"))
    lines += ["", f"## Evidence retrieval ({r.get('num_queries', 0)} claims with gold evidence abstracts)", "",
              "| Metric | " + " | ".join(f"@{k}" for k in ks) + " |", "|---" * (len(ks) + 1) + "|",
              "| Recall | " + " | ".join(_pct(r[f'recall@{k}']) for k in ks) + " |",
              "| Hit rate | " + " | ".join(_pct(r[f'hit@{k}']) for k in ks) + " |",
              "", f"**MRR**: {r['mrr']:.4f} (ranking depth {r.get('depth', '-')})", "",
              "## SciFact abstract-level evaluation", "",
              "| | Precision | Recall | F1 |", "|---|---|---|---|",
              f"| Label-only | {_pct(a['label_only']['precision'])} | {_pct(a['label_only']['recall'])} "
              f"| {_pct(a['label_only']['f1'])} |",
              f"| Rationalized | {_pct(a['rationalized']['precision'])} | {_pct(a['rationalized']['recall'])} "
              f"| {_pct(a['rationalized']['f1'])} |",
              f"| Sentence selection | {_pct(s['precision'])} | {_pct(s['recall'])} | {_pct(s['f1'])} |", ""]
    return "\n".join(lines)


def save_report(result: Dict, out_dir: str | Path, title: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in result["predictions"]:
            f.write(json.dumps(p) + "\n")
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(markdown_report(result["metrics"], title))
    return out_dir
