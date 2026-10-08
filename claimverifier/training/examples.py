"""Build supervised examples from SciFact for rationale selection and claim-evidence NLI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.scifact import Claim, Document
from ..labels import CONTRADICTED, INSUFFICIENT, SUPPORTED
from ..text import content_words


@dataclass
class RationaleExample:
    claim: str
    sentence: str
    label: int  # 1 = rationale (evidence) sentence, 0 = not
    claim_id: int = -1
    doc_id: int = -1
    sentence_index: int = -1
    num_sentences: int = 1


@dataclass
class NLIExample:
    premise: str
    hypothesis: str
    label: str  # canonical label
    claim_id: int = -1
    doc_id: int = -1


def format_premise(sentences: Sequence[str], title: Optional[str] = None) -> str:
    body = " ".join(s.strip() for s in sentences)
    return f"{title.strip()}. {body}" if title else body


def _candidate_docs(claim: Claim, doc_by_id: Dict[int, Document], extra: Sequence[int] = ()) -> List[int]:
    ids: List[int] = []
    for d in list(claim.evidence.keys()) + list(claim.cited_doc_ids) + list(extra):
        if d in doc_by_id and d not in ids:
            ids.append(d)
    return ids


def build_rationale_examples(claims: Sequence[Claim], corpus: Sequence[Document],
                             neg_ratio: Optional[float] = None, seed: int = 42,
                             retrieved: Optional[Dict[int, Sequence[int]]] = None) -> List[RationaleExample]:
    """One example per (claim, sentence) for every sentence of the evidence and cited abstracts.

    ``retrieved`` optionally maps claim id -> retrieved doc ids; those abstracts are added as hard
    negatives (all their sentences are non-rationale unless the abstract is gold evidence), which
    teaches the selector to reject topically similar but irrelevant papers found by the retriever.
    ``neg_ratio`` optionally down-samples negatives to at most ``neg_ratio`` x positives.
    """
    doc_by_id = {d.doc_id: d for d in corpus}
    positives, negatives = [], []
    for claim in claims:
        extra = (retrieved or {}).get(claim.id, ())
        for doc_id in _candidate_docs(claim, doc_by_id, extra):
            doc = doc_by_id[doc_id]
            rationale = claim.rationale_sentences(doc_id)
            for i, sent in enumerate(doc.sentences):
                ex = RationaleExample(claim.claim, sent, int(i in rationale), claim.id, doc_id, i, len(doc.sentences))
                (positives if ex.label else negatives).append(ex)
    if neg_ratio is not None and len(negatives) > neg_ratio * len(positives):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(negatives), size=int(neg_ratio * len(positives)), replace=False)
        negatives = [negatives[i] for i in sorted(keep)]
    return positives + negatives


def _overlap(claim_words: set, sentence: str) -> float:
    words = content_words(sentence)
    return len(claim_words & words) / (len(claim_words) or 1)


def build_nli_examples(claims: Sequence[Claim], corpus: Sequence[Document], include_title: bool = False,
                       nei_sentences: int = 2, seed: int = 42,
                       retrieved: Optional[Dict[int, Sequence[int]]] = None) -> List[NLIExample]:
    """Claim-level NLI examples (premise = evidence sentences, hypothesis = claim).

    * SUPPORTED / CONTRADICTED: each gold rationale set of an evidence abstract (and their union).
    * INSUFFICIENT_EVIDENCE: the sentences most lexically similar to the claim taken from
      abstracts that are cited but contain no evidence, from the non-rationale sentences of
      evidence abstracts and (optionally) from the ``retrieved`` abstracts (claim id -> doc ids).
      These hard negatives teach the model that topical overlap alone is not evidence.
    """
    rng = np.random.default_rng(seed)
    doc_by_id = {d.doc_id: d for d in corpus}
    examples: List[NLIExample] = []
    for claim in claims:
        claim_words = content_words(claim.claim)
        for doc_id in _candidate_docs(claim, doc_by_id, (retrieved or {}).get(claim.id, ())):
            doc = doc_by_id[doc_id]
            title = doc.title if include_title else None
            rationale = claim.rationale_sentences(doc_id)
            sets = claim.evidence.get(doc_id, [])
            seen = set()
            for es in sets:
                key = tuple(sorted(es.sentences))
                if key in seen or not key:
                    continue
                seen.add(key)
                sents = [doc.sentences[i] for i in key if i < len(doc.sentences)]
                examples.append(NLIExample(format_premise(sents, title), claim.claim, es.label, claim.id, doc_id))
            if len(sets) > 1:
                union = tuple(sorted(rationale))
                if union not in seen:
                    sents = [doc.sentences[i] for i in union if i < len(doc.sentences)]
                    examples.append(NLIExample(format_premise(sents, title), claim.claim,
                                               claim.doc_label(doc_id), claim.id, doc_id))
            # NEI: hardest non-rationale sentences of this abstract.
            candidates = [i for i in range(len(doc.sentences)) if i not in rationale]
            if not candidates:
                continue
            candidates.sort(key=lambda i: -_overlap(claim_words, doc.sentences[i]))
            n = int(rng.integers(1, nei_sentences + 1))
            chosen = sorted(candidates[:n])
            examples.append(NLIExample(format_premise([doc.sentences[i] for i in chosen], title),
                                       claim.claim, INSUFFICIENT, claim.id, doc_id))
    return examples


# ---------------------------------------------------------------------------- augmentation

# Antonym pairs for effect direction (both directions are added below).
_ANTONYMS = {
    "increase": "decrease", "increases": "decreases", "increased": "decreased", "increasing": "decreasing",
    "higher": "lower", "raises": "lowers", "raised": "lowered", "raise": "lower",
    "promotes": "inhibits", "promoted": "inhibited", "promote": "inhibit", "promoting": "inhibiting",
    "upregulates": "downregulates", "upregulated": "downregulated", "upregulation": "downregulation",
    "improves": "worsens", "improved": "worsened", "improve": "worsen",
    "enhances": "impairs", "enhanced": "impaired", "enhance": "impair",
    "accelerates": "slows", "accelerated": "slowed", "more": "less", "greater": "smaller",
    "activates": "inhibits", "activated": "inhibited", "positively": "negatively", "positive": "negative",
    "elevated": "reduced", "reduces": "increases", "reduce": "increase", "reducing": "increasing",
    "declines": "rises", "declined": "rose", "longer": "shorter", "gain": "loss", "stimulates": "suppresses", "stimulated": "suppressed",
}
ANTONYMS: Dict[str, str] = {**_ANTONYMS, **{v: k for k, v in _ANTONYMS.items() if v not in _ANTONYMS}}

# Same-direction paraphrases (label-preserving).
_SYNONYM_GROUPS = [
    ["reduces", "decreases", "lowers", "diminishes", "lessens"],
    ["reduced", "decreased", "lowered", "diminished"],
    ["reduce", "decrease", "lower", "diminish"],
    ["increases", "raises", "elevates", "boosts"],
    ["increased", "raised", "elevated", "boosted"],
    ["increase", "raise", "elevate", "boost"],
    ["inhibits", "suppresses", "blocks"],
    ["inhibited", "suppressed", "blocked"],
    ["promotes", "stimulates", "drives"],
    ["improves", "ameliorates"],
    ["impairs", "compromises"],
]
SYNONYMS: Dict[str, List[str]] = {w: [x for x in g if x != w] for g in _SYNONYM_GROUPS for w in g}
_NEGATION_RE = re.compile(r"\b(no|not|never|without|n't|unrelated|independent|none)\b", re.I)


def _swap_first(text: str, mapping: Dict[str, object], choose) -> Optional[str]:
    for m in re.finditer(r"[A-Za-z]+", text):
        word = m.group(0)
        key = word.lower()
        if key in mapping:
            new = choose(mapping[key])
            if word[0].isupper():
                new = new[0].upper() + new[1:]
            return text[:m.start()] + new + text[m.end():]
    return None


def augment_nli_examples(examples: Sequence[NLIExample], seed: int = 42) -> List[NLIExample]:
    """Direction-aware augmentation for claim verification (returns only the new examples).

    * Flip the effect direction of a *supported* claim without negation ("X reduces Y" -> "X increases Y"):
      the same evidence now contradicts it.
    * Paraphrase the direction word with a same-direction synonym ("reduces" -> "lowers"): label unchanged.
    """
    rng = np.random.default_rng(seed)
    out: List[NLIExample] = []
    for e in examples:
        if e.label == SUPPORTED and not _NEGATION_RE.search(e.hypothesis):
            flipped = _swap_first(e.hypothesis, ANTONYMS, lambda v: v)
            if flipped:
                out.append(NLIExample(e.premise, flipped, CONTRADICTED, e.claim_id, e.doc_id))
        if e.label in (SUPPORTED, CONTRADICTED):
            para = _swap_first(e.hypothesis, SYNONYMS, lambda v: v[int(rng.integers(len(v)))])
            if para:
                out.append(NLIExample(e.premise, para, e.label, e.claim_id, e.doc_id))
    return out
