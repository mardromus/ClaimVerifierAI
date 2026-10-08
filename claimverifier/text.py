"""Lightweight text utilities (tokenisation, sentence splitting, lexical cues)."""

from __future__ import annotations

import re
from typing import List, Set

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

# Negation cues are kept out of the stop-word list: they matter for contradiction.
NEGATIONS = {
    "no", "not", "none", "never", "neither", "nor", "without", "lack", "lacks", "lacking",
    "absent", "absence", "cannot", "fail", "fails", "failed", "unable", "non", "nothing",
    "insignificant", "unaffected", "unchanged", "independent",
}
INCREASE = {
    "increase", "increases", "increased", "increasing", "higher", "high", "elevated", "elevate",
    "elevates", "raise", "raises", "raised", "more", "greater", "enhance", "enhances", "enhanced",
    "promote", "promotes", "promoted", "upregulate", "upregulates", "upregulated", "upregulation",
    "induce", "induces", "induced", "activate", "activates", "activated", "improve", "improves",
    "improved", "accelerate", "accelerates", "accelerated", "stimulate", "stimulates", "positive",
    "positively", "gain", "larger", "longer", "exacerbates", "worsens", "boosts",
}
DECREASE = {
    "decrease", "decreases", "decreased", "decreasing", "lower", "low", "reduce", "reduces",
    "reduced", "reduction", "less", "fewer", "inhibit", "inhibits", "inhibited", "inhibition",
    "suppress", "suppresses", "suppressed", "downregulate", "downregulates", "downregulated",
    "downregulation", "impair", "impairs", "impaired", "attenuate", "attenuates", "attenuated",
    "block", "blocks", "blocked", "prevent", "prevents", "prevented", "diminish", "diminishes",
    "diminished", "decline", "declines", "declined", "negative", "negatively", "loss", "smaller",
    "shorter", "slows", "protects", "protective", "abolishes", "abolished",
}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - NEGATIONS - INCREASE - DECREASE)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def content_words(text: str) -> Set[str]:
    return {t for t in tokenize(text) if len(t) > 1}


def numbers(text: str) -> Set[str]:
    return set(_NUMBER_RE.findall(text))


_ABBREVIATIONS = {
    "e.g.", "i.e.", "al.", "vs.", "fig.", "figs.", "approx.", "ca.", "dr.", "no.", "nos.", "ref.", "refs.",
    "cf.", "resp.", "eq.", "eqs.", "vol.", "pp.", "st.", "mr.", "mrs.", "ms.", "prof.", "inc.", "ltd.", "co.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.", "sp.", "spp.",
}
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"<h\d[^>]*>(.*?)</h\d>", flags=re.I | re.S)


def split_sentences(text: str) -> List[str]:
    """Rule-based sentence splitter for scientific abstracts (SciFact abstracts are already split).

    Splits after ``.``, ``!`` or ``?`` followed by whitespace and an upper-case letter, digit or bracket,
    but not after common abbreviations (``e.g.``, ``et al.``, ``Fig.``, ``vs.`` ...). Single capital letters
    are not treated as initials because biomedical sentences often end in one ("vitamin D.", "hepatitis B.").
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    sentences: List[str] = []
    for piece in pieces:
        if sentences:
            last_word = sentences[-1].rsplit(" ", 1)[-1].lower()
            if last_word in _ABBREVIATIONS:
                sentences[-1] = f"{sentences[-1]} {piece}"
                continue
        sentences.append(piece)
    return sentences


def clean_abstract_html(text: str) -> str:
    """Strip HTML/JATS markup from an abstract; section headings become ``Heading:`` prefixes."""
    import html as _html

    text = _HEADING_RE.sub(lambda m: f" {m.group(1).strip().rstrip(':')}: ", text or "")
    text = re.sub(r"</?(p|div|sec|br)[^>]*>", " ", text, flags=re.I)
    text = _HTML_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", _html.unescape(text)).strip()


def cue_counts(text: str) -> dict:
    tokens = _TOKEN_RE.findall(text.lower())
    return {
        "neg": sum(t in NEGATIONS for t in tokens) + text.lower().count("n't"),
        "inc": sum(t in INCREASE for t in tokens),
        "dec": sum(t in DECREASE for t in tokens),
    }


_CLAIM_CUES = re.compile(
    r"\b(increas\w*|decreas\w*|reduc\w*|lower\w*|rais\w*|caus\w*|prevent\w*|improv\w*|worsen\w*|associat\w*|linked|"
    r"link|risk|effect\w*|leads?|led|results? in|protect\w*|inhibit\w*|promot\w*|induc\w*|correlat\w*|"
    r"predict\w*|treat\w*|cure\w*|boost\w*|block\w*|trigger\w*|contribut\w*|impair\w*|enhanc\w*|"
    r"is required|are required|necessary|essential|important for|is not|are not|does not|do not|no effect)\b",
    re.I,
)
_HEDGES = re.compile(r"\b(we|our|this study|in this paper|here we|methods?|participants were|were recruited)\b", re.I)


def extract_claims(text: str, max_claims: int = 20) -> List[dict]:
    """Pick check-worthy factual statements out of free text (news article, abstract, report).

    Sentences are scored by causal / effect-direction cues, numbers and length; questions, very short
    sentences and study-procedure sentences ("we recruited ...") are down-weighted.
    """
    out = []
    for i, sent in enumerate(split_sentences(text)):
        words = sent.split()
        if len(words) < 5 or sent.endswith("?"):
            continue
        cues = len(_CLAIM_CUES.findall(sent))
        score = min(cues, 3) * 1.0 + 0.5 * bool(numbers(sent)) + (0.5 if 8 <= len(words) <= 40 else 0.0)
        score -= 1.0 * bool(_HEDGES.search(sent))
        if cues and score > 1.0:
            out.append({"index": i, "claim": sent.strip(), "score": round(score, 2)})
    out.sort(key=lambda c: (-c["score"], c["index"]))
    return sorted(out[:max_claims], key=lambda c: c["index"])
