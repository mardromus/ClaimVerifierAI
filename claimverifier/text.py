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


def split_sentences(text: str) -> List[str]:
    """Simple rule-based sentence splitter (SciFact abstracts are already split)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def cue_counts(text: str) -> dict:
    tokens = _TOKEN_RE.findall(text.lower())
    return {
        "neg": sum(t in NEGATIONS for t in tokens) + text.lower().count("n't"),
        "inc": sum(t in INCREASE for t in tokens),
        "dec": sum(t in DECREASE for t in tokens),
    }
