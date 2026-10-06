"""Canonical verdict labels shared by every component.

All models output probabilities in the fixed order of :data:`LABELS`:
``[SUPPORTED, CONTRADICTED, INSUFFICIENT]``.
"""

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

LABELS = [SUPPORTED, CONTRADICTED, INSUFFICIENT]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

DISPLAY_NAMES = {
    SUPPORTED: "Supported",
    CONTRADICTED: "Contradicted",
    INSUFFICIENT: "Insufficient Evidence",
}

# SciFact annotations use SUPPORT / CONTRADICT; claims without evidence are NEI.
SCIFACT_TO_LABEL = {
    "SUPPORT": SUPPORTED,
    "CONTRADICT": CONTRADICTED,
    "NEI": INSUFFICIENT,
}


def normalize_model_label(name: str) -> str:
    """Map a label name from an NLI / fact-checking model config to a canonical label.

    Handles MNLI-style names (entailment / neutral / contradiction), SciFact-style
    names (SUPPORT / CONTRADICT / NEI) and FEVER-style names (SUPPORTS / REFUTES /
    NOT ENOUGH INFO).
    """
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key.startswith("entail") or key.startswith("support"):
        return SUPPORTED
    if key.startswith("contradict") or key.startswith("refute"):
        return CONTRADICTED
    if key in {"neutral", "nei", "not_entailment", "not_enough_info", "insufficient_evidence"}:
        return INSUFFICIENT
    raise ValueError(f"Unrecognised NLI label name: {name!r}")
