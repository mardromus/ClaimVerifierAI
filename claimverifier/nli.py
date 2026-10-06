"""Natural Language Inference between retrieved evidence (premise) and the claim (hypothesis).

Every model returns probabilities in canonical order ``[SUPPORTED, CONTRADICTED, INSUFFICIENT]``.

* :class:`TransformerNLI` - any Hugging Face NLI cross-encoder; default
  ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` (zero-shot), or a DeBERTa-v3 checkpoint
  fine-tuned on SciFact with ``claimverifier train-nli``.
* :class:`LiteNLI` - scikit-learn stance classifier (support vs. contradict) over n-gram and
  lexical-cue features, trained on SciFact (offline fallback; much weaker than DeBERTa-v3). In
  this mode relevance (i.e. "insufficient evidence") is left to the rationale selector.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .labels import INSUFFICIENT, LABEL2ID, LABELS, normalize_model_label
from .text import content_words, cue_counts, numbers

logger = logging.getLogger(__name__)


class NLIModel:
    method = "base"

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """``pairs`` = [(premise, hypothesis), ...] -> array (n, 3) in canonical label order."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"method": self.method}


class TransformerNLI(NLIModel):
    method = "transformer"

    def __init__(self, name_or_path: str, device: str = "cpu", batch_size: int = 16, max_length: int = 512):
        from .hf_utils import load_sequence_classifier

        self.name_or_path = name_or_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer, self.model = load_sequence_classifier(name_or_path, device)
        id2label = self.model.config.id2label
        # Projection from model label ids onto canonical labels (handles 2- and 3-way heads in any order).
        self.projection = np.zeros((len(id2label), len(LABELS)), dtype=np.float32)
        for i, name in id2label.items():
            self.projection[int(i), LABEL2ID[normalize_model_label(name)]] = 1.0
        logger.info("Loaded NLI model %s with labels %s", name_or_path, dict(id2label))

    def predict(self, pairs):
        from .hf_utils import predict_pair_probs

        probs = predict_pair_probs(self.model, self.tokenizer, list(pairs), self.batch_size,
                                   self.max_length, self.device)
        return probs @ self.projection

    def describe(self):
        return {"method": self.method, "model": self.name_or_path}


class LiteNLI(NLIModel):
    """Logistic regression over claim-evidence cross features (lexical overlap, number agreement,
    negation and increase/decrease cue (dis)agreement).

    With ``stance_only=True`` (default) the model is trained on SUPPORTED / CONTRADICTED evidence only
    and its INSUFFICIENT probability is 0; the pipeline's rationale gate decides relevance.
    (Grouped 5-fold CV on SciFact train showed that adding TF-IDF n-gram features over-fits.)
    """

    method = "lite"
    FILE = "nli_lite.joblib"

    def __init__(self, stance_only: bool = True):
        self.stance_only = stance_only
        self.scaler = None
        self.clf = None

    @staticmethod
    def cross_features(premise: str, claim: str) -> List[float]:
        c, p = cue_counts(claim), cue_counts(premise)
        c_words, p_words = content_words(claim), content_words(premise)
        shared = c_words & p_words
        c_nums, p_nums = numbers(claim), numbers(premise)
        lower = premise.lower()
        no_effect = any(k in lower for k in ("no significant", "not significant", "did not", "was not",
                                              "were not", "no difference", "no association", "no effect",
                                              "not associated", "no evidence", "nonsignificant", "non-significant"))
        c_neg, p_neg = c["neg"] > 0, p["neg"] > 0 or no_effect
        c_up, c_down = c["inc"] > c["dec"], c["dec"] > c["inc"]
        p_up, p_down = p["inc"] > p["dec"], p["dec"] > p["inc"]
        return [
            len(shared) / (len(c_words) or 1),
            len(shared) / (len(c_words | p_words) or 1),
            float(bool(c_nums)), float(bool(c_nums & p_nums)), float(bool(c_nums - p_nums)),
            float(c_neg), float(p_neg), float(c_neg != p_neg), float(c_neg and p_neg), float(no_effect),
            float(c_up), float(c_down), float(p_up), float(p_down),
            float(c_up and p_down), float(c_down and p_up), float(c_up and p_up), float(c_down and p_down),
            float((c_up and p_down) or (c_down and p_up)) * (1.0 - float(c_neg != p_neg)),
            math.log1p(len(premise.split())),
        ]

    def _features(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        dense = np.asarray([self.cross_features(p, h) for p, h in pairs], dtype=np.float64)
        return self.scaler.transform(dense)

    def fit(self, pairs: Sequence[Tuple[str, str]], labels: Sequence[str], C: float = 1.0, seed: int = 42) -> "LiteNLI":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if self.stance_only:
            kept = [(pair, lab) for pair, lab in zip(pairs, labels) if lab != INSUFFICIENT]
            pairs, labels = [k[0] for k in kept], [k[1] for k in kept]
        dense = np.asarray([self.cross_features(p, h) for p, h in pairs], dtype=np.float64)
        self.scaler = StandardScaler().fit(dense)
        self.clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
        self.clf.fit(self.scaler.transform(dense), [LABEL2ID[lab] for lab in labels])
        return self

    def predict(self, pairs):
        if self.clf is None:
            raise RuntimeError("LiteNLI is not trained")
        if len(pairs) == 0:
            return np.zeros((0, len(LABELS)), dtype=np.float32)
        probs = self.clf.predict_proba(self._features(list(pairs)))
        out = np.zeros((len(pairs), len(LABELS)), dtype=np.float32)
        out[:, self.clf.classes_] = probs
        return out

    def describe(self):
        return {"method": self.method, "stance_only": self.stance_only}

    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / self.FILE)

    @classmethod
    def load(cls, directory: Path) -> "LiteNLI":
        import joblib

        return joblib.load(directory / cls.FILE)


def create_nli_model(cfg, artifacts_dir: str | Path, device: str = "cpu") -> NLIModel:
    """Instantiate the configured NLI model, falling back along ``cfg.nli.fallbacks``."""
    ncfg = cfg.nli
    methods = [ncfg.method] + [m for m in ncfg.fallbacks if m != ncfg.method]
    lite_dir = Path(artifacts_dir) / "lite"
    errors = []
    for method in methods:
        try:
            if method == "transformer":
                path = ncfg.finetuned_path
                source = path if path and (Path(path) / "config.json").exists() else ncfg.model_name
                model = TransformerNLI(source, device, ncfg.batch_size, ncfg.max_length)
            elif method == "lite":
                if not (lite_dir / LiteNLI.FILE).exists():
                    raise FileNotFoundError(f"no lite NLI model in {lite_dir} (run `python -m claimverifier train-lite`)")
                model = LiteNLI.load(lite_dir)
            else:
                raise ValueError(f"unknown NLI method {method!r}")
        except Exception as e:  # noqa: BLE001 - fall back to the next method
            errors.append(f"{method}: {e}")
            logger.warning("NLI model %r unavailable: %s", method, e)
            continue
        if method != ncfg.method:
            logger.warning("Using fallback NLI model %r", method)
        return model
    raise RuntimeError("No NLI model available: " + "; ".join(errors))
