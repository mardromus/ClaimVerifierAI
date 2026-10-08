"""Download pretrained checkpoints that are not on the Hugging Face Hub.

``verisci``: the RoBERTa-large claim verifier released by the SciFact authors (Wadden et al., 2020), trained on
FEVER and then SciFact for SUPPORT / CONTRADICT / NOT ENOUGH INFO label prediction. It is a much more robust
verifier than a model fine-tuned on SciFact alone, and it runs on CPU. The published config has generic label
names, so they are rewritten (CONTRADICT, NOT_ENOUGH_INFO, SUPPORT, as in the SciFact code) after download.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .data.scifact import _safe_members

logger = logging.getLogger(__name__)

CHECKPOINTS = {
    "verisci": {
        "url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/models/label_roberta_large_fever_scifact.tar.gz",
        "id2label": {"0": "CONTRADICT", "1": "NOT_ENOUGH_INFO", "2": "SUPPORT"},
        "description": "RoBERTa-large verifier (FEVER + SciFact) from the SciFact authors, 1.3 GB",
    },
}


def fix_labels(model_dir: str | Path, id2label: dict) -> None:
    path = Path(model_dir) / "config.json"
    config = json.loads(path.read_text())
    config["id2label"] = dict(id2label)
    config["label2id"] = {v: int(k) for k, v in id2label.items()}
    path.write_text(json.dumps(config, indent=2))


def fetch_checkpoint(name: str = "verisci", dest: str | Path = "models/verisci-roberta-large", force: bool = False) -> Path:
    if name not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint {name!r} (available: {sorted(CHECKPOINTS)})")
    spec = CHECKPOINTS[name]
    dest = Path(dest)
    if (dest / "config.json").exists() and not force:
        logger.info("%s already present in %s", name, dest)
        fix_labels(dest, spec["id2label"])
        return dest
    with tempfile.TemporaryDirectory(dir=dest.parent if dest.parent.exists() else None) as tmp:
        archive = Path(tmp) / "model.tar.gz"
        logger.info("Downloading %s (%s) from %s", name, spec["description"], spec["url"])
        with urllib.request.urlopen(spec["url"], timeout=120) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        extract = Path(tmp) / "extract"
        with tarfile.open(archive, "r:gz") as tar:
            extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(extract, members=list(_safe_members(tar)), **extra)
        source = next(p.parent for p in extract.rglob("config.json"))
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
    fix_labels(dest, spec["id2label"])
    logger.info("Saved %s to %s", name, dest)
    return dest
