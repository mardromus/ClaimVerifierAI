"""SciFact dataset: download, loading and statistics.

SciFact (Wadden et al., EMNLP 2020) is released as JSONL files:

* ``corpus.jsonl``: ``{"doc_id": int, "title": str, "abstract": [sentences], "structured": bool}``
* ``claims_{train,dev,test}.jsonl``:
  ``{"id": int, "claim": str, "evidence": {doc_id: [{"sentences": [int], "label": "SUPPORT"|"CONTRADICT"}]},
  "cited_doc_ids": [int]}``. Test claims are unlabelled (no ``evidence`` key).

A labelled claim with no evidence is *Insufficient Evidence*. A custom corpus may use plain-text abstracts,
which are sentence-split on load.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional, Set, Union

from ..labels import CONTRADICTED, INSUFFICIENT, LABELS, SCIFACT_TO_LABEL, SUPPORTED, normalize_model_label
from ..text import split_sentences

logger = logging.getLogger(__name__)

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SPLITS = ("train", "dev", "test")
REQUIRED_FILES = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl")


# ---------------------------------------------------------------------------------------------- data classes
@dataclass
class Document:
    doc_id: int
    title: str
    sentences: List[str]
    structured: bool = False
    meta: Dict = field(default_factory=dict)  # source, journal, year, authors, doi, url (live / custom documents)

    @property
    def full_text(self) -> str:
        return f"{self.title} {' '.join(self.sentences)}".strip()

    @property
    def url(self) -> str:
        if "url" in self.meta:
            return str(self.meta["url"] or "")
        # SciFact doc_ids are Semantic Scholar corpus ids; user-supplied documents have negative ids.
        return f"https://api.semanticscholar.org/CorpusID:{self.doc_id}" if self.doc_id >= 0 else ""

    def to_dict(self) -> Dict:
        return {"doc_id": self.doc_id, "title": self.title, "abstract": list(self.sentences),
                "structured": self.structured, "url": self.url, "meta": dict(self.meta)}


@dataclass
class EvidenceSet:
    sentences: List[int]
    label: str  # SUPPORTED or CONTRADICTED

    def __post_init__(self):
        self.sentences = [int(s) for s in self.sentences]
        self.label = _to_label(self.label)


@dataclass
class Claim:
    id: int
    claim: str
    evidence: Dict[int, List[EvidenceSet]] = field(default_factory=dict)
    cited_doc_ids: List[int] = field(default_factory=list)
    labelled: bool = True

    @property
    def label(self) -> Optional[str]:
        """Claim-level gold label: the stance of its evidence, or Insufficient Evidence if it has none."""
        if not self.labelled:
            return None
        counts = Counter(es.label for sets in self.evidence.values() for es in sets)
        if not counts:
            return INSUFFICIENT
        # SciFact claims are consistent across abstracts; on a tie prefer the canonical label order.
        return max(counts, key=lambda lab: (counts[lab], -LABELS.index(lab)))

    @property
    def gold_doc_ids(self) -> List[int]:
        """Abstracts annotated as evidence (supporting or contradicting) for this claim."""
        return sorted(self.evidence)

    def doc_label(self, doc_id: int) -> Optional[str]:
        sets = self.evidence.get(doc_id)
        return sets[0].label if sets else None

    def rationale_sentences(self, doc_id: int) -> Set[int]:
        return {i for es in self.evidence.get(doc_id, []) for i in es.sentences}


def _to_label(name: str) -> str:
    if name in LABELS:
        return name
    return SCIFACT_TO_LABEL.get(name.upper(), None) or normalize_model_label(name)


# ---------------------------------------------------------------------------------------------- loading
def _read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e


def _missing(path: Path, what: str) -> FileNotFoundError:
    return FileNotFoundError(f"{what} not found at {path}. Run `python -m claimverifier download` "
                             f"(or `setup`) to fetch SciFact, or set data.corpus_path to your own corpus.")


def parse_document(obj: dict) -> Document:
    abstract = obj.get("abstract") or []
    sentences = split_sentences(abstract) if isinstance(abstract, str) else [str(s).strip() for s in abstract]
    sentences = [s for s in sentences if s]
    meta = dict(obj.get("meta") or {})
    if obj.get("url"):
        meta.setdefault("url", obj["url"])
    return Document(doc_id=int(obj["doc_id"]), title=str(obj.get("title") or "").strip(), sentences=sentences,
                    structured=bool(obj.get("structured", False)), meta=meta)


def load_corpus(data_dir: Union[str, Path], corpus_path: Optional[Union[str, Path]] = None) -> List[Document]:
    path = Path(corpus_path) if corpus_path else Path(data_dir) / "corpus.jsonl"
    if not path.exists():
        raise _missing(path, "Corpus")
    docs = [d for d in (parse_document(o) for o in _read_jsonl(path)) if d.sentences]
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate doc_id values")
    return docs


def parse_claim(obj: dict) -> Claim:
    labelled = "evidence" in obj
    evidence = {int(doc_id): [EvidenceSet(es["sentences"], es["label"]) for es in sets]
                for doc_id, sets in (obj.get("evidence") or {}).items()}
    return Claim(id=int(obj["id"]), claim=str(obj["claim"]).strip(), evidence=evidence,
                 cited_doc_ids=[int(d) for d in obj.get("cited_doc_ids") or []], labelled=labelled)


def load_claims(data_dir: Union[str, Path], split: str) -> List[Claim]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r} (expected one of {SPLITS})")
    path = Path(data_dir) / f"claims_{split}.jsonl"
    if not path.exists():
        raise _missing(path, f"SciFact {split} claims")
    return [parse_claim(o) for o in _read_jsonl(path)]


def dataset_statistics(data_dir: Union[str, Path]) -> Dict:
    data_dir = Path(data_dir)
    corpus = load_corpus(data_dir)
    stats: Dict = {"corpus_documents": len(corpus),
                   "corpus_sentences": sum(len(d.sentences) for d in corpus),
                   "avg_sentences_per_abstract": round(sum(len(d.sentences) for d in corpus) / max(1, len(corpus)), 2)}
    for split in SPLITS:
        if not (data_dir / f"claims_{split}.jsonl").exists():
            continue
        claims = load_claims(data_dir, split)
        stats[f"{split}_claims"] = len(claims)
        labelled = [c for c in claims if c.labelled]
        if labelled:
            counts = Counter(c.label for c in labelled)
            stats[f"{split}_labels"] = {lab: counts.get(lab, 0) for lab in (SUPPORTED, CONTRADICTED, INSUFFICIENT)}
            stats[f"{split}_evidence_abstracts"] = sum(len(c.evidence) for c in labelled)
    return stats


# ---------------------------------------------------------------------------------------------- download
def _safe_members(tar: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    """Yield archive members, refusing absolute paths, ``..`` components and links (path traversal)."""
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(f"Unsafe path in archive: {member.name!r}")
        yield member


def download_scifact(data_dir: Union[str, Path] = "data/scifact", force: bool = False, url: str = SCIFACT_URL) -> Path:
    """Download and extract the SciFact release into ``data_dir`` (corpus + train/dev/test claims)."""
    data_dir = Path(data_dir)
    if not force and all((data_dir / f).exists() for f in REQUIRED_FILES):
        logger.info("SciFact already present in %s", data_dir)
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        archive = Path(tmp) / "data.tar.gz"
        logger.info("Downloading SciFact from %s", url)
        with urllib.request.urlopen(url, timeout=120) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        extract = Path(tmp) / "extract"
        with tarfile.open(archive, "r:gz") as tar:
            extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(extract, members=list(_safe_members(tar)), **extra)
        source = next((p.parent for p in extract.rglob("corpus.jsonl") if not p.name.startswith("._")), None)
        if source is None:
            raise RuntimeError(f"corpus.jsonl not found in the archive from {url}")
        for name in REQUIRED_FILES:
            shutil.copy2(source / name, data_dir / name)
    logger.info("SciFact saved to %s", data_dir)
    return data_dir
