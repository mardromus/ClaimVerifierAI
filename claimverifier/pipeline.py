"""End-to-end claim verification pipeline.

  claim ──► retrieval (Sentence-BERT + FAISS, fused with BM25)
        ──► rationale selection (SciBERT) over each retrieved abstract
        ──► NLI (DeBERTa-v3) between selected evidence and the claim
        ──► aggregation into Supported / Contradicted / Insufficient Evidence + confidence
        ──► explanation with citations (Llama 3 / Qwen 2.5 Instruct, RAG prompt)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .aggregation import aggregate, doc_stance
from .config import Config, resolve_device
from .data.scifact import Document, load_corpus
from .explain import EvidenceForPrompt, ExplanationGenerator
from .labels import CONTRADICTED, DISPLAY_NAMES, INSUFFICIENT, LABELS, SUPPORTED
from .nli import NLIModel, create_nli_model
from .rationale import RationaleSelector, create_rationale_selector, select_sentences
from .retrieval.retriever import RetrievedDoc, Retriever
from .training.examples import format_premise

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSentence:
    index: int
    text: str
    rationale_score: float
    stance: str
    stance_probs: Dict[str, float]


@dataclass
class DocumentResult:
    doc_id: int
    title: str
    url: str
    citation: int
    rank: int
    retrieval_score: float
    dense_score: float
    bm25_score: float
    relevance: float           # best rationale score among the selected sentences
    has_evidence: bool         # at least one sentence passed the rationale threshold
    evidence_weight: float     # min(1, relevance / threshold): how much this abstract counts in the verdict
    stance: str                # NLI label for this abstract's evidence
    stance_probs: Dict[str, float]
    evidence: List[EvidenceSentence]
    sentences: List[str]
    sentence_scores: List[float]
    retrieval_rank: int = 0    # rank given by the retriever before re-ranking
    meta: Dict = field(default_factory=dict)  # journal, year, authors, source, ... (live / custom documents)


@dataclass
class VerificationResult:
    claim: str
    verdict: str
    verdict_display: str
    confidence: float
    scores: Dict[str, float]
    support_strength: float
    contradict_strength: float
    mixed_evidence: bool
    documents: List[DocumentResult]
    explanation: str = ""
    explanation_backend: str = ""
    citations: List[int] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    components: Dict[str, Dict] = field(default_factory=dict)
    source: str = "corpus"
    candidates_scanned: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    def abstract_predictions(self) -> Dict[int, Dict]:
        """SciFact-style abstract-level predictions (abstracts predicted to support / contradict)."""
        return {d.doc_id: {"label": d.stance, "sentences": [e.index for e in d.evidence]}
                for d in self.documents if d.has_evidence and d.stance != INSUFFICIENT}


# Abstracts whose evidence weight reaches this value are shown as (partial) evidence.
MIN_SHOWN_WEIGHT = 0.5


def evidence_strength(d: "DocumentResult") -> float:
    """How strongly an abstract pushes the verdict towards Supported or Contradicted."""
    return d.evidence_weight * max(d.stance_probs[SUPPORTED], d.stance_probs[CONTRADICTED])


def shown_documents(result: "VerificationResult") -> List["DocumentResult"]:
    """Abstracts that contribute to the verdict, strongest first."""
    docs = [d for d in result.documents if d.evidence_weight >= MIN_SHOWN_WEIGHT]
    return sorted(docs, key=lambda d: -evidence_strength(d))


def _preview(r: RetrievedDoc) -> Dict:
    return {"doc_id": r.doc.doc_id, "title": r.doc.title, "url": r.doc.url, "rank": r.rank,
            "dense_score": round(r.dense_score, 4), "bm25_score": round(r.bm25_score, 3), "meta": dict(r.doc.meta),
            "num_sentences": len(r.doc.sentences)}


def _probs_dict(p: Sequence[float]) -> Dict[str, float]:
    return {lab: float(v) for lab, v in zip(LABELS, p)}


class ClaimVerifier:
    def __init__(self, cfg: Config, corpus: List[Document], retriever: Retriever, rationale: RationaleSelector,
                 nli: NLIModel, explainer: Optional[ExplanationGenerator] = None):
        self.cfg = cfg
        self.corpus = corpus
        self.retriever = retriever
        self.rationale = rationale
        self.nli = nli
        self.explainer = explainer or ExplanationGenerator(cfg.explanation)
        self.doc_by_id = {d.doc_id: d for d in corpus}
        self.calibration: Optional[Dict] = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: Config, build_index_if_missing: bool = True) -> "ClaimVerifier":
        device = resolve_device(cfg.device)
        logger.info("Loading ClaimVerifier '%s' on %s", cfg.name, device)
        corpus = load_corpus(cfg.data.data_dir, cfg.data.corpus_path)
        if Retriever.exists(cfg.artifacts_dir):
            retriever = Retriever.load(cfg.retrieval, corpus, cfg.artifacts_dir, device)
        elif build_index_if_missing:
            logger.info("No index found in %s - building it now", cfg.artifacts_dir)
            retriever = Retriever.build(cfg.retrieval, corpus, cfg.artifacts_dir, device, cfg.seed)
        else:
            raise FileNotFoundError(f"No index in {cfg.artifacts_dir}; run `python -m claimverifier index`.")
        rationale = create_rationale_selector(cfg, retriever.embedder, cfg.artifacts_dir, device)
        nli = create_nli_model(cfg, cfg.artifacts_dir, device)
        explainer = ExplanationGenerator(cfg.explanation, device)
        verifier = cls(cfg, corpus, retriever, rationale, nli, explainer)
        verifier.apply_calibration()
        return verifier

    def apply_calibration(self) -> Optional[Dict]:
        """Load calibrated threshold / nei_weight if they were computed for the current models."""
        from .evaluation.calibrate import load_calibration

        self.calibration = None
        if not self.cfg.aggregation.use_calibration:
            return None
        cal = load_calibration(self.cfg.artifacts_dir)
        if cal is None:
            return None
        current = {"rationale": self.rationale.describe(), "nli": self.nli.describe()}
        stored = {k: cal.get("components", {}).get(k) for k in current}
        if stored != current:
            logger.warning("Ignoring %s/calibration.json: it was computed for different models %s",
                           self.cfg.artifacts_dir, stored)
            return None
        self.cfg.rationale.threshold = float(cal["threshold"])
        self.cfg.aggregation.nei_weight = float(cal["nei_weight"])
        self.calibration = cal
        logger.info("Using calibrated threshold=%.2f nei_weight=%.2f (from %s split)", cal["threshold"],
                    cal["nei_weight"], cal.get("split"))
        return cal

    def components(self) -> Dict[str, Dict]:
        return {
            "retrieval": {**self.retriever.embedder.describe(), "index": self.cfg.retrieval.index_type,
                          "hybrid_bm25": self.retriever.bm25 is not None,
                          **({"rrf_weights": {"dense": self.cfg.retrieval.dense_weight,
                                              "bm25": self.cfg.retrieval.bm25_weight}}
                             if self.retriever.bm25 is not None else {})},
            "rationale": self.rationale.describe(),
            "nli": self.nli.describe(),
            "decision": {"threshold": self.cfg.rationale.threshold, "nei_weight": self.cfg.aggregation.nei_weight,
                         "calibrated": self.calibration is not None},
            "explanation": {"backend": self.explainer.backend_name},
        }

    # ------------------------------------------------------------------ public API
    def verify(self, claim: str, top_k: Optional[int] = None, explain: bool = True) -> VerificationResult:
        return self.verify_batch([claim], top_k=top_k, explain=explain)[0]

    def verify_against(self, claim: str, documents: List[Document], explain: bool = True) -> VerificationResult:
        """Verify a claim against user-supplied abstracts instead of the indexed corpus."""
        retrieved = [RetrievedDoc(doc=d, rank=i + 1, score=0.0, dense_score=0.0) for i, d in enumerate(documents)]
        result = self._analyze([claim], [retrieved], explain=explain, sentence_stance=True)[0]
        result.source, result.candidates_scanned = "custom", len(documents)
        return result

    def verify_batch(self, claims: List[str], top_k: Optional[int] = None, explain: bool = False,
                     sentence_stance: bool = True) -> List[VerificationResult]:
        retrieved, scores, timings = self.retrieve_and_score(claims, top_k)
        results = self._analyze(claims, retrieved, explain=explain, sentence_stance=sentence_stance,
                                rationale_scores=scores)
        depth = self.retrieval_depth(top_k)
        for r in results:
            r.timings_ms = {**timings, **r.timings_ms}
            r.candidates_scanned = depth
        return results

    # ------------------------------------------------------------------ retrieval + re-ranking
    def retrieval_depth(self, top_k: Optional[int] = None) -> int:
        return max(top_k or self.cfg.retrieval.top_k, self.cfg.retrieval.rerank_depth)

    def retrieve_and_score(self, claims: List[str], top_k: Optional[int] = None):
        """Retrieve abstracts, score their sentences with the rationale selector and (optionally) re-rank.

        Returns ``(retrieved, rationale_scores, timings_ms)`` with ``top_k`` abstracts per claim.
        """
        top_k = top_k or self.cfg.retrieval.top_k
        n = max(len(claims), 1)
        t0 = time.perf_counter()
        retrieved = self.retriever.search_batch(claims, self.retrieval_depth(top_k))
        t1 = time.perf_counter()
        scores = self.score_rationales(claims, retrieved)
        retrieved, scores = self.rerank(retrieved, scores, top_k)
        t2 = time.perf_counter()
        return retrieved, scores, {"retrieval": round((t1 - t0) * 1000 / n, 1),
                                   "rationale": round((t2 - t1) * 1000 / n, 1)}

    def rerank(self, retrieved: List[List[RetrievedDoc]], scores: List[np.ndarray], top_k: int):
        """Keep, per claim, the ``top_k`` abstracts with the strongest evidence sentence.

        score(d) = max_s P(rationale | claim, s) + rerank_weight * (1 - (rank(d) - 1) / n)
        """
        weight = self.cfg.retrieval.rerank_weight
        out_docs, out_scores, offset = [], [], 0
        for docs in retrieved:
            doc_scores = scores[offset:offset + len(docs)]
            offset += len(docs)
            n = len(docs)
            if n <= top_k:
                out_docs.append(docs)
                out_scores.extend(doc_scores)
                continue
            key = [(float(sc.max()) if len(sc) else 0.0) + weight * (1 - i / n) for i, sc in enumerate(doc_scores)]
            order = sorted(range(n), key=lambda i: (-key[i], i))[:top_k]
            out_docs.append([dataclass_replace(docs[i], rank=j + 1, retrieval_rank=docs[i].rank)
                             for j, i in enumerate(order)])
            out_scores.extend(doc_scores[i] for i in order)
        return out_docs, out_scores

    # ------------------------------------------------------------------ streaming
    def verify_stream(self, claim: str, top_k: Optional[int] = None, explain: bool = True,
                      source=None, documents: Optional[List[Document]] = None) -> Iterator[Tuple[str, Dict]]:
        """Run the pipeline for one claim, yielding ``(event, data)`` tuples as each stage completes.

        Events: ``stage`` (start / done of retrieval, rationale, nli, explanation), ``candidates`` (papers
        found), ``result`` (verdict + evidence), ``token`` / ``reset`` / ``explanation`` (streamed
        explanation) and ``done``. ``source`` is a live :class:`~claimverifier.sources.LiteratureSource`;
        ``documents`` are user-supplied abstracts. Without either, the indexed corpus is searched.
        """
        top_k = top_k or self.cfg.retrieval.top_k
        start = time.perf_counter()
        source_label = (source.label if source is not None else
                        "Your abstracts" if documents is not None else "Indexed corpus")
        source_key = source.name if source is not None else ("custom" if documents is not None else "corpus")
        yield "stage", {"stage": "retrieval", "status": "start", "source": source_label}
        t0 = time.perf_counter()
        if documents is not None:
            retrieved = [RetrievedDoc(doc=d, rank=i + 1, score=0.0, dense_score=0.0) for i, d in enumerate(documents)]
        elif source is not None:
            docs = source.search(claim, max(self.cfg.sources.fetch_size, top_k))
            retrieved = self._live_candidates(claim, docs)
        else:
            retrieved = self.retriever.search_batch([claim], self.retrieval_depth(top_k))[0]
        retrieval_ms = (time.perf_counter() - t0) * 1000
        yield "stage", {"stage": "retrieval", "status": "done", "ms": round(retrieval_ms, 1), "count": len(retrieved)}
        yield "candidates", {"documents": [_preview(r) for r in retrieved]}
        if not retrieved:
            raise LookupError(f"No abstracts found in {source_label} for this claim.")

        yield "stage", {"stage": "rationale", "status": "start", "count": len(retrieved)}
        t0 = time.perf_counter()
        scores = self.score_rationales([claim], [retrieved])
        if documents is None:
            kept, scores = self.rerank([retrieved], scores, top_k)
            kept = kept[0]
        else:
            kept = retrieved
        rationale_ms = (time.perf_counter() - t0) * 1000
        yield "stage", {"stage": "rationale", "status": "done", "ms": round(rationale_ms, 1),
                        "sentences": int(sum(len(r.doc.sentences) for r in retrieved)), "kept": len(kept)}

        yield "stage", {"stage": "nli", "status": "start", "count": len(kept)}
        t0 = time.perf_counter()
        result = self._analyze([claim], [kept], explain=False, sentence_stance=True, rationale_scores=scores)[0]
        nli_ms = (time.perf_counter() - t0) * 1000
        result.source, result.candidates_scanned = source_key, len(retrieved)
        result.timings_ms = {"retrieval": round(retrieval_ms, 1), "rationale": round(rationale_ms, 1),
                             "nli": round(nli_ms, 1)}
        yield "stage", {"stage": "nli", "status": "done", "ms": round(nli_ms, 1)}
        yield "result", result.to_dict()

        if explain and self.explainer.backend_name != "none":
            yield "stage", {"stage": "explanation", "status": "start", "backend": self.explainer.backend_name}
            t0 = time.perf_counter()
            for kind, payload in self.explainer.stream(claim, result.verdict, result.confidence,
                                                       self.evidence_for_prompt(result), result.mixed_evidence):
                if kind == "token":
                    yield "token", {"text": payload}
                elif kind == "reset":
                    yield "reset", {}
                else:
                    result.explanation, result.explanation_backend = payload.text, payload.backend
                    result.citations = payload.citations
                    yield "explanation", {"text": payload.text, "backend": payload.backend,
                                          "citations": payload.citations, "error": payload.error}
            result.timings_ms["explanation"] = round((time.perf_counter() - t0) * 1000, 1)
            yield "stage", {"stage": "explanation", "status": "done", "ms": result.timings_ms["explanation"]}
        yield "done", {"timings_ms": result.timings_ms,
                       "total_ms": round((time.perf_counter() - start) * 1000, 1)}

    def _live_candidates(self, claim: str, docs: List[Document]) -> List[RetrievedDoc]:
        if not docs:
            return []
        emb = self.retriever.embedder
        cos = emb.encode([d.full_text for d in docs]) @ emb.encode([claim])[0]
        return [RetrievedDoc(doc=d, rank=i + 1, score=float(c), dense_score=float(c)) for i, (d, c) in
                enumerate(zip(docs, cos))]

    # ------------------------------------------------------------------ core
    def score_rationales(self, claims: List[str], retrieved: List[List[RetrievedDoc]]) -> List[np.ndarray]:
        items = [(claim, r.doc.sentences) for claim, docs in zip(claims, retrieved) for r in docs]
        return self.rationale.score_batch(items) if items else []

    def _analyze(self, claims: List[str], retrieved: List[List[RetrievedDoc]], explain: bool,
                 sentence_stance: bool, rationale_scores: Optional[List[np.ndarray]] = None,
                 threshold: Optional[float] = None, nei_weight: Optional[float] = None) -> List[VerificationResult]:
        rcfg, ncfg = self.cfg.rationale, self.cfg.nli
        threshold = rcfg.threshold if threshold is None else threshold
        nei_weight = self.cfg.aggregation.nei_weight if nei_weight is None else nei_weight
        n_claims = max(len(claims), 1)

        # 1) rationale scores for every sentence of every retrieved abstract
        timings: Dict[str, float] = {}
        if rationale_scores is None:
            t0 = time.perf_counter()
            all_scores = self.score_rationales(claims, retrieved)
            timings["rationale"] = round((time.perf_counter() - t0) * 1000 / n_claims, 1)
        else:
            all_scores = rationale_scores

        # 2) select evidence sentences and build NLI inputs
        selections, doc_pairs, sent_pairs = [], [], []
        k = 0
        for claim, docs in zip(claims, retrieved):
            per_doc = []
            for r in docs:
                scores = all_scores[k]
                k += 1
                chosen, passed = select_sentences(scores, threshold, rcfg.max_sentences)
                per_doc.append((r, scores, chosen, passed))
                title = r.doc.title if ncfg.include_title else None
                doc_pairs.append((format_premise([r.doc.sentences[i] for i in chosen], title), claim))
                if sentence_stance:
                    sent_pairs.extend((r.doc.sentences[i], claim) for i in chosen)
            selections.append(per_doc)

        # 3) NLI: document-level evidence (for the verdict) and sentence-level (for highlighting)
        t0 = time.perf_counter()
        doc_probs = self.nli.predict(doc_pairs) if doc_pairs else np.zeros((0, len(LABELS)))
        sent_probs = self.nli.predict(sent_pairs) if sent_pairs else np.zeros((0, len(LABELS)))
        timings["nli"] = round((time.perf_counter() - t0) * 1000 / n_claims, 1)

        # 4) aggregate + 5) explain
        results, di, si = [], 0, 0
        components = self.components()
        for claim, per_doc in zip(claims, selections):
            documents, relevances, probs_list = [], [], []
            for r, scores, chosen, passed in per_doc:
                p = doc_probs[di]
                di += 1
                evidence = []
                for i in chosen:
                    if sentence_stance:
                        sp = sent_probs[si]
                        si += 1
                    else:
                        sp = p
                    evidence.append(EvidenceSentence(i, r.doc.sentences[i], float(scores[i]), doc_stance(sp),
                                                     _probs_dict(sp)))
                relevance = float(max(scores[i] for i in chosen)) if chosen else 0.0
                relevances.append(relevance)
                probs_list.append(p)
                documents.append(DocumentResult(
                    doc_id=r.doc.doc_id, title=r.doc.title, url=r.doc.url, citation=r.rank, rank=r.rank,
                    retrieval_score=r.score, dense_score=r.dense_score, bm25_score=r.bm25_score,
                    relevance=relevance, has_evidence=passed,
                    evidence_weight=float(min(1.0, relevance / max(threshold, 1e-6))),
                    stance=doc_stance(p), stance_probs=_probs_dict(p),
                    evidence=evidence, sentences=list(r.doc.sentences),
                    sentence_scores=[round(float(s), 4) for s in scores],
                    retrieval_rank=r.retrieval_rank or r.rank, meta=dict(r.doc.meta)))
            verdict = aggregate(relevances, probs_list, threshold, nei_weight, self.cfg.aggregation.min_relevance)
            result = VerificationResult(
                claim=claim, verdict=verdict.label, verdict_display=DISPLAY_NAMES[verdict.label],
                confidence=verdict.confidence, scores=verdict.scores, support_strength=verdict.support_strength,
                contradict_strength=verdict.contradict_strength, mixed_evidence=verdict.mixed, documents=documents,
                timings_ms=dict(timings), components=components)
            if explain:
                t0 = time.perf_counter()
                exp = self.explainer.generate(claim, verdict.label, verdict.confidence,
                                              self.evidence_for_prompt(result), verdict.mixed)
                result.explanation, result.explanation_backend, result.citations = exp.text, exp.backend, exp.citations
                result.timings_ms["explanation"] = round((time.perf_counter() - t0) * 1000, 1)
            results.append(result)
        return results

    @staticmethod
    def evidence_for_prompt(result: VerificationResult, max_weak: int = 2) -> List[EvidenceForPrompt]:
        """Evidence passed to the explanation step: the abstracts that drive the verdict, strongest first
        (or the closest abstracts if none is relevant enough)."""
        docs = sorted([d for d in result.documents if d.evidence_weight >= MIN_SHOWN_WEIGHT],
                      key=lambda d: -evidence_strength(d))
        if not docs:
            docs = sorted(result.documents, key=lambda d: -d.relevance)[:max_weak]
        return [EvidenceForPrompt(citation=d.citation, title=d.title, stance=d.stance, relevance=d.relevance,
                                  sentences=[e.text for e in d.evidence], url=d.url) for d in docs]
