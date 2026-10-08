"""REST API (FastAPI) and web UI server.

Run with ``python -m claimverifier serve --config configs/default.yaml`` (or ``uvicorn claimverifier.api:app``;
the config is then read from the ``CLAIMVERIFIER_CONFIG`` environment variable) and open http://localhost:8000.

Endpoints (all under ``/api``; the original un-prefixed routes remain as aliases)
  GET  /api/health               status and loaded components
  GET  /api/info                 components, dataset statistics, live sources, defaults
  POST /api/verify               verify a claim (indexed corpus, live Europe PMC / PubMed, or your abstracts)
  POST /api/verify/stream        same, streamed as server-sent events (pipeline stages + explanation tokens)
  POST /api/verify/batch         verify up to 64 claims
  POST /api/verify/custom        verify a claim against abstracts supplied in the request
  GET  /api/search?q=            search the indexed corpus
  GET  /api/documents/{doc_id}   one abstract from the corpus
  GET  /api/examples             labelled SciFact claims
  GET  /api/reports              evaluation reports;  GET /api/reports/{name} for one report
The web UI (built from ``web/`` into ``claimverifier/web/dist``) is served at ``/``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Config, load_config
from .data.scifact import Document
from .labels import DISPLAY_NAMES
from .text import split_sentences

logger = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
Source = Literal["corpus", "europepmc", "pubmed", "custom"]


class CustomDocument(BaseModel):
    title: str = ""
    abstract: Union[str, List[str]]


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=1000, examples=["Vitamin D supplementation reduces fractures."])
    top_k: Optional[int] = Field(None, ge=1, le=20, description="number of abstracts to analyse")
    explain: bool = Field(True, description="generate a natural-language explanation with citations")
    source: Source = Field("corpus", description="where to look for evidence")
    documents: Optional[List[CustomDocument]] = Field(None, max_length=20, description="for source='custom'")


class BatchVerifyRequest(BaseModel):
    claims: List[str] = Field(..., min_length=1, max_length=64)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    explain: bool = False


class CustomVerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=1000)
    documents: List[CustomDocument] = Field(..., min_length=1, max_length=20)
    explain: bool = True


def _default_config_path() -> Optional[str]:
    path = os.environ.get("CLAIMVERIFIER_CONFIG")
    if path:
        return path
    return "configs/default.yaml" if os.path.exists("configs/default.yaml") else None


def _to_documents(docs: List[CustomDocument]) -> List[Document]:
    out = []
    for i, d in enumerate(docs):
        sentences = d.abstract if isinstance(d.abstract, list) else split_sentences(d.abstract)
        sentences = [s.strip() for s in sentences if s and s.strip()]
        if not sentences:
            raise HTTPException(status_code=422, detail=f"document {i + 1} has an empty abstract")
        out.append(Document(doc_id=-(i + 1), title=d.title.strip() or f"Your abstract {i + 1}", sentences=sentences,
                            meta={"source": "Your abstracts", "url": ""}))
    return out


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


class ResultCache:
    """Small thread-safe LRU of completed streams (list of events), keyed by request."""

    def __init__(self, size: int = 128):
        self.size = size
        self._data: "OrderedDict[tuple, List[Tuple[str, dict]]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key, events) -> None:
        with self._lock:
            self._data[key] = events
            while len(self._data) > self.size:
                self._data.popitem(last=False)


def create_app(verifier=None, config_path: Optional[str] = None, eager: Optional[bool] = None,
               serve_ui: bool = True) -> FastAPI:
    """``eager`` loads the models at startup (default: env ``CLAIMVERIFIER_EAGER=1``, set by ``serve``)."""
    if eager is None:
        eager = os.environ.get("CLAIMVERIFIER_EAGER", "0") == "1"
    state: Dict = {"verifier": verifier, "error": None, "cfg": verifier.cfg if verifier is not None else None,
                   "sources": {}, "stats": None}
    load_lock, run_lock = threading.Lock(), threading.Lock()
    cache = ResultCache()

    @asynccontextmanager
    async def lifespan(_app):
        if eager and state["verifier"] is None:
            try:
                await run_in_threadpool(get_verifier)
            except HTTPException:
                pass  # reported by /health; requests will return 503
        yield

    app = FastAPI(title="ClaimVerifier AI", version=__version__, lifespan=lifespan,
                  description="Scientific claim verification with retrieval-augmented NLI and LLM explanations.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # ------------------------------------------------------------------ helpers
    def get_cfg() -> Config:
        if state["cfg"] is None:
            overrides = json.loads(os.environ.get("CLAIMVERIFIER_OVERRIDES", "[]"))
            state["cfg"] = load_config(config_path or _default_config_path(), overrides)
        return state["cfg"]

    def get_verifier():
        if state["verifier"] is None:
            with load_lock:
                if state["verifier"] is None:
                    from .pipeline import ClaimVerifier

                    try:
                        state["verifier"] = ClaimVerifier.from_config(get_cfg())
                        state["error"] = None
                    except Exception as e:  # noqa: BLE001
                        state["error"] = f"{type(e).__name__}: {e}"
                        logger.exception("Failed to load the verifier")
                        raise HTTPException(status_code=503, detail=f"Verifier unavailable: {state['error']}")
        return state["verifier"]

    def get_source(name: str):
        from .sources import create_source

        cfg = get_cfg()
        if name not in cfg.sources.enabled:
            raise HTTPException(status_code=400, detail=f"source {name!r} is not enabled")
        if name not in state["sources"]:
            kwargs = {"email": cfg.sources.email} if name == "pubmed" else {}
            state["sources"][name] = create_source(name, timeout=cfg.sources.timeout, **kwargs)
        return state["sources"][name]

    def stream_events(req: VerifyRequest) -> Iterator[Tuple[str, dict]]:
        v = get_verifier()
        claim = req.claim.strip()
        documents = source = None
        if req.source == "custom":
            if not req.documents:
                raise HTTPException(status_code=422, detail="source='custom' requires documents")
            documents = _to_documents(req.documents)
        elif req.source != "corpus":
            source = get_source(req.source)
        key = None
        if documents is None:
            key = (claim.lower(), req.top_k or v.cfg.retrieval.top_k, req.source, req.explain)
            cached = cache.get(key)
            if cached is not None:
                yield from cached
                return
        events: List[Tuple[str, dict]] = []
        with run_lock:
            for event in v.verify_stream(claim, top_k=req.top_k, explain=req.explain, source=source,
                                         documents=documents):
                events.append(event)
                yield event
        if key is not None:
            cache.put(key, events)

    # ------------------------------------------------------------------ routes
    router = APIRouter()

    @router.get("/health")
    def health():
        v = state["verifier"]
        return {"status": "ok" if v is not None else ("error" if state["error"] else "not_loaded"),
                "version": __version__, "error": state["error"],
                "components": v.components() if v is not None else None,
                "documents": len(v.corpus) if v is not None else None}

    @router.get("/info")
    def info():
        from .sources import SOURCES

        v = get_verifier()
        cfg = v.cfg
        if state["stats"] is None:
            from .data.scifact import load_claims

            stats = {"documents": len(v.corpus)}
            for split in ("train", "dev"):
                try:
                    stats[f"{split}_claims"] = len(load_claims(cfg.data.data_dir, split))
                except FileNotFoundError:
                    pass
            state["stats"] = stats
        sources = [{"name": "corpus", "label": "SciFact corpus",
                    "description": f"{len(v.corpus):,} indexed research abstracts (offline)"}]
        sources += [SOURCES[n]().describe() for n in cfg.sources.enabled if n in SOURCES]
        sources.append({"name": "custom", "label": "Your abstracts",
                        "description": "Paste abstracts or paper text to check the claim against"})
        return {"version": __version__, "name": cfg.name, "components": v.components(), "stats": state["stats"],
                "sources": sources, "labels": DISPLAY_NAMES,
                "defaults": {"top_k": cfg.retrieval.top_k, "rerank_depth": cfg.retrieval.rerank_depth,
                             "explanation_backend": cfg.explanation.backend,
                             "explanation_model": (cfg.explanation.model_name if cfg.explanation.backend == "transformers"
                                                   else cfg.explanation.ollama_model if cfg.explanation.backend == "ollama"
                                                   else cfg.explanation.openai_model if cfg.explanation.backend == "openai"
                                                   else cfg.explanation.backend)}}

    @router.post("/verify")
    def verify(req: VerifyRequest):
        result, explanation = None, None
        for event, data in stream_events(req):
            if event == "result":
                result = dict(data)
            elif event == "explanation":
                explanation = data
            elif event == "done" and result is not None:
                result["timings_ms"] = data["timings_ms"]
        if result is None:
            raise HTTPException(status_code=500, detail="verification produced no result")
        if explanation:
            result.update(explanation=explanation["text"], explanation_backend=explanation["backend"],
                          citations=explanation["citations"])
        return result

    @router.post("/verify/stream")
    def verify_stream(req: VerifyRequest):
        # Validate cheaply up-front so obvious errors are proper HTTP errors, not stream events.
        if req.source == "custom" and not req.documents:
            raise HTTPException(status_code=422, detail="source='custom' requires documents")
        if req.source not in ("corpus", "custom"):
            get_source(req.source)
        get_verifier()

        def gen():
            try:
                for event, data in stream_events(req):
                    yield _sse(event, data)
            except HTTPException as e:
                yield _sse("error", {"message": e.detail})
            except LookupError as e:
                yield _sse("error", {"message": str(e)})
            except Exception as e:  # noqa: BLE001
                logger.exception("Streaming verification failed")
                yield _sse("error", {"message": f"{type(e).__name__}: {e}"})

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.post("/verify/batch")
    def verify_batch(req: BatchVerifyRequest):
        v = get_verifier()
        claims = [c.strip() for c in req.claims if c.strip()]
        if not claims:
            raise HTTPException(status_code=422, detail="no non-empty claims")
        with run_lock:
            return [r.to_dict() for r in v.verify_batch(claims, top_k=req.top_k, explain=req.explain)]

    @router.post("/verify/custom")
    def verify_custom(req: CustomVerifyRequest):
        v = get_verifier()
        docs = _to_documents(req.documents)
        with run_lock:
            return v.verify_against(req.claim.strip(), docs, explain=req.explain).to_dict()

    @router.get("/search")
    def search(q: str = Query(..., min_length=2, max_length=500), k: int = Query(10, ge=1, le=50)):
        v = get_verifier()
        with run_lock:
            hits = v.retriever.search(q, k)
        return [{"doc_id": h.doc.doc_id, "title": h.doc.title, "url": h.doc.url, "rank": h.rank,
                 "score": h.score, "dense_score": h.dense_score, "bm25_score": h.bm25_score,
                 "snippet": " ".join(h.doc.sentences[:2])[:400], "num_sentences": len(h.doc.sentences)}
                for h in hits]

    @router.get("/documents/{doc_id}")
    def document(doc_id: int):
        v = get_verifier()
        doc = v.doc_by_id.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
        return doc.to_dict()

    @router.get("/examples")
    def examples(n: int = Query(8, ge=1, le=300), seed: Optional[int] = None,
                 label: Optional[str] = Query(None, description="SUPPORTED, CONTRADICTED or INSUFFICIENT_EVIDENCE")):
        from .data.scifact import load_claims

        cfg = get_cfg()
        try:
            claims = load_claims(cfg.data.data_dir, "dev")
        except FileNotFoundError:
            return []
        if label:
            claims = [c for c in claims if c.label == label]
        rng = random.Random(seed)
        picked = rng.sample(claims, min(n, len(claims)))
        return [{"id": c.id, "claim": c.claim, "label": c.label, "label_display": DISPLAY_NAMES[c.label]}
                for c in picked]

    @router.get("/reports")
    def reports():
        out = []
        for path in sorted(Path(get_cfg().reports_dir).glob("*/metrics.json")):
            try:
                m = json.loads(path.read_text())
                c, r = m["verdict_classification"], m["evidence_retrieval"]
                out.append({"name": path.parent.name, "num_claims": m["num_claims"],
                            "accuracy": c["accuracy"], "macro_f1": c["macro_f1"],
                            "macro_precision": c["macro_precision"], "macro_recall": c["macro_recall"],
                            "recall_at_5": r.get("recall@5"), "mrr": r.get("mrr"),
                            "abstract_label_f1": m["scifact_abstract_level"]["label_only"]["f1"],
                            "components": m.get("components", {})})
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed report %s: %s", path, e)
        return out

    @router.get("/reports/{name}")
    def report(name: str):
        base = Path(get_cfg().reports_dir).resolve()
        path = (base / name / "metrics.json").resolve()
        if base not in path.parents or not path.exists():
            raise HTTPException(status_code=404, detail=f"report {name!r} not found")
        return {"name": name, "metrics": json.loads(path.read_text())}

    app.include_router(router, prefix="/api")
    app.include_router(router, include_in_schema=False)  # backwards-compatible un-prefixed routes

    # ------------------------------------------------------------------ web UI
    if serve_ui:
        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            index = WEB_DIST / "index.html"
            if not index.exists():
                return HTMLResponse(
                    "<h1>ClaimVerifier AI API</h1><p>The web UI is not built. Run <code>cd web && npm install && "
                    "npm run build</code>, or use the API at <a href='/docs'>/docs</a>.</p>")
            candidate = (WEB_DIST / path).resolve()
            if path and WEB_DIST in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
