"""REST API (FastAPI).

Run with ``python -m claimverifier serve --config configs/default.yaml`` (or ``uvicorn claimverifier.api:app``;
the config is then read from the ``CLAIMVERIFIER_CONFIG`` environment variable).

Endpoints
  GET  /health                 status and loaded components
  POST /verify                 verify one claim against the indexed corpus
  POST /verify/batch           verify up to 32 claims
  POST /verify/custom          verify a claim against abstracts supplied in the request
  GET  /documents/{doc_id}     fetch an abstract from the corpus
  GET  /examples               sample labelled SciFact claims
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .config import load_config
from .data.scifact import Document
from .labels import DISPLAY_NAMES
from .text import split_sentences

logger = logging.getLogger(__name__)


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=1000, examples=["Vitamin D supplementation reduces fractures."])
    top_k: Optional[int] = Field(None, ge=1, le=20, description="number of abstracts to retrieve")
    explain: bool = Field(True, description="generate a natural-language explanation with citations")


class BatchVerifyRequest(BaseModel):
    claims: List[str] = Field(..., min_length=1, max_length=32)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    explain: bool = False


class CustomDocument(BaseModel):
    title: str = ""
    abstract: Union[str, List[str]]


class CustomVerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=1000)
    documents: List[CustomDocument] = Field(..., min_length=1, max_length=20)
    explain: bool = True


def _default_config_path() -> Optional[str]:
    path = os.environ.get("CLAIMVERIFIER_CONFIG")
    if path:
        return path
    return "configs/default.yaml" if os.path.exists("configs/default.yaml") else None


def create_app(verifier=None, config_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="ClaimVerifier AI", version=__version__,
                  description="Scientific claim verification with retrieval-augmented NLI and LLM explanations.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    state = {"verifier": verifier, "error": None}
    load_lock, run_lock = threading.Lock(), threading.Lock()

    def get_verifier():
        if state["verifier"] is None:
            with load_lock:
                if state["verifier"] is None:
                    from .pipeline import ClaimVerifier

                    overrides = json.loads(os.environ.get("CLAIMVERIFIER_OVERRIDES", "[]"))
                    cfg = load_config(config_path or _default_config_path(), overrides)
                    try:
                        state["verifier"] = ClaimVerifier.from_config(cfg)
                    except Exception as e:  # noqa: BLE001
                        state["error"] = f"{type(e).__name__}: {e}"
                        logger.exception("Failed to load the verifier")
                        raise HTTPException(status_code=503, detail=f"Verifier unavailable: {state['error']}")
        return state["verifier"]

    @app.get("/health")
    def health():
        v = state["verifier"]
        return {"status": "ok" if v is not None else ("error" if state["error"] else "not_loaded"),
                "version": __version__, "error": state["error"],
                "components": v.components() if v is not None else None,
                "documents": len(v.corpus) if v is not None else None}

    @app.post("/verify")
    def verify(req: VerifyRequest):
        v = get_verifier()
        with run_lock:
            return v.verify(req.claim.strip(), top_k=req.top_k, explain=req.explain).to_dict()

    @app.post("/verify/batch")
    def verify_batch(req: BatchVerifyRequest):
        v = get_verifier()
        claims = [c.strip() for c in req.claims if c.strip()]
        if not claims:
            raise HTTPException(status_code=422, detail="no non-empty claims")
        with run_lock:
            return [r.to_dict() for r in v.verify_batch(claims, top_k=req.top_k, explain=req.explain)]

    @app.post("/verify/custom")
    def verify_custom(req: CustomVerifyRequest):
        v = get_verifier()
        docs = []
        for i, d in enumerate(req.documents):
            sentences = d.abstract if isinstance(d.abstract, list) else split_sentences(d.abstract)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                raise HTTPException(status_code=422, detail=f"document {i + 1} has an empty abstract")
            docs.append(Document(doc_id=-(i + 1), title=d.title or f"User document {i + 1}", sentences=sentences))
        with run_lock:
            return v.verify_against(req.claim.strip(), docs, explain=req.explain).to_dict()

    @app.get("/documents/{doc_id}")
    def document(doc_id: int):
        v = get_verifier()
        doc = v.doc_by_id.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
        return doc.to_dict()

    @app.get("/examples")
    def examples(n: int = Query(8, ge=1, le=50), seed: Optional[int] = None):
        from .data.scifact import load_claims

        v = get_verifier()
        try:
            claims = load_claims(v.cfg.data.data_dir, "dev")
        except FileNotFoundError:
            return []
        rng = random.Random(seed)
        picked = rng.sample(claims, min(n, len(claims)))
        return [{"id": c.id, "claim": c.claim, "label": c.label, "label_display": DISPLAY_NAMES[c.label]}
                for c in picked]

    return app


app = create_app()
