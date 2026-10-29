"""FastAPI backend: document management, search and investigation.

Run:  uvicorn investigator.api:app --port 8000
UI:   http://localhost:8000/       (one static page, src/investigator/web/index.html)
Docs: http://localhost:8000/docs   (interactive OpenAPI page)

Document writes are serialised by one lock (an add, update or delete touches three stores in a
fixed order). Searches run concurrently; the search index guards its in-memory BM25 itself.
That is enough for a single-instance deployment.
"""

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .analytics import summarize
from .factory import Components, build
from .ingestion import DocumentNotFound, IndexingError
from .investigate import Investigator
from .llm import LLMError, make_llm
from .parsers import ParseError, parse_file

STATE: dict = {}
LOCK = threading.RLock()
PROCESS_START = time.time()


def memory_mb() -> float | None:
    """Current resident memory of this process (Linux: /proc). None where unavailable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def init(components: Components | None = None, llm="auto") -> None:
    """Build the app's components. Tests pass their own; production reads the environment."""
    components = components or build()
    if llm == "auto":
        s = components.settings
        try:
            llm = make_llm(s.llm_provider, s.llm_api_key, s.llm_model, getattr(s, "llm_fallback_models", None))
        except LLMError as exc:  # e.g. provider set but key missing: run retrieval-only, say why
            STATE["llm_error"] = str(exc)
            llm = None
    STATE.update(components=components, llm=llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "components" not in STATE:
        init()
    c = STATE["components"]
    # Build the keyword index before the first request.
    c.index.warm_bm25(c.settings.chunk_strategy, getattr(c.settings, "bm25_build_subprocess", False))
    STATE["startup_seconds"] = round(time.time() - PROCESS_START, 1)
    yield


app = FastAPI(title="AI Investigation Engine", version="0.1.0", lifespan=lifespan)
WEB_PAGE = Path(__file__).parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
def web_ui():
    """The browser UI: one static page that calls the JSON endpoints below."""
    return FileResponse(WEB_PAGE, media_type="text/html")


def C() -> Components:
    return STATE["components"]


def _doc_summary(doc) -> dict:
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "format": doc.format,
        "source_type": doc.source_type,
        "version": doc.version,
        "status": doc.status.value,
        "duplicate_of": doc.duplicate_of,
        "sentences": len(doc.sentences),
        "metadata": doc.metadata,
    }


@app.get("/health")
def health():
    c = C()
    llm = STATE.get("llm")
    return {
        "status": "ok",
        "documents": c.registry.count_documents(),
        "documents_indexed": c.registry.count_documents(status="indexed"),
        "strategy": c.settings.chunk_strategy,
        "strategies": list(c.service.chunkers),
        "embedding_model": c.index.embedder.model_name,
        "llm": f"{llm.provider}:{llm.model}" if llm else None,
        "llm_error": STATE.get("llm_error"),
        "memory_mb": memory_mb(),
        "startup_seconds": STATE.get("startup_seconds"),
        "uptime_seconds": round(time.time() - PROCESS_START),
        "commit": (os.getenv("RENDER_GIT_COMMIT") or "")[:7] or None,
    }


@app.get("/documents")
def list_documents(
    q: str = "", status: str | None = None, limit: int = Query(50, le=200), offset: int = Query(0, ge=0)
):
    c = C()
    docs = c.registry.list_documents(q, status, limit, offset)
    return {"total": c.registry.count_documents(status=status, query=q), "items": [_doc_summary(d) for d in docs]}


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    c = C()
    doc = c.registry.get(doc_id)
    if not doc:
        raise HTTPException(404, f"document {doc_id} not found")
    out = _doc_summary(doc)
    out["text_sentences"] = [{"index": s.index, "text": s.text, "page": s.page} for s in doc.sentences]
    out["chunks"] = {name: len(c.registry.chunks_for(doc_id, name)) for name in c.service.chunkers}
    out["duplicates"] = [d.doc_id for d in c.registry.duplicates_of(doc_id)]
    return out


async def _read(file: UploadFile):
    data = await file.read()
    try:
        return parse_file(file.filename or "upload", data)
    except ParseError as exc:
        raise HTTPException(400, str(exc)) from None


@app.post("/documents", status_code=201)
async def upload_documents(file: UploadFile = File(...)):
    parsed = await _read(file)
    results = []
    with LOCK:
        for p in parsed:
            try:
                doc = C().service.add(p)
            except IndexingError as exc:
                raise HTTPException(503, f"saved but not indexed, retry later: {exc}") from None
            results.append(_doc_summary(doc))
    return {"documents": results}


@app.put("/documents/{doc_id}")
async def update_document(doc_id: str, file: UploadFile = File(...)):
    parsed = await _read(file)
    if len(parsed) != 1:
        raise HTTPException(400, "an update must contain exactly one document")
    with LOCK:
        try:
            doc = C().service.update(doc_id, parsed[0])
        except DocumentNotFound:
            raise HTTPException(404, f"document {doc_id} not found") from None
        except IndexingError as exc:
            raise HTTPException(503, str(exc)) from None
    return _doc_summary(doc)


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    with LOCK:
        try:
            C().service.delete(doc_id)
        except DocumentNotFound:
            raise HTTPException(404, f"document {doc_id} not found") from None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    strategy: str | None = None
    k: int = Field(10, ge=1, le=50)


def _strategy(name: str | None) -> str:
    c = C()
    name = name or c.settings.chunk_strategy
    if name not in c.service.chunkers:
        raise HTTPException(400, f"strategy must be one of {list(c.service.chunkers)}")
    return name


@app.post("/search")
def search(req: SearchRequest):
    strategy = _strategy(req.strategy)
    hits = C().index.search(req.query, strategy, req.mode, req.k)
    return {
        "hits": [
            {
                "chunk_id": h.chunk.chunk_id,
                "doc_id": h.chunk.doc_id,
                "title": h.chunk.title,
                "text": h.chunk.text,
                "sentence_start": h.chunk.sentence_start,
                "sentence_end": h.chunk.sentence_end,
                "score": h.score,
                "semantic_rank": h.semantic_rank,
                "keyword_rank": h.keyword_rank,
            }
            for h in hits
        ]
    }


class InvestigateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy: str | None = None
    mode: Literal["semantic", "keyword", "hybrid"] = "hybrid"


@app.post("/investigate")
def investigate(req: InvestigateRequest):
    strategy = _strategy(req.strategy)
    result = Investigator(C().index, STATE.get("llm"), strategy=strategy, mode=req.mode).run(req.question)
    if result.llm_calls:
        try:  # analytics must never break an answer
            C().registry.record_usage(result.llm_model, result.status, req.mode, result.llm_calls,
                                      result.input_tokens, result.output_tokens, result.seconds)
        except Exception as exc:  # noqa: BLE001 - logged in the answer, not raised
            result.warnings.append(f"usage not recorded: {type(exc).__name__}")
    return result.to_dict()


@app.get("/analytics")
def analytics(days: int = Query(14, ge=1, le=90)):
    """LLM usage of the live app (tokens, calls, models) and what it would cost on other models."""
    llm = STATE.get("llm")
    out = summarize(C().registry.usage_rows(), days=days)
    out["current_model"] = getattr(llm, "models", [getattr(llm, "model", None)])[0] if llm else None
    return out
