"""HTTP wrapper around the persistent MultiHop-RAG retrieval index."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - FastAPI is supplied by remote runtime
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = RuntimeError  # type: ignore[assignment,misc]

    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default

from .embedding import MODEL_NAME
from .index import DEFAULT_CORPUS, DEFAULT_PERSIST_ROOT, IndexConfig, MultiHopIndex, load_index
from RAG.rl.error_policy import is_rate_limit_error


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1)
    embedding_model: str = MODEL_NAME


class RetrievalItemResponse(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any]


class RetrievalResponseModel(BaseModel):
    query: str
    results: list[RetrievalItemResponse]
    count: int
    embedding_usage: dict[str, float | int]
    index_identity: str


class HealthResponse(BaseModel):
    status: str
    index_identity: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


active_index: Optional[MultiHopIndex] = None


def _provider_http_status(error: BaseException) -> int | None:
    return 429 if is_rate_limit_error(error) else None


def _config_from_env() -> IndexConfig:
    return IndexConfig(
        corpus_path=os.getenv("RAG_CORPUS_PATH", str(DEFAULT_CORPUS)),
        persist_root=os.getenv("RAG_VECTOR_INDEX_ROOT", str(DEFAULT_PERSIST_ROOT)),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "256")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("RAG_RETRIEVAL_TOP_K", "3")),
    )


if FastAPI is not None:

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        global active_index
        active_index = load_index(_config_from_env())
        yield
        active_index = None

    app = FastAPI(
        title="MultiHop-RAG Retrieval Service",
        description="Persistent LlamaIndex + FAISS retrieval using text-embedding-v3",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        if active_index is None:
            raise HTTPException(status_code=503, detail="Retrieval index is not loaded")
        cfg = active_index.config
        return {
            "status": "healthy",
            "index_identity": active_index.index_identity,
            "embedding_model": cfg.embedding_model,
            "chunk_size": int(cfg.chunk_size),
            "chunk_overlap": int(cfg.chunk_overlap),
            "top_k": int(cfg.top_k),
        }

    @app.post("/retrieve", response_model=RetrievalResponseModel)
    async def retrieve(request: RetrievalRequest) -> dict[str, Any]:
        if active_index is None:
            raise HTTPException(status_code=503, detail="Retrieval index is not loaded")
        if request.embedding_model != MODEL_NAME:
            raise HTTPException(status_code=400, detail=f"embedding_model must be {MODEL_NAME!r}")
        try:
            response = active_index.retrieve(request.query, top_k=request.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            status_code = _provider_http_status(exc)
            if status_code is not None:
                raise HTTPException(
                    status_code=status_code,
                    detail="embedding provider rate limit: limit_requests",
                ) from exc
            raise
        return {
            "query": response.query,
            "results": [asdict(item) for item in response.results],
            "count": len(response.results),
            "embedding_usage": {
                "input_tokens": response.embedding_input_tokens,
                "latency_ms": response.embedding_latency_ms,
            },
            "index_identity": response.index_identity,
        }
else:  # pragma: no cover - allows local static import before FastAPI installation
    app = None
