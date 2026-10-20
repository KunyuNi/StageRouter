"""Injectable vector-retrieval boundary for RAG operators."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol
from urllib.request import Request, urlopen

from .model_config import EMBEDDING_MODEL


@dataclass(frozen=True)
class RetrievalItem:
    """One vector-search hit, including the artifacts needed for replay."""

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        if not str(self.chunk_id).strip():
            raise ValueError("retrieval result requires chunk_id")
        if not str(self.text).strip():
            raise ValueError("retrieval result requires non-empty text")
        if not math.isfinite(float(self.score)):
            raise ValueError("retrieval result score must be finite")
        if int(self.rank) < 1:
            raise ValueError("retrieval result rank must be at least one")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResponse:
    """Rich response returned by the fixed embedding retrieval service."""

    query: str
    results: List[RetrievalItem]
    embedding_input_tokens: int = 0
    embedding_latency_ms: float = 0.0
    index_identity: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [item.to_json_dict() for item in self.results],
            "embedding_usage": {
                "input_tokens": int(self.embedding_input_tokens),
                "latency_ms": float(self.embedding_latency_ms),
            },
            "index_identity": self.index_identity,
        }


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> RetrievalResponse:
        ...


class HttpRetriever:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (
            base_url or os.getenv("RAG_RETRIEVAL_BASE_URL", "http://localhost:8008")
        ).rstrip("/")
        self.timeout = float(timeout)

    @staticmethod
    def build_request(query: str, top_k: int = 3) -> dict:
        return {
            "query": query,
            "top_k": int(top_k),
            "embedding_model": EMBEDDING_MODEL,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> RetrievalResponse:
        if embedding_model != EMBEDDING_MODEL:
            raise ValueError(
                f"RAG retrieval model is fixed to {EMBEDDING_MODEL!r}, "
                f"got {embedding_model!r}"
            )
        body = json.dumps(self.build_request(query, top_k)).encode("utf-8")
        request = Request(
            f"{self.base_url}/retrieve",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("retrieval response results must be a list")

        items: List[RetrievalItem] = []
        for index, result in enumerate(raw_results, start=1):
            if not isinstance(result, dict):
                raise ValueError("retrieval result must be an object")
            # Do not silently discard malformed hits: a partial result cannot
            # be replayed into a later state encoder.
            if "chunk_id" not in result:
                raise ValueError("retrieval result missing chunk_id")
            if "text" not in result:
                raise ValueError("retrieval result missing text")
            if "score" not in result:
                raise ValueError("retrieval result missing score")
            rank = result.get("rank", index)
            metadata = result.get("metadata", {})
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError("retrieval result metadata must be an object")
            items.append(
                RetrievalItem(
                    chunk_id=str(result["chunk_id"]),
                    text=str(result["text"]),
                    score=float(result["score"]),
                    rank=int(rank),
                    metadata=dict(metadata),
                )
            )

        usage = payload.get("embedding_usage") or payload.get("embedding", {})
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        latency_ms = float(usage.get("latency_ms", 0.0) or 0.0)
        if input_tokens < 0 or not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("invalid embedding usage in retrieval response")
        response_query = str(payload.get("query", query))
        return RetrievalResponse(
            query=response_query,
            results=items,
            embedding_input_tokens=input_tokens,
            embedding_latency_ms=latency_ms,
            index_identity=str(payload.get("index_identity", "")),
        )


def legacy_retrieval_response(value: Any, query: str) -> RetrievalResponse:
    """Normalize an old injected ``list[str]`` fake for compatibility.

    Production HTTP responses are parsed strictly above.  This adapter-only
    compatibility path lets existing offline tests/fakes continue to supply
    text lists while new operators still expose a uniform rich response.
    """
    if isinstance(value, RetrievalResponse):
        return value
    if isinstance(value, dict):
        raw_results = value.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("retrieval response results must be a list")
        items = []
        for index, item in enumerate(raw_results, start=1):
            if isinstance(item, RetrievalItem):
                items.append(item)
            elif isinstance(item, dict):
                items.append(
                    RetrievalItem(
                        chunk_id=str(item.get("chunk_id", f"legacy:{index}")),
                        text=str(item.get("text", "")),
                        score=float(item.get("score", 0.0)),
                        rank=int(item.get("rank", index)),
                        metadata=dict(item.get("metadata") or {}),
                    )
                )
            else:
                items.append(
                    RetrievalItem(
                        chunk_id=f"legacy:{index}",
                        text=str(item),
                        score=0.0,
                        rank=index,
                        metadata={},
                    )
                )
        usage = value.get("embedding_usage") or {}
        return RetrievalResponse(
            query=str(value.get("query", query)),
            results=items,
            embedding_input_tokens=int(usage.get("input_tokens", 0) or 0),
            embedding_latency_ms=float(usage.get("latency_ms", 0.0) or 0.0),
            index_identity=str(value.get("index_identity", "")),
        )
    if isinstance(value, (list, tuple)):
        items = [
            RetrievalItem(
                chunk_id=f"legacy:{index}",
                text=str(item),
                score=0.0,
                rank=index,
                metadata={},
            )
            for index, item in enumerate(value, start=1)
            if str(item).strip()
        ]
        return RetrievalResponse(query=query, results=items)
    raise TypeError("retriever must return RetrievalResponse or a legacy text list")


_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = HttpRetriever()
    return _retriever


def set_retriever(retriever: Retriever) -> None:
    global _retriever
    _retriever = retriever
