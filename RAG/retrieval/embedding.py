"""OpenAI-compatible embedding adapter with per-call usage tracking.

The RAG workflow uses one embedding model for both retrieval hops.  Keeping the
adapter here (instead of in the LLM operator adapter) makes the vector index
independent from the four tiered LLM actions while still exposing token usage to
the retrieval stage.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # LlamaIndex is installed on the remote runtime, but not necessarily locally.
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import Field, PrivateAttr
except ImportError:  # pragma: no cover - exercised only before offline install
    class BaseEmbedding:  # type: ignore[no-redef]
        """Small import-time fallback so config/syntax checks work without LlamaIndex."""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default

    def PrivateAttr(default: Any = None, default_factory: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default_factory() if default_factory is not None else default


MODEL_NAME = "text-embedding-v3"


def _runtime_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _load_runtime_env() -> None:
    """Load the project `.env` using the same convention as the LLM adapter.

    The remote worker keeps the shared DashScope credentials in the project
    `.env`, while the embedding adapter is instantiated independently of the
    LLM adapter.  Loading it here preserves the existing runtime contract
    without copying or printing any secret values.
    """

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


class TrackedOpenAIEmbedding(BaseEmbedding):
    """LlamaIndex ``BaseEmbedding`` backed by an OpenAI-compatible endpoint.

    ``model``/``batch_size`` are accepted as convenience aliases in addition to
    LlamaIndex's canonical ``model_name``/``embed_batch_size`` fields.  Usage is
    kept in thread-local storage because a retrieval request may be served while
    another request is embedding a different query.
    """

    model_name: str = Field(default=MODEL_NAME)
    # DashScope text-embedding-v3 accepts at most ten inputs per request.
    embed_batch_size: int = Field(default=10, gt=0, le=2048)
    api_key: str = Field(default="", exclude=True)
    base_url: str = Field(default="", exclude=True)
    timeout: float = Field(default=120.0, gt=0, exclude=True)
    sdk_max_retries: int = Field(default=1, ge=0, le=1, exclude=True)

    _client: Any = PrivateAttr(default=None)
    _usage_local: Any = PrivateAttr(default_factory=threading.local)

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 10,
        sdk_max_retries: int = 1,
        **data: Any,
    ) -> None:
        _load_runtime_env()
        if model is not None:
            data.setdefault("model_name", model)
        data.setdefault("model_name", MODEL_NAME)
        data.setdefault("embed_batch_size", batch_size)
        data.setdefault("sdk_max_retries", int(sdk_max_retries))
        data.setdefault(
            "api_key",
            api_key
            or _runtime_value(
                "RAG_EMBEDDING_API_KEY",
                "RAG_LLM_API_KEY",
                "LLM_API_KEY",
                "DASHSCOPE_API_KEY",
                "OPENAI_API_KEY",
            ),
        )
        data.setdefault(
            "base_url",
            (
                base_url
                or _runtime_value(
                    "RAG_EMBEDDING_BASE_URL",
                    "RAG_LLM_BASE_URL",
                    "LLM_BASE_URL",
                    "OPENAI_BASE_URL",
                )
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ).rstrip("/"),
        )
        super().__init__(**data)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is remote-pinned
                raise RuntimeError("The openai package is required for embeddings") from exc
            if not self.api_key:
                raise RuntimeError(
                    "Embedding credentials are missing; configure RAG_EMBEDDING_API_KEY "
                    "or an OpenAI-compatible API key"
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=int(self.sdk_max_retries),
                timeout=self.timeout,
            )
        return self._client

    @staticmethod
    def _usage_tokens(usage: Any, texts: Iterable[str]) -> int:
        for name in ("prompt_tokens", "input_tokens", "total_tokens"):
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            try:
                value = int(value or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value
        # Conservative fallback used only when an endpoint omits usage.
        return sum(max(1, (len(text) + 3) // 4) for text in texts)

    def _request(self, texts: list[str], *, query: bool) -> list[list[float]]:
        if not texts:
            return []
        started = time.perf_counter()
        response = self._get_client().embeddings.create(
            model=self.model_name,
            input=texts,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(response, "usage", None)
        tokens = self._usage_tokens(usage, texts)
        record = {"input_tokens": int(tokens), "latency_ms": float(latency_ms)}
        if query:
            self._usage_local.query = record
        else:
            self._usage_local.document = record

        data = getattr(response, "data", None) or []
        # OpenAI returns data in input order, but sorting by index protects us from
        # compatible endpoints that serialize the response in a different order.
        data = sorted(data, key=lambda item: int(getattr(item, "index", 0) or 0))
        vectors: list[list[float]] = []
        for item in data:
            vector = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            if not vector:
                raise RuntimeError("Embedding endpoint returned an empty vector")
            vectors.append([float(value) for value in vector])
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding endpoint returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def pop_query_usage(self) -> dict[str, float | int]:
        """Return and clear usage for the most recent query embedding call."""

        value = getattr(self._usage_local, "query", None)
        self._usage_local.query = None
        return dict(value or {"input_tokens": 0, "latency_ms": 0.0})

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._request([query], query=True)[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        # BaseEmbedding may call this from async retrievers.  The OpenAI client is
        # deliberately kept sync to match the existing RAG adapter, so delegate to
        # a worker thread without sharing usage state with another request.
        import asyncio

        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._request([text], query=False)[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._request(list(texts), query=False)
