"""First fixed-embedding retrieval stage."""

from __future__ import annotations

from .._shared.model_config import EMBEDDING_MODEL
from .._shared.retriever import get_retriever, legacy_retrieval_response
from .._shared.state_payload import StatePayload


def retrieval1(
    payload: StatePayload, retriever=None, top_k: int = 3
) -> StatePayload:
    query = (payload.query_1 or payload.question or "").strip()
    if not query:
        raise ValueError("retrieval1 requires query_1 or question")
    response = (retriever or get_retriever()).retrieve(
        query,
        top_k=int(top_k),
        embedding_model=EMBEDDING_MODEL,
    )
    response = legacy_retrieval_response(response, query)
    output = payload.snapshot()
    output.context = [item.text for item in response.results]
    output.metadata["retrieval1"] = {
        "query": query,
        "top_k": int(top_k),
        "embedding_model": EMBEDDING_MODEL,
        "index_identity": response.index_identity,
        "embedding_input_tokens": response.embedding_input_tokens,
        "embedding_latency_ms": response.embedding_latency_ms,
        "results": [item.to_json_dict() for item in response.results],
    }
    output.cost_tracker.record_embedding_call(
        stage="retrieval1",
        model=EMBEDDING_MODEL,
        input_tokens=response.embedding_input_tokens,
        latency_ms=response.embedding_latency_ms,
    )
    return output
