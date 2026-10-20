"""Second fixed-embedding retrieval and evidence merge stage."""

from __future__ import annotations

from .._shared.model_config import EMBEDDING_MODEL
from .._shared.retriever import get_retriever, legacy_retrieval_response
from .._shared.state_payload import StatePayload


def _stable_deduplicate(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def retrieval2(
    payload: StatePayload, retriever=None, top_k: int = 3
) -> StatePayload:
    query = (payload.query_2 or "").strip()
    if not query:
        raise ValueError("retrieval2 requires query_2")
    response = (retriever or get_retriever()).retrieve(
        query,
        top_k=int(top_k),
        embedding_model=EMBEDDING_MODEL,
    )
    response = legacy_retrieval_response(response, query)
    output = payload.snapshot()
    # Rich responses are deduplicated by stable chunk id.  A legacy injected
    # text-list fake has no identity, so only that path falls back to exact
    # text matching (preserving the historical operator behavior).
    known_chunk_ids = set()
    for stage_name in ("retrieval1", "retrieval2"):
        stage_data = payload.metadata.get(stage_name, {})
        for item in stage_data.get("results", []) if isinstance(stage_data, dict) else []:
            if isinstance(item, dict) and item.get("chunk_id"):
                known_chunk_ids.add(str(item["chunk_id"]))

    merged_context = list(payload.context)
    seen_text = set(merged_context)
    # If the incoming payload has no retrieval artifacts (for example an
    # older serialized state or an injected fake), there is no chunk identity
    # to compare against, so retain the historical exact-text fallback.
    use_text_fallback = not known_chunk_ids
    for item in response.results:
        is_legacy = item.chunk_id.startswith("legacy:")
        if (not is_legacy) and item.chunk_id in known_chunk_ids:
            continue
        if (is_legacy or use_text_fallback) and item.text in seen_text:
            continue
        merged_context.append(item.text)
        seen_text.add(item.text)
        if not is_legacy:
            known_chunk_ids.add(item.chunk_id)
    output.context = merged_context
    output.metadata["retrieval2"] = {
        "query": query,
        "top_k": int(top_k),
        "embedding_model": EMBEDDING_MODEL,
        "index_identity": response.index_identity,
        "embedding_input_tokens": response.embedding_input_tokens,
        "embedding_latency_ms": response.embedding_latency_ms,
        "results": [item.to_json_dict() for item in response.results],
    }
    output.cost_tracker.record_embedding_call(
        stage="retrieval2",
        model=EMBEDDING_MODEL,
        input_tokens=response.embedding_input_tokens,
        latency_ms=response.embedding_latency_ms,
    )
    return output
