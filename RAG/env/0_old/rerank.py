"""Contextual pairwise reranking for retrieved passages."""

from __future__ import annotations

import json
import re

from .._shared.llm_adapter import get_adapter, record_llm_result
from .._shared.model_config import tier_model
from .._shared.state_payload import StatePayload
from .prompts import PAIRWISE_RERANK_PROMPT

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _score(text: str) -> float:
    match = _JSON_OBJECT.search(text)
    if not match:
        return 0.0
    try:
        value = float(json.loads(match.group(0)).get("score", 0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    return min(100.0, max(0.0, value))


def rerank(
    payload: StatePayload,
    tier: int,
    adapter=None,
    top_k: int = 6,
    **options,
) -> StatePayload:
    if not payload.question:
        raise ValueError("rerank requires question")
    if not payload.context:
        raise ValueError("rerank requires context")

    output = payload.snapshot()
    model = tier_model(tier)
    client = adapter or get_adapter()
    call_options = dict(options)
    # Retrieval operators keep rich artifacts in metadata while preserving a
    # text-only prompt context.  Re-associate each passage with its retrieval
    # identity/rank for the replayable rerank record.
    retrieval_artifacts = []
    for stage_name in ("retrieval1", "retrieval2"):
        stage_data = payload.metadata.get(stage_name, {})
        if not isinstance(stage_data, dict):
            continue
        values = stage_data.get("results", [])
        if isinstance(values, list):
            retrieval_artifacts.extend(
                item for item in values if isinstance(item, dict)
            )
    used_artifacts = set()
    scored = []
    for index, passage in enumerate(payload.context):
        result = client.call(
            model,
            PAIRWISE_RERANK_PROMPT.format(
                query=payload.question,
                passage=passage,
            ),
            **call_options,
        )
        record_llm_result(output, "rerank", model, result)
        artifact = None
        for artifact_index, candidate in enumerate(retrieval_artifacts):
            if artifact_index in used_artifacts:
                continue
            if candidate.get("text") == passage:
                artifact = candidate
                used_artifacts.add(artifact_index)
                break
        if artifact is None:
            artifact = {
                "chunk_id": f"context:{index}",
                "rank": index + 1,
                "score": 0.0,
                "metadata": {},
            }
        scored.append(
            {
                "score": _score(result.text),
                "original_index": index,
                "chunk_id": str(artifact.get("chunk_id", f"context:{index}")),
                "text": passage,
                "original_retrieval_rank": int(
                    artifact.get("rank", index + 1) or index + 1
                ),
                "retrieval_score": float(artifact.get("score", 0.0) or 0.0),
                "retrieval_metadata": dict(artifact.get("metadata") or {}),
            }
        )

    scored.sort(key=lambda item: (-item["score"], item["original_index"]))
    rerank_results = []
    for final_rank, item in enumerate(scored, start=1):
        rerank_results.append(
            {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "original_retrieval_rank": item["original_retrieval_rank"],
                "rerank_score": item["score"],
                "rerank_rank": final_rank,
                "retrieval_score": item["retrieval_score"],
                "retrieval_metadata": item["retrieval_metadata"],
            }
        )
    output.context = [item["text"] for item in rerank_results[: int(top_k)]]
    output.metadata["rerank"] = {
        "query": payload.question,
        "top_k": int(top_k),
        "results": rerank_results,
    }
    return output
