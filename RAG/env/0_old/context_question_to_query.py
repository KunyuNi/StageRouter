"""Context-aware follow-up query rewriting."""

from __future__ import annotations

import re

from .._shared.llm_adapter import get_adapter, record_llm_result
from .._shared.model_config import tier_model
from .._shared.state_payload import StatePayload
from .prompts import CONTEXT_QUESTION_TO_QUERY_PROMPT


def _parse_query(text: str) -> str:
    match = re.search(r"search\s+query\s*:\s*(.+)", text, re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def context_question_to_query(
    payload: StatePayload, tier: int, adapter=None, **options
) -> StatePayload:
    if not payload.question:
        raise ValueError("context_question_to_query requires question")
    if not payload.context:
        raise ValueError("context_question_to_query requires context")
    output = payload.snapshot()
    model = tier_model(tier)
    result = (adapter or get_adapter()).call(
        model,
        CONTEXT_QUESTION_TO_QUERY_PROMPT.format(
            context=payload.context,
            question=payload.question,
        ),
        **options,
    )
    record_llm_result(output, "context_question_to_query", model, result)
    # A few provider responses can be empty despite a successful HTTP call.
    # Keep the pipeline executable by falling back to the previous-hop query
    # (or the original question) instead of persisting an unusable query_2.
    parsed_query = _parse_query(result.text or "")
    output.query_2 = (
        parsed_query
        or (payload.query_1 or "").strip()
        or (payload.question or "").strip()
    )
    if not output.query_2:
        raise ValueError("context_question_to_query produced an empty query_2")
    return output
