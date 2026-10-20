"""Initial question rewriting for retrieval."""

from __future__ import annotations

import re

from .._shared.llm_adapter import get_adapter, record_llm_result
from .._shared.model_config import tier_model
from .._shared.state_payload import StatePayload
from .prompts import QUESTION_TO_QUERY_PROMPT


def _parse_query(text: str) -> str:
    match = re.search(r"search\s+query\s*:\s*(.+)", text, re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def question_to_query(
    payload: StatePayload, tier: int, adapter=None, **options
) -> StatePayload:
    if not payload.question:
        raise ValueError("question_to_query requires question")
    output = payload.snapshot()
    model = tier_model(tier)
    result = (adapter or get_adapter()).call(
        model,
        QUESTION_TO_QUERY_PROMPT.format(question=payload.question),
        **options,
    )
    record_llm_result(output, "question_to_query", model, result)
    output.query_1 = _parse_query(result.text)
    return output

