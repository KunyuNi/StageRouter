"""Final grounded answer generation."""

from __future__ import annotations

import re

from .._shared.llm_adapter import get_adapter, record_llm_result
from .._shared.model_config import tier_model
from .._shared.state_payload import StatePayload
from .prompts import CONTEXT_QUESTION_TO_ANSWER_PROMPT


def _parse_answer(text: str) -> str:
    match = re.search(r"answer\s*:\s*(.+)", text, re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def context_question_to_answer(
    payload: StatePayload, tier: int, adapter=None, **options
) -> StatePayload:
    if not payload.question:
        raise ValueError("context_question_to_answer requires question")
    if not payload.context:
        raise ValueError("context_question_to_answer requires context")
    output = payload.snapshot()
    model = tier_model(tier)
    result = (adapter or get_adapter()).call(
        model,
        CONTEXT_QUESTION_TO_ANSWER_PROMPT.format(
            context=payload.context,
            question=payload.question,
        ),
        **options,
    )
    record_llm_result(output, "context_question_to_answer", model, result)
    output.answer = _parse_answer(result.text)
    return output

