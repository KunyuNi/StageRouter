"""Advanced-QA baseline operators registered with RAG-local contracts."""

from __future__ import annotations

from functools import partial

from .._shared.state_payload import CostTracker, StatePayload
from .._shared.stage_action_registry import (
    SKIP_ACTION,
    T1_ACTION,
    T2_ACTION,
    T3_ACTION,
    StageActionRegistry,
)
from .._shared.tier_registry import TierRegistry
from .context_question_to_answer import context_question_to_answer
from .context_question_to_query import context_question_to_query
from .qa_evaluation import qa_evaluation
from .question_to_query import question_to_query
from .rerank import rerank
from .retrieval1 import retrieval1
from .retrieval2 import retrieval2


def _tier_operator(function, tier: int):
    operator = partial(function, tier=tier)
    operator.__name__ = f"{function.__name__}_tier{tier}"
    operator.__module__ = function.__module__
    return operator


for _stage_name, _function in {
    "question_to_query": question_to_query,
    "context_question_to_query": context_question_to_query,
    "rerank": rerank,
    "context_question_to_answer": context_question_to_answer,
}.items():
    for _tier in (1, 2, 3):
        TierRegistry.register(_stage_name, _tier, _tier_operator(_function, _tier))


FIXED_OPERATORS = {
    "retrieval1": retrieval1,
    "retrieval2": retrieval2,
    "qa_evaluation": qa_evaluation,
}


_ALL_ACTIONS = (T1_ACTION, T2_ACTION, T3_ACTION, SKIP_ACTION)
_LLM_ACTIONS = (T1_ACTION, T2_ACTION, T3_ACTION)

# Stage 1 always flows into retrieval1, including when query rewriting is skipped.
StageActionRegistry.register(
    "question_to_query",
    _ALL_ACTIONS,
    fixed_followups_by_action={action: ("retrieval1",) for action in _ALL_ACTIONS},
)

# Stage 2 is a coupled second-hop block: SKIP bypasses both rewriting and retrieval2.
StageActionRegistry.register(
    "context_question_to_query",
    _ALL_ACTIONS,
    fixed_followups_by_action={action: ("retrieval2",) for action in _LLM_ACTIONS},
)

StageActionRegistry.register("rerank", _ALL_ACTIONS)
StageActionRegistry.register("context_question_to_answer", _LLM_ACTIONS)

__all__ = [
    "CostTracker",
    "FIXED_OPERATORS",
    "SKIP_ACTION",
    "StatePayload",
    "StageActionRegistry",
    "T1_ACTION",
    "T2_ACTION",
    "T3_ACTION",
    "context_question_to_answer",
    "context_question_to_query",
    "qa_evaluation",
    "question_to_query",
    "rerank",
    "retrieval1",
    "retrieval2",
]
