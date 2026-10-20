"""Mapping between RAG decision stages and physical pipeline stages."""

from __future__ import annotations

from typing import Sequence


PHYSICAL_STAGE_NAMES = (
    "question_to_query",
    "retrieval1",
    "context_question_to_query",
    "retrieval2",
    "rerank",
    "context_question_to_answer",
    "qa_evaluation",
)

DECISION_TO_PHYSICAL = (0, 2, 4, 5)
FIXED_STAGE_IDXS = (1, 3, 6)
FIXED_STAGE_DEFAULT_TIER = 0
NUM_PHYSICAL_STAGES = len(PHYSICAL_STAGE_NAMES)
NUM_DECISION_STAGES = len(DECISION_TO_PHYSICAL)
NUM_PIPELINE_STAGES = NUM_PHYSICAL_STAGES
NUM_RL_STAGES = NUM_DECISION_STAGES

# Descriptive aliases make the mapping convenient for code ported from the
# NL2SQL implementation without exposing any SQL-specific terminology.
PIPELINE_STAGE_NAMES = PHYSICAL_STAGE_NAMES
RL_DECISION_TO_PIPELINE_STAGE_IDXS = DECISION_TO_PHYSICAL
FIXED_PIPELINE_STAGE_IDXS = FIXED_STAGE_IDXS


def decision_to_physical_stage_idx(decision_idx: int) -> int:
    """Return the physical index for a decision index (0 through 3)."""

    try:
        return DECISION_TO_PHYSICAL[int(decision_idx)]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid RAG decision index: {decision_idx!r}") from exc


def physical_to_decision_idx(stage_idx: int) -> int | None:
    """Return the decision index, or ``None`` for a fixed physical stage."""

    try:
        return DECISION_TO_PHYSICAL.index(int(stage_idx))
    except ValueError:
        return None


def is_fixed_stage_idx(stage_idx: int) -> bool:
    return int(stage_idx) in FIXED_STAGE_IDXS


# Compatibility aliases for callers ported from NL2SQL's naming convention.
decision_to_pipeline_stage_idx = decision_to_physical_stage_idx
pipeline_to_decision_idx = physical_to_decision_idx
is_fixed_pipeline_stage_idx = is_fixed_stage_idx


def route_key(actions: Sequence[int]) -> str:
    """Canonical route key for the four RAG decision actions."""

    values = [int(action) for action in actions]
    if len(values) != NUM_DECISION_STAGES:
        raise ValueError(
            f"RAG route requires {NUM_DECISION_STAGES} actions, got {len(values)}"
        )
    if any(value not in (0, 1, 2, 3) for value in values):
        raise ValueError(f"RAG actions must be in 0..3, got {values!r}")
    return "-".join(str(value) for value in values)
