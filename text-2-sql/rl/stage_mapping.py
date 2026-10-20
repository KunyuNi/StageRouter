"""Shared mapping between RL decision stages and physical pipeline stages."""

from __future__ import annotations

from typing import Optional, Sequence

PIPELINE_STAGE_NAMES = [
    "keyword_extraction",
    "entity_retrieval",
    "context_retrieval",
    "column_filtering",
    "table_selection",
    "column_selection",
    "candidate_generation",
    "revision",
    "evaluation",
]

NUM_PIPELINE_STAGES = 8
CACHED_PIPELINE_STAGE_IDXS = tuple(range(NUM_PIPELINE_STAGES))

FIXED_PIPELINE_STAGE_IDXS = (1, 2)
FIXED_STAGE_DEFAULT_TIER = 0

RL_DECISION_TO_PIPELINE_STAGE_IDXS = (0, 3, 4, 5, 6, 7)
PIPELINE_TO_RL_DECISION_IDX = {
    pipeline_idx: decision_idx
    for decision_idx, pipeline_idx in enumerate(RL_DECISION_TO_PIPELINE_STAGE_IDXS)
}

NUM_RL_STAGES = len(RL_DECISION_TO_PIPELINE_STAGE_IDXS)

COLUMN_FILTERING_PIPELINE_IDX = 3
COLUMN_FILTERING_DECISION_IDX = PIPELINE_TO_RL_DECISION_IDX[
    COLUMN_FILTERING_PIPELINE_IDX
]
CANDIDATE_GENERATION_PIPELINE_IDX = 6
CANDIDATE_GENERATION_DECISION_IDX = PIPELINE_TO_RL_DECISION_IDX[
    CANDIDATE_GENERATION_PIPELINE_IDX
]
NON_SKIPPABLE_PIPELINE_STAGE_IDXS = (
    COLUMN_FILTERING_PIPELINE_IDX,
    CANDIDATE_GENERATION_PIPELINE_IDX,
)
NON_SKIPPABLE_DECISION_STAGE_IDXS = (
    COLUMN_FILTERING_DECISION_IDX,
    CANDIDATE_GENERATION_DECISION_IDX,
)
REVISION_PIPELINE_IDX = 7
REVISION_DECISION_IDX = PIPELINE_TO_RL_DECISION_IDX[REVISION_PIPELINE_IDX]


def legal_decision_actions(decision_idx: int) -> tuple[int, ...]:
    """Return the authoritative legal actions for one RL decision stage."""
    if int(decision_idx) in NON_SKIPPABLE_DECISION_STAGE_IDXS:
        return (0, 1, 2)
    return (0, 1, 2, 3)


def decision_to_pipeline_stage_idx(decision_idx: int) -> int:
    return RL_DECISION_TO_PIPELINE_STAGE_IDXS[int(decision_idx)]


def pipeline_to_decision_idx(pipeline_stage_idx: int) -> Optional[int]:
    return PIPELINE_TO_RL_DECISION_IDX.get(int(pipeline_stage_idx))


def is_fixed_pipeline_stage_idx(pipeline_stage_idx: int) -> bool:
    return int(pipeline_stage_idx) in FIXED_PIPELINE_STAGE_IDXS


def required_cache_hit_counts(
    cache_hits: Sequence[bool],
    decision_actions: Sequence[int],
) -> tuple[int, int]:
    """Count cache hits only for stages that were physically executed.

    A legal SKIP action does not read a transition and therefore must not be
    reported as a cache miss. Both the current eight-physical-stage layout and
    older decision-stage-only hit vectors are supported.
    """
    hits = [bool(value) for value in cache_hits]
    actions = [int(value) for value in decision_actions]
    if len(hits) == NUM_PIPELINE_STAGES:
        skipped = {
            RL_DECISION_TO_PIPELINE_STAGE_IDXS[decision_idx]
            for decision_idx, action in enumerate(actions[:NUM_RL_STAGES])
            if action == 3
        }
    elif len(hits) == len(actions):
        skipped = {idx for idx, action in enumerate(actions) if action == 3}
    else:
        skipped = set()
    required = [idx for idx in range(len(hits)) if idx not in skipped]
    return sum(hits[idx] for idx in required), len(required)
