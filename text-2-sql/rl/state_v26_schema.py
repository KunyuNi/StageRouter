"""Frozen State V26 extension over the State V25 vector layout."""

from __future__ import annotations

from collections import OrderedDict

from .state_v25_schema import STATE_V25_DIM


STATE_V26_VERSION = "state_v26"

COST_CONTEXT_FIELDS = (
    "operator_work_units_norm",
    "operator_call_count_norm",
    "operator_input_tokens_norm",
    "operator_output_tokens_prior_norm",
    "action_cost_prior_T1_norm",
    "action_cost_prior_T2_norm",
    "action_cost_prior_T3_norm",
    "incremental_cost_T3_to_T2_norm",
    "incremental_cost_T2_to_T1_norm",
    "cost_prior_confidence",
    "cost_prior_valid",
)

S4_DIFFICULTY_CONTEXT_FIELDS = (
    "question_column_similarity_mean",
    "question_column_similarity_max",
    "question_column_similarity_std",
    "top1_top2_similarity_margin",
    "column_relevance_entropy",
    "retrieval_evidence_coverage",
    "profile_missing_ratio",
    "difficulty_context_valid",
)

CONTEXT_FIELD_NAMES = OrderedDict(
    (
        ("cost_context", COST_CONTEXT_FIELDS),
        ("s4_difficulty_context", S4_DIFFICULTY_CONTEXT_FIELDS),
    )
)


def _build_context_slices() -> tuple[OrderedDict[str, slice], int]:
    slices: OrderedDict[str, slice] = OrderedDict()
    offset = STATE_V25_DIM
    for block, fields in CONTEXT_FIELD_NAMES.items():
        slices[block] = slice(offset, offset + len(fields))
        offset += len(fields)
    return slices, offset


CONTEXT_SLICES, STATE_V26_DIM = _build_context_slices()
STATE_V25_SLICE = slice(0, STATE_V25_DIM)


def validate_state_v26_schema() -> None:
    """Fail closed if the additive State V26 contract drifts."""
    if tuple(CONTEXT_FIELD_NAMES) != (
        "cost_context",
        "s4_difficulty_context",
    ):
        raise ValueError("State V26 context block order changed")
    names = [name for fields in CONTEXT_FIELD_NAMES.values() for name in fields]
    if len(names) != 19 or len(set(names)) != len(names):
        raise ValueError("State V26 context fields must be 19 unique names")
    cursor = STATE_V25_DIM
    for block in CONTEXT_FIELD_NAMES:
        block_slice = CONTEXT_SLICES[block]
        if block_slice.start != cursor:
            raise ValueError("State V26 context slices contain a gap or overlap")
        cursor = block_slice.stop
    if cursor != 5256 or STATE_V26_DIM != 5256:
        raise ValueError("State V26 total dimension must be 5256")


validate_state_v26_schema()
