"""
Central state schema for the NL2SQL router.

The physical state remains a flat vector:
    [query(22) | route_cost(57) | artifact(41) | semantic(1024)]

Policy can read all towers. logZ can read only schema-static query features
and semantic features.
"""

from __future__ import annotations

ACTION_DIM = 4
STAGE_COUNT = 6
S7_STAGE_IDX = 4

QUERY_FEATURE_DIM = 22
ROUTE_COST_FEATURE_DIM = 57
ARTIFACT_FEATURE_DIM = 41
STRUCTURED_STATE_DIM = QUERY_FEATURE_DIM + ROUTE_COST_FEATURE_DIM + ARTIFACT_FEATURE_DIM

SEMANTIC_DIM = 1024
STATE_DIM = STRUCTURED_STATE_DIM + SEMANTIC_DIM

QUERY_SLICE = slice(0, QUERY_FEATURE_DIM)
ROUTE_COST_SLICE = slice(QUERY_SLICE.stop, QUERY_SLICE.stop + ROUTE_COST_FEATURE_DIM)
ARTIFACT_SLICE = slice(ROUTE_COST_SLICE.stop, ROUTE_COST_SLICE.stop + ARTIFACT_FEATURE_DIM)
SEMANTIC_SLICE = slice(ARTIFACT_SLICE.stop, ARTIFACT_SLICE.stop + SEMANTIC_DIM)

SCHEMA_STATIC_QUERY_OFFSET = 6
SCHEMA_STATIC_QUERY_SLICE = slice(SCHEMA_STATIC_QUERY_OFFSET, QUERY_FEATURE_DIM)
SCHEMA_STATIC_STATE_SLICE = slice(QUERY_SLICE.start + SCHEMA_STATIC_QUERY_OFFSET, QUERY_SLICE.stop)

LOGZ_INPUT_SLICES = (SCHEMA_STATIC_STATE_SLICE, SEMANTIC_SLICE)

QUERY_FIELD_NAMES = [
    "question_length_norm",
    "evidence_length_norm",
    "question_has_number",
    "question_has_date_or_time",
    "question_has_superlative",
    "question_has_negation",
    "log_table_count_norm",
    "log_column_count_norm",
    "log_fk_count_norm",
    "fk_density",
    "avg_columns_per_table_norm",
    "max_columns_per_table_norm",
    "schema_has_fk",
    "schema_many_table_flag",
    "query_aggregation_hint_norm",
    "query_grouping_pattern_hint_norm",
    "query_comparison_hint_norm",
    "query_ranking_hint_norm",
    "query_set_pattern_hint_norm",
    "query_nested_pattern_hint_norm",
    "query_condition_connector_norm",
    "query_constraint_density",
]

ROUTE_COST_FIELD_NAMES = (
    [f"stage_onehot_{i}" for i in range(STAGE_COUNT)]
    + [f"action_mask_{i}" for i in range(ACTION_DIM)]
    + [f"tier_history_s{i}_{j}" for i in range(STAGE_COUNT) for j in range(5)]
    + [f"cost_per_stage_{i}" for i in range(STAGE_COUNT)]
    + [
        "total_cost_so_far_norm",
        "s7_cost_norm",
        "s8_cost_norm",
        "max_stage_cost_norm",
        "recent_cost_trend",
        "count_t1_norm",
        "count_t2_norm",
        "count_t3_norm",
        "count_skip_norm",
        "skip_ratio",
        "consecutive_skip_flag",
    ]
)

TIER_HISTORY_5WAY = {
    "T1": [1, 0, 0, 0, 0],
    "T2": [0, 1, 0, 0, 0],
    "T3": [0, 0, 1, 0, 0],
    "Skip": [0, 0, 0, 1, 0],
    "NotExecuted": [0, 0, 0, 0, 1],
}

ARTIFACT_FIELD_NAMES = [
    "has_tentative_schema",
    "selected_table_count_norm",
    "selected_column_count_norm",
    "selected_fk_count_norm",
    "selected_table_ratio",
    "selected_column_ratio",
    "selected_tables_connected_flag",
    "min_join_path_length_norm",
    "join_graph_component_count_norm",
    "schema_empty_flag",
    "schema_too_large_flag",
    "has_entity_retrieval",
    "entity_count_norm",
    "has_context_retrieval",
    "context_item_count_norm",
    "avail_has_question",
    "avail_has_evidence",
    "avail_has_full_schema",
    "avail_has_tentative_schema",
    "avail_has_candidate_sql",
    "avail_has_final_sql",
    "avail_has_execution_feedback",
    "avail_has_revision_output",
    "has_generated_sql",
    "sql_length_norm",
    "sql_has_where",
    "sql_has_join",
    "sql_has_group_by",
    "sql_has_order_by",
    "sql_has_limit",
    "sql_has_aggregation",
    "sql_table_count_norm",
    "sql_parseable_flag",
    "has_execution_feedback",
    "execution_success_observed",
    "error_none",
    "error_syntax",
    "error_schema",
    "error_runtime",
    "error_empty_result",
    "error_timeout",
]

TOWER_LAYOUT = {
    "query": (QUERY_SLICE, True, False),
    "route_cost": (ROUTE_COST_SLICE, True, False),
    "artifact": (ARTIFACT_SLICE, True, False),
    "semantic": (SEMANTIC_SLICE, True, True),
}

_BANNED_NAME_SUBSTRINGS = ("reserved", "padding", "unused", "dummy")


def _validate_schema() -> None:
    assert ACTION_DIM == 4
    assert STAGE_COUNT == 6
    assert S7_STAGE_IDX == 4

    assert QUERY_FEATURE_DIM == 22, QUERY_FEATURE_DIM
    assert ROUTE_COST_FEATURE_DIM == 57, ROUTE_COST_FEATURE_DIM
    assert ARTIFACT_FEATURE_DIM == 41, ARTIFACT_FEATURE_DIM
    assert STRUCTURED_STATE_DIM == 120, STRUCTURED_STATE_DIM
    assert SEMANTIC_DIM == 1024, SEMANTIC_DIM
    assert STATE_DIM == 1144, STATE_DIM

    assert len(QUERY_FIELD_NAMES) == QUERY_FEATURE_DIM, len(QUERY_FIELD_NAMES)
    assert len(ROUTE_COST_FIELD_NAMES) == ROUTE_COST_FEATURE_DIM, len(ROUTE_COST_FIELD_NAMES)
    assert len(ARTIFACT_FIELD_NAMES) == ARTIFACT_FEATURE_DIM, len(ARTIFACT_FIELD_NAMES)

    ordered = [QUERY_SLICE, ROUTE_COST_SLICE, ARTIFACT_SLICE, SEMANTIC_SLICE]
    assert ordered[0].start == 0
    for left, right in zip(ordered, ordered[1:]):
        assert left.stop == right.start, (left, right)
    assert ordered[-1].stop == STATE_DIM, ordered[-1].stop

    assert SCHEMA_STATIC_QUERY_SLICE == slice(6, 22)
    assert SCHEMA_STATIC_STATE_SLICE == slice(6, 22)
    assert LOGZ_INPUT_SLICES == (SCHEMA_STATIC_STATE_SLICE, SEMANTIC_SLICE)

    all_names = QUERY_FIELD_NAMES + list(ROUTE_COST_FIELD_NAMES) + ARTIFACT_FIELD_NAMES
    assert len(set(all_names)) == len(all_names), "state field names must be globally unique"
    for name in all_names:
        lowered = name.lower()
        for banned in _BANNED_NAME_SUBSTRINGS:
            assert banned not in lowered, name


_validate_schema()


if __name__ == "__main__":
    _validate_schema()
    print("state_schema OK")
    print(f"  QUERY_FEATURE_DIM      = {QUERY_FEATURE_DIM} -> {QUERY_SLICE}")
    print(f"  ROUTE_COST_FEATURE_DIM = {ROUTE_COST_FEATURE_DIM} -> {ROUTE_COST_SLICE}")
    print(f"  ARTIFACT_FEATURE_DIM   = {ARTIFACT_FEATURE_DIM} -> {ARTIFACT_SLICE}")
    print(f"  SEMANTIC_DIM           = {SEMANTIC_DIM} -> {SEMANTIC_SLICE}")
    print(f"  STRUCTURED_STATE_DIM   = {STRUCTURED_STATE_DIM}")
    print(f"  STATE_DIM              = {STATE_DIM}")
