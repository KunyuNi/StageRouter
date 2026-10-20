"""Versioned schema contract for the six-tower State V3 vector."""

from __future__ import annotations

from rl import state_schema as v22


STATE_V3_VERSION = "state_v23"

QUERY_DIM = 22
SCHEMA_CONTEXT_DIM = 24
ROUTE_COST_DIM = 57
ARTIFACT_V3_DIM = 62
QUESTION_SEMANTIC_DIM = 1024
SCHEMA_SEMANTIC_DIM = 1024
STATE_V3_DIM = 2213

QUERY_FIELD_NAMES = list(v22.QUERY_FIELD_NAMES)

SCHEMA_CONTEXT_FIELD_NAMES = [
    "static_table_count_log_norm",
    "static_column_count_log_norm",
    "static_primary_key_count_log_norm",
    "static_foreign_key_count_log_norm",
    "static_fk_density",
    "static_component_count_norm",
    "static_isolated_table_ratio",
    "static_wide_table_ratio",
    "static_avg_columns_per_table_norm",
    "static_max_columns_per_table_norm",
    "active_schema_item_count_norm",
    "query_table_top1_similarity",
    "query_table_top3_mean_similarity",
    "query_table_top1_top2_margin",
    "query_table_similarity_entropy",
    "query_column_top1_similarity",
    "query_column_top3_mean_similarity",
    "query_column_top1_top2_margin",
    "query_column_similarity_entropy",
    "query_schema_token_coverage",
    "active_table_ratio",
    "active_column_ratio",
    "active_alignment_delta_from_full",
    "schema_semantic_available",
]

ROUTE_COST_FIELD_NAMES = list(v22.ROUTE_COST_FIELD_NAMES)

ARTIFACT_V3_EXTRA_FIELD_NAMES = [
    "keyword_count_norm",
    "keyword_entity_ratio",
    "keyword_predicate_ratio",
    "keyword_aggregation_ratio",
    "keyword_condition_value_ratio",
    "keyword_ordering_ratio",
    "keyword_mean_confidence",
    "similar_value_count_norm",
    "similar_value_mean_score",
    "similar_value_max_score",
    "similar_column_count_norm",
    "similar_column_table_coverage",
    "tentative_table_reduction_ratio",
    "tentative_column_reduction_ratio",
    "sql_unknown_table_count_norm",
    "sql_unknown_column_count_norm",
    "sql_reference_valid_ratio",
    "sql_referenced_tables_connected_flag",
    "sql_selected_schema_coverage",
    "question_sql_semantic_similarity",
    "sql_sqlite_explain_valid_flag",
]

ARTIFACT_V3_FIELD_NAMES = (
    list(v22.ARTIFACT_FIELD_NAMES) + ARTIFACT_V3_EXTRA_FIELD_NAMES
)

TOWER_SLICES = {
    "query": slice(0, 22),
    "schema_context": slice(22, 46),
    "route_cost": slice(46, 103),
    "artifact": slice(103, 165),
    "question_semantic": slice(165, 1189),
    "schema_semantic": slice(1189, 2213),
}

_TOWER_DIMS = {
    "query": QUERY_DIM,
    "schema_context": SCHEMA_CONTEXT_DIM,
    "route_cost": ROUTE_COST_DIM,
    "artifact": ARTIFACT_V3_DIM,
    "question_semantic": QUESTION_SEMANTIC_DIM,
    "schema_semantic": SCHEMA_SEMANTIC_DIM,
}

_BANNED_NAME_SUBSTRINGS = (
    "gold",
    "reward",
    "difficulty",
    "evaluation",
    "latency",
)


def validate_state_v3_schema() -> None:
    """Validate State V3 dimensions, layout, and structured field names."""
    assert v22.STATE_DIM == 1144, v22.STATE_DIM
    assert QUERY_DIM == v22.QUERY_FEATURE_DIM == len(QUERY_FIELD_NAMES)
    assert SCHEMA_CONTEXT_DIM == len(SCHEMA_CONTEXT_FIELD_NAMES)
    assert ROUTE_COST_DIM == v22.ROUTE_COST_FEATURE_DIM == len(ROUTE_COST_FIELD_NAMES)
    assert ARTIFACT_V3_DIM == len(ARTIFACT_V3_FIELD_NAMES)
    assert len(ARTIFACT_V3_EXTRA_FIELD_NAMES) == 21

    assert QUERY_FIELD_NAMES == list(v22.QUERY_FIELD_NAMES)
    assert ROUTE_COST_FIELD_NAMES == list(v22.ROUTE_COST_FIELD_NAMES)
    assert ARTIFACT_V3_FIELD_NAMES[: v22.ARTIFACT_FEATURE_DIM] == list(
        v22.ARTIFACT_FIELD_NAMES
    )
    assert ARTIFACT_V3_FIELD_NAMES[v22.ARTIFACT_FEATURE_DIM :] == (
        ARTIFACT_V3_EXTRA_FIELD_NAMES
    )

    assert tuple(TOWER_SLICES) == tuple(_TOWER_DIMS)
    ordered_slices = list(TOWER_SLICES.values())
    assert ordered_slices[0].start == 0
    for tower_name, tower_slice in TOWER_SLICES.items():
        assert tower_slice.step is None
        assert tower_slice.stop - tower_slice.start == _TOWER_DIMS[tower_name]
    for left, right in zip(ordered_slices, ordered_slices[1:]):
        assert left.stop == right.start, (left, right)

    assert sum(_TOWER_DIMS.values()) == STATE_V3_DIM
    assert ordered_slices[-1].stop == STATE_V3_DIM

    all_names = (
        QUERY_FIELD_NAMES
        + SCHEMA_CONTEXT_FIELD_NAMES
        + ROUTE_COST_FIELD_NAMES
        + ARTIFACT_V3_FIELD_NAMES
    )
    assert len(all_names) == len(set(all_names)), (
        "state v3 field names must be globally unique"
    )
    for name in all_names:
        lowered = name.lower()
        for banned in _BANNED_NAME_SUBSTRINGS:
            assert banned not in lowered, name


validate_state_v3_schema()
