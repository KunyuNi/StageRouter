"""Frozen layout and feature names for State V25."""

from __future__ import annotations

from collections import OrderedDict


STATE_V25_VERSION = "state_v25"
SEMANTIC_DIM = 1024

TASK_FIELDS = (
    "question_token_count_norm",
    "evidence_token_count_norm",
    "evidence_present",
    "question_has_number",
    "question_has_date_or_time",
    "question_has_superlative",
    "question_has_negation",
    "aggregation_hint_norm",
    "grouping_hint_norm",
    "comparison_hint_norm",
    "ranking_topk_hint_norm",
    "set_operation_hint_norm",
    "nested_query_hint_norm",
    "condition_connector_norm",
    "constraint_density",
)

DATABASE_FIELDS = (
    "full_table_count_log_norm",
    "full_column_count_log_norm",
    "full_primary_key_count_log_norm",
    "full_foreign_key_count_log_norm",
    "full_fk_density",
    "full_component_count_norm",
    "full_isolated_table_ratio",
    "full_wide_table_ratio",
    "full_avg_columns_per_table_norm",
    "full_max_columns_per_table_norm",
)

RETRIEVAL_FIELDS = (
    "retrieval_not_yet",
    "retrieval_ready_empty",
    "retrieval_ready_nonempty",
    "keywords_nonempty",
    "keyword_count_norm",
    "keyword_token_count_norm",
    "similar_values_nonempty",
    "matched_value_count_norm",
    "matched_column_count_norm",
    "matched_table_count_norm",
    "descriptions_nonempty",
    "described_column_count_norm",
    "described_table_count_norm",
    "description_token_count_norm",
    "description_column_coverage",
)

CURRENT_SCHEMA_FIELDS = (
    "schema_source_none",
    "schema_source_full",
    "schema_source_tentative",
    "schema_source_selected_tables",
    "schema_source_selected_columns",
    "schema_empty",
    "current_table_count_log_norm",
    "current_column_count_log_norm",
    "current_pk_count_log_norm",
    "current_fk_count_log_norm",
    "table_retention_ratio",
    "column_retention_ratio",
    "fk_retention_ratio",
    "component_count_norm",
    "isolated_table_ratio",
    "selected_tables_connected",
    "avg_columns_per_table_norm",
    "max_columns_per_table_norm",
    "schema_token_count_norm",
    "schema_too_large",
)

CANDIDATE_SQL_FIELDS = (
    "sql_not_yet",
    "sql_empty",
    "sql_nonempty",
    "sql_token_count_norm",
    "sql_parseable",
    "sql_has_distinct",
    "sql_has_where",
    "sql_has_join",
    "sql_has_group_by",
    "sql_has_having",
    "sql_has_order_by",
    "sql_has_limit",
    "sql_aggregation_count_norm",
    "sql_subquery_depth_norm",
    "sql_has_set_operation",
    "sql_referenced_table_count_norm",
    "sql_referenced_column_count_norm",
    "sql_unknown_table_count_norm",
    "sql_unknown_column_count_norm",
    "sql_reference_valid_ratio",
    "sql_referenced_tables_connected",
)

STAGE_LABELS = ("S1", "S4", "S5", "S6", "S7", "S8")
PROVENANCE_LABELS = ("T1", "T2", "T3", "SKIP", "NOT_EXECUTED")
CONTROL_FIELDS = tuple(f"current_stage_{stage}" for stage in STAGE_LABELS) + tuple(
    f"provenance_{stage}_{label}"
    for stage in STAGE_LABELS
    for label in PROVENANCE_LABELS
)

STRUCTURED_FIELD_NAMES = OrderedDict(
    (
        ("task", TASK_FIELDS),
        ("database", DATABASE_FIELDS),
        ("retrieval", RETRIEVAL_FIELDS),
        ("current_schema", CURRENT_SCHEMA_FIELDS),
        ("candidate_sql", CANDIDATE_SQL_FIELDS),
        ("control", CONTROL_FIELDS),
    )
)

SEMANTIC_DIMS = OrderedDict(
    (
        ("task", SEMANTIC_DIM),
        ("database", SEMANTIC_DIM),
        ("retrieval", SEMANTIC_DIM),
        ("current_schema", SEMANTIC_DIM),
        ("candidate_sql", SEMANTIC_DIM),
        ("control", 0),
    )
)


def _build_slices():
    block_slices = OrderedDict()
    structured_slices = OrderedDict()
    semantic_slices = OrderedDict()
    offset = 0
    for block, fields in STRUCTURED_FIELD_NAMES.items():
        block_start = offset
        structured_slices[block] = slice(offset, offset + len(fields))
        offset += len(fields)
        semantic_slices[block] = slice(offset, offset + SEMANTIC_DIMS[block])
        offset += SEMANTIC_DIMS[block]
        block_slices[block] = slice(block_start, offset)
    return block_slices, structured_slices, semantic_slices, offset


BLOCK_SLICES, STRUCTURED_SLICES, SEMANTIC_SLICES, STATE_V25_DIM = _build_slices()


def validate_state_v25_schema() -> None:
    expected_blocks = (
        "task",
        "database",
        "retrieval",
        "current_schema",
        "candidate_sql",
        "control",
    )
    if tuple(STRUCTURED_FIELD_NAMES) != expected_blocks:
        raise ValueError("State V25 block order changed")
    if tuple(SEMANTIC_DIMS) != expected_blocks:
        raise ValueError("State V25 semantic block order changed")
    names = [name for fields in STRUCTURED_FIELD_NAMES.values() for name in fields]
    if len(names) != 117 or len(set(names)) != len(names):
        raise ValueError("State V25 structured fields must be 117 unique names")
    if tuple(SEMANTIC_DIMS.values()) != (1024, 1024, 1024, 1024, 1024, 0):
        raise ValueError("State V25 semantic widths changed")
    cursor = 0
    for block in expected_blocks:
        block_slice = BLOCK_SLICES[block]
        structured_slice = STRUCTURED_SLICES[block]
        semantic_slice = SEMANTIC_SLICES[block]
        if block_slice.start != cursor or structured_slice.start != cursor:
            raise ValueError("State V25 slices contain a gap or overlap")
        if structured_slice.stop != semantic_slice.start:
            raise ValueError("State V25 structured and semantic slices are not contiguous")
        if semantic_slice.stop != block_slice.stop:
            raise ValueError("State V25 block slice does not cover its fields")
        cursor = block_slice.stop
    if cursor != 5237 or STATE_V25_DIM != 5237:
        raise ValueError("State V25 total dimension must be 5237")


validate_state_v25_schema()
