"""Causal pre-decision context and structured features for State V25."""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from rl.state_v3_context import (
    StaticSchemaContext,
    analyze_sql_quality,
    compute_graph_stats,
    payload_get,
    visible_descriptions_for_stage,
)
from rl.state_v25_schema import STRUCTURED_FIELD_NAMES


AGG = ("average", "avg", "count", "how many", "maximum", "minimum", "max", "min", "sum", "total")
GROUP = ("each", "every", "for each", "group by", "grouped by", "per")
COMPARE = (
    "after", "at least", "at most", "before", "between", "equal",
    "fewer than", "greater than", "less than", "more than",
    "no later than", "no more than",
)
RANK = ("bottom", "first", "highest", "last", "least", "lowest", "most", "rank", "top")
SET = ("both", "either", "except", "in common", "intersection", "neither", "not in", "union")
NESTED = ("exists", "for which", "not exists", "whose", "with no", "without any")
CONNECTOR = ("and", "but", "except", "nor", "or", "without")
NEGATION = ("except", "no", "not", "never", "neither", "nor", "without")
SUPERLATIVE = (
    "best", "biggest", "earliest", "fastest", "highest", "largest",
    "latest", "least", "longest", "lowest", "maximum", "minimum",
    "most", "shortest", "smallest", "top",
)
DATE_TIME = ("date", "day", "hour", "minute", "month", "quarter", "time", "week", "year")

NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")
ISO_DATE_TIME_RE = re.compile(
    r"(?:\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}\b|\b\d{4}\b)"
)
SCHEMA_SOURCES = ("none", "full", "tentative", "selected_tables", "selected_columns")
ACTION_LABELS = ("T1", "T2", "T3", "SKIP")
PROVENANCE_LABELS = ACTION_LABELS + ("NOT_EXECUTED",)


def normalize_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def lin(value: float, maximum: float) -> float:
    return clip01(float(value) / float(maximum))


def logn(value: float, maximum: float) -> float:
    return clip01(math.log1p(max(0.0, float(value))) / math.log1p(float(maximum)))


def _pattern(patterns: Sequence[str]) -> re.Pattern[str]:
    choices = "|".join(re.escape(item) for item in sorted(patterns, key=lambda item: (-len(item), item)))
    return re.compile(rf"(?<![\w])(?:{choices})(?![\w])", re.IGNORECASE)


_PATTERNS = {
    name: _pattern(values)
    for name, values in {
        "agg": AGG,
        "group": GROUP,
        "compare": COMPARE,
        "rank": RANK,
        "set": SET,
        "nested": NESTED,
        "connector": CONNECTOR,
        "negation": NEGATION,
        "superlative": SUPERLATIVE,
        "date_time": DATE_TIME,
    }.items()
}


def hits(text: str, pattern_name: str) -> int:
    return sum(1 for _ in _PATTERNS[pattern_name].finditer(normalize_text(text)))


@dataclass(frozen=True, eq=False)
class StateV25Context:
    stage_idx: int
    schema_source: str
    current_schema: Mapping[str, list[str]]
    retrieval_status: str
    candidate_sql: str
    provenance: tuple[str, ...]
    structured: Mapping[str, np.ndarray]
    semantic_texts: Mapping[str, str]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateV25Context):
            return NotImplemented
        return (
            self.stage_idx == other.stage_idx
            and self.schema_source == other.schema_source
            and dict(self.current_schema) == dict(other.current_schema)
            and self.retrieval_status == other.retrieval_status
            and self.candidate_sql == other.candidate_sql
            and self.provenance == other.provenance
            and dict(self.semantic_texts) == dict(other.semantic_texts)
            and all(
                np.array_equal(self.structured[name], other.structured[name])
                for name in self.structured
            )
        )


def _full_schema(static: StaticSchemaContext) -> dict[str, list[str]]:
    return {table: list(columns) for table, columns in static.tables.items()}


def _filtered_schema(raw: Any, static: StaticSchemaContext) -> dict[str, list[str]]:
    if not isinstance(raw, Mapping):
        return {}
    table_lookup = {table.casefold(): table for table in static.tables}
    result: dict[str, list[str]] = {}
    for raw_table, raw_columns in raw.items():
        table = table_lookup.get(normalize_text(raw_table).casefold())
        if table is None or isinstance(raw_columns, str) or not isinstance(raw_columns, Sequence):
            continue
        column_lookup = {column.casefold(): column for column in static.tables[table]}
        columns: list[str] = []
        for raw_column in raw_columns:
            column = column_lookup.get(normalize_text(raw_column).casefold())
            if column is not None and column not in columns:
                columns.append(column)
        result[table] = columns
    return _sorted_schema(result)


def _selected_table_schema(raw: Any, static: StaticSchemaContext) -> dict[str, list[str]]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        return {}
    table_lookup = {table.casefold(): table for table in static.tables}
    result = {}
    for raw_table in raw:
        table = table_lookup.get(normalize_text(raw_table).casefold())
        if table is not None:
            result[table] = list(static.tables[table])
    return _sorted_schema(result)


def _sorted_schema(schema: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    return {
        table: sorted((str(column) for column in columns), key=str.casefold)
        for table, columns in sorted(schema.items(), key=lambda item: str(item[0]).casefold())
    }


def _current_schema(payload: Any, stage_idx: int, static: StaticSchemaContext):
    if stage_idx == 0:
        return "none", {}
    if stage_idx in (1, 5):
        return "full", _sorted_schema(_full_schema(static))
    if stage_idx == 2:
        return "tentative", _filtered_schema(payload_get(payload, "tentative_schema", {}), static)
    if stage_idx == 3:
        return "selected_tables", _selected_table_schema(payload_get(payload, "selected_tables", ()), static)
    if stage_idx == 4:
        return "selected_columns", _filtered_schema(payload_get(payload, "tentative_schema", {}), static)
    raise ValueError("State V25 stage_idx must be in [0, 5]")


def _candidate_sql(payload: Any, stage_idx: int) -> str:
    if stage_idx != 5:
        return ""
    candidates = payload_get(payload, "candidates", ())
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)) and candidates:
        return normalize_text(payload_get(candidates[0], "sql", ""))
    return ""


def _token_count(token_count: Callable[[str], int], text: str) -> int:
    value = token_count(text)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token_count must return a non-negative integer")
    return value


def _task_features(question: str, evidence: str, token_count: Callable[[str], int]) -> np.ndarray:
    question_tokens = _token_count(token_count, question)
    number_count = len(NUMBER_RE.findall(question))
    compare_count = hits(question, "compare")
    negation_count = hits(question, "negation")
    connector_count = hits(question, "connector")
    constraint_hits = number_count + compare_count + negation_count + connector_count
    return _array(
        (
            lin(question_tokens, 128),
            lin(_token_count(token_count, evidence), 256),
            bool(evidence),
            number_count > 0,
            hits(question, "date_time") > 0 or bool(ISO_DATE_TIME_RE.search(question)),
            hits(question, "superlative") > 0,
            negation_count > 0,
            lin(hits(question, "agg"), 4),
            lin(hits(question, "group"), 4),
            lin(compare_count, 6),
            lin(hits(question, "rank"), 4),
            lin(hits(question, "set"), 4),
            lin(hits(question, "nested"), 4),
            lin(connector_count, 6),
            clip01(8.0 * constraint_hits / max(1, question_tokens)),
        )
    )


def _database_features(static: StaticSchemaContext) -> np.ndarray:
    table_count = static.table_count
    column_count = static.column_count
    graph = static.graph_stats(tuple(static.tables))
    widths = [len(columns) for columns in static.tables.values()]
    return _array(
        (
            logn(table_count, 128),
            logn(column_count, 2048),
            logn(static.primary_key_count, 512),
            logn(static.foreign_key_count, 512),
            0.0 if table_count < 2 else clip01(2.0 * graph.edge_count / (table_count * (table_count - 1))),
            lin(graph.component_count, 32),
            graph.isolated_table_count / max(1, table_count),
            sum(width >= 20 for width in widths) / max(1, table_count),
            lin(column_count / max(1, table_count), 64),
            lin(max(widths, default=0), 256),
        )
    )


def _keyword_records(payload: Any) -> list[str]:
    keywords = payload_get(payload, "keywords", ())
    if isinstance(keywords, str) or not isinstance(keywords, Sequence):
        return []
    values = []
    for item in keywords:
        value = normalize_text(payload_get(item, "value", item if isinstance(item, str) else ""))
        if value:
            values.append(value)
    return values


def _matched_value_records(payload: Any) -> list[tuple[str, str, str]]:
    raw = payload_get(payload, "similar_values", {})
    if not isinstance(raw, Mapping):
        return []
    records = []
    for table, columns in raw.items():
        if not isinstance(columns, Mapping):
            continue
        for column, values in columns.items():
            if isinstance(values, str) or not isinstance(values, Sequence):
                continue
            for item in values:
                value = normalize_text(payload_get(item, "value", item if isinstance(item, str) else ""))
                if value:
                    records.append((normalize_text(table), normalize_text(column), value))
    return records


def _description_records(payload: Any, stage_idx: int, static: StaticSchemaContext) -> list[tuple[str, str, str]]:
    if stage_idx == 0:
        return []
    visible = visible_descriptions_for_stage(payload, stage_idx, static)
    records = []
    for table, columns in visible.items():
        for column, description in columns.items():
            text = normalize_text(description)
            if text:
                records.append((normalize_text(table), normalize_text(column), text))
    return records


def _retrieval(
    payload: Any,
    stage_idx: int,
    static: StaticSchemaContext,
    token_count: Callable[[str], int],
):
    keywords = _keyword_records(payload) if stage_idx > 0 else []
    matched = _matched_value_records(payload) if stage_idx > 0 else []
    descriptions = _description_records(payload, stage_idx, static)
    nonempty = bool(keywords or matched or descriptions)
    status = "not_yet" if stage_idx == 0 else ("ready_nonempty" if nonempty else "ready_empty")
    matched_columns = {(table.casefold(), column.casefold()) for table, column, _ in matched}
    matched_tables = {table for table, _ in matched_columns}
    described_columns = {(table.casefold(), column.casefold()) for table, column, _ in descriptions}
    described_tables = {table for table, _ in described_columns}
    joined_keywords = "\n".join(keywords)
    joined_descriptions = "\n".join(text for _, _, text in descriptions)
    features = _array(
        (
            status == "not_yet",
            status == "ready_empty",
            status == "ready_nonempty",
            bool(keywords),
            lin(len(keywords), 32),
            lin(_token_count(token_count, joined_keywords), 256),
            bool(matched),
            lin(len(matched), 128),
            lin(len(matched_columns), 64),
            lin(len(matched_tables), 32),
            bool(descriptions),
            logn(len(described_columns), 512),
            logn(len(described_tables), 128),
            logn(_token_count(token_count, joined_descriptions), 8192),
            len(described_columns) / max(1, static.column_count),
        )
    )
    lines = ["retrieval"]
    lines.extend(f"keyword: {value}" for value in sorted(keywords, key=str.casefold))
    lines.extend(
        f"value: {table}.{column} = {value}"
        for table, column, value in sorted(matched, key=lambda row: tuple(part.casefold() for part in row))
    )
    lines.extend(
        f"description: {table}.{column} | {text}"
        for table, column, text in sorted(descriptions, key=lambda row: tuple(part.casefold() for part in row))
    )
    return status, features, "\n".join(lines) if nonempty else ""


def _schema_column_lookup(static: StaticSchemaContext):
    return {(column.table.casefold(), column.name.casefold()): column for column in static.columns}


def _visible_fk_constraints(static: StaticSchemaContext, schema: Mapping[str, Sequence[str]]):
    selected = {
        table.casefold(): {column.casefold() for column in columns}
        for table, columns in schema.items()
    }
    constraints = []
    if not selected:
        return constraints
    with sqlite3.connect(f"{static.db_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        for table in static.tables:
            if table.casefold() not in selected:
                continue
            quoted = '"' + table.replace('"', '""') + '"'
            grouped: dict[int, list[tuple[str, str, str]]] = {}
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted})"):
                grouped.setdefault(int(row[0]), []).append((str(row[3]), str(row[2]), str(row[4] or "")))
            for constraint_id, parts in grouped.items():
                source_visible = all(source.casefold() in selected[table.casefold()] for source, _, _ in parts)
                target_visible = all(
                    target_table.casefold() in selected
                    and (not target_column or target_column.casefold() in selected[target_table.casefold()])
                    for _, target_table, target_column in parts
                )
                if source_visible and target_visible:
                    constraints.append((table, constraint_id, parts[0][1]))
    return constraints


def _canonical_schema_text(role: str, schema: Mapping[str, Sequence[str]], static: StaticSchemaContext) -> str:
    if not schema:
        return ""
    lookup = _schema_column_lookup(static)
    lines = [role]
    for table, columns in _sorted_schema(schema).items():
        lines.append(f"table: {table}")
        for column_name in columns:
            column = lookup[(table.casefold(), column_name.casefold())]
            foreign_keys = ",".join(
                f"{target_table}.{target_column}" for target_table, target_column in column.foreign_keys
            )
            lines.append(
                "column: "
                f"{table}.{column.name} | type={column.data_type} | "
                f"pk={int(column.primary_key)} | fk={foreign_keys or '-'}"
            )
    return "\n".join(lines)


def _schema_features(
    schema_source: str,
    schema: Mapping[str, Sequence[str]],
    semantic_text: str,
    static: StaticSchemaContext,
    token_count: Callable[[str], int],
) -> np.ndarray:
    table_count = len(schema)
    column_count = sum(len(columns) for columns in schema.values())
    lookup = _schema_column_lookup(static)
    primary_keys = sum(
        int(lookup[(table.casefold(), column.casefold())].primary_key)
        for table, columns in schema.items()
        for column in columns
    )
    fk_constraints = _visible_fk_constraints(static, schema)
    edges = sorted({tuple(sorted((source, target))) for source, _, target in fk_constraints if source.casefold() != target.casefold()})
    graph = compute_graph_stats(edges, tuple(schema))
    widths = [len(columns) for columns in schema.values()]
    schema_tokens = _token_count(token_count, semantic_text)
    source_onehot = [float(schema_source == source) for source in SCHEMA_SOURCES]
    values = source_onehot + [
        float(table_count == 0 or column_count == 0),
        logn(table_count, 128),
        logn(column_count, 2048),
        logn(primary_keys, 512),
        logn(len(fk_constraints), 512),
        clip01(table_count / max(1, static.table_count)),
        clip01(column_count / max(1, static.column_count)),
        1.0 if not fk_constraints and static.foreign_key_count == 0 else clip01(len(fk_constraints) / max(1, static.foreign_key_count)),
        lin(graph.component_count, 32),
        graph.isolated_table_count / max(1, table_count),
        float(table_count > 0 and graph.component_count == 1),
        lin(column_count / max(1, table_count), 64),
        lin(max(widths, default=0), 256),
        lin(schema_tokens, 8192),
        float(schema_tokens > 8192),
    ]
    return _array(values)


def _sql_structure(sql: str):
    if not sql:
        return None, {}
    try:
        parsed = parse_one(sql, read="sqlite")
    except (SqlglotError, RecursionError):
        return None, {}
    if parsed is None:
        return None, {}
    selects = list(parsed.find_all(exp.Select))
    aggregate_types = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
    max_depth = 0
    for select in selects:
        depth = 0
        parent = select.parent
        while parent is not None:
            depth += int(isinstance(parent, exp.Select))
            parent = parent.parent
        max_depth = max(max_depth, depth)
    tables = list(parsed.find_all(exp.Table))
    return parsed, {
        "distinct": any(select.args.get("distinct") is not None for select in selects),
        "where": any(select.args.get("where") is not None for select in selects),
        "join": bool(list(parsed.find_all(exp.Join))) or len(tables) > 1,
        "group": any(select.args.get("group") is not None for select in selects),
        "having": any(select.args.get("having") is not None for select in selects),
        "order": any(select.args.get("order") is not None for select in selects),
        "limit": any(select.args.get("limit") is not None for select in selects),
        "aggregates": sum(1 for node in parsed.walk() if isinstance(node, aggregate_types)),
        "subquery_depth": max_depth,
        "set_operation": any(isinstance(node, (exp.Union, exp.Intersect, exp.Except)) for node in parsed.walk()),
    }


def _candidate_sql_features(
    sql: str,
    stage_idx: int,
    static: StaticSchemaContext,
    token_count: Callable[[str], int],
) -> np.ndarray:
    status = "not_yet" if stage_idx < 5 else ("nonempty" if sql else "empty")
    parsed, structure = _sql_structure(sql)
    quality = analyze_sql_quality(sql, static) if sql else None
    known_tables = () if quality is None else quality.referenced_tables
    values = (
        status == "not_yet",
        status == "empty",
        status == "nonempty",
        lin(_token_count(token_count, sql), 512),
        parsed is not None,
        structure.get("distinct", False),
        structure.get("where", False),
        structure.get("join", False),
        structure.get("group", False),
        structure.get("having", False),
        structure.get("order", False),
        structure.get("limit", False),
        lin(structure.get("aggregates", 0), 8),
        lin(structure.get("subquery_depth", 0), 4),
        structure.get("set_operation", False),
        lin(len(known_tables), 32),
        lin(0 if quality is None else len(quality.referenced_columns), 128),
        lin(0 if quality is None else quality.unknown_table_count, 16),
        lin(0 if quality is None else quality.unknown_column_count, 32),
        0.0 if quality is None else quality.reference_valid_ratio,
        bool(known_tables) and bool(quality.referenced_tables_connected),
    )
    return _array(values)


def _control(stage_idx: int, actions: Sequence[int]):
    if len(actions) != stage_idx:
        raise ValueError("actions must contain exactly the completed router decisions")
    if any(isinstance(action, bool) or not isinstance(action, int) or action not in range(4) for action in actions):
        raise ValueError("actions must contain integer action ids in [0, 3]")
    provenance = tuple(
        ACTION_LABELS[actions[index]] if index < len(actions) else "NOT_EXECUTED"
        for index in range(6)
    )
    values = [float(index == stage_idx) for index in range(6)]
    for selected in provenance:
        values.extend(float(selected == label) for label in PROVENANCE_LABELS)
    return provenance, _array(values)


def _array(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError("State V25 structured features must be finite values in [0, 1]")
    return array


def build_state_v25_context(
    payload: Any,
    stage_idx: int,
    actions: Sequence[int],
    static: StaticSchemaContext,
    *,
    token_count: Callable[[str], int],
) -> StateV25Context:
    """Build a causal State V25 context for one pre-decision state."""
    if isinstance(stage_idx, bool) or not isinstance(stage_idx, int) or stage_idx not in range(6):
        raise ValueError("State V25 stage_idx must be an integer in [0, 5]")
    question = normalize_text(payload_get(payload, "question", ""))
    evidence = normalize_text(payload_get(payload, "evidence", ""))
    schema_source, current_schema = _current_schema(payload, stage_idx, static)
    candidate_sql = _candidate_sql(payload, stage_idx)
    retrieval_status, retrieval_features, retrieval_text = _retrieval(
        payload, stage_idx, static, token_count
    )
    database_text = _canonical_schema_text("database", _full_schema(static), static)
    schema_text = _canonical_schema_text("current_schema", current_schema, static)
    provenance, control_features = _control(stage_idx, actions)
    structured = {
        "task": _task_features(question, evidence, token_count),
        "database": _database_features(static),
        "retrieval": retrieval_features,
        "current_schema": _schema_features(
            schema_source, current_schema, schema_text, static, token_count
        ),
        "candidate_sql": _candidate_sql_features(
            candidate_sql, stage_idx, static, token_count
        ),
        "control": control_features,
    }
    for block, fields in STRUCTURED_FIELD_NAMES.items():
        if structured[block].shape != (len(fields),):
            raise AssertionError(f"State V25 {block} structured width changed")
    task_text = question if not evidence else f"{question} [SEP] {evidence}"
    semantic_texts = {
        "task": f"task\n{task_text}" if task_text else "",
        "database": database_text,
        "retrieval": retrieval_text,
        "current_schema": schema_text,
        "candidate_sql": f"candidate_sql\n{candidate_sql}" if candidate_sql else "",
    }
    return StateV25Context(
        stage_idx=stage_idx,
        schema_source=schema_source,
        current_schema=current_schema,
        retrieval_status=retrieval_status,
        candidate_sql=candidate_sql,
        provenance=provenance,
        structured=structured,
        semantic_texts=semantic_texts,
    )
