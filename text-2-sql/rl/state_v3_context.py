"""Static schema context and stage-visible State V3 artifacts."""

from __future__ import annotations

import sqlite3
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from env.database_utils.db_catalog.csv_utils import (
    load_tables_concatenated_description,
)


_STAGE_INDICES = range(6)


def payload_get(payload: Any, key: str, default: Any = None) -> Any:
    """Read a payload field consistently from mappings and objects."""
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return getattr(payload, key, default)


@dataclass(frozen=True)
class ColumnContext:
    table: str
    name: str
    data_type: str
    description: str
    primary_key: bool
    foreign_keys: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SchemaGraphStats:
    table_count: int
    edge_count: int
    component_count: int
    isolated_table_count: int
    connected: bool
    min_connection_edges: int
    density: float


@dataclass(frozen=True)
class StaticSchemaContext:
    db_id: str
    db_path: Path
    columns: tuple[ColumnContext, ...]
    tables: Mapping[str, tuple[str, ...]]
    table_edges: tuple[tuple[str, str], ...]
    primary_key_count: int
    foreign_key_count: int

    def __post_init__(self) -> None:
        frozen_tables = {
            str(table): tuple(str(column) for column in columns)
            for table, columns in self.tables.items()
        }
        object.__setattr__(self, "db_path", Path(self.db_path))
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "tables", MappingProxyType(frozen_tables))
        object.__setattr__(self, "table_edges", tuple(self.table_edges))

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def column_count(self) -> int:
        return len(self.columns)

    @property
    def component_count(self) -> int:
        return self.graph_stats(tuple(self.tables)).component_count

    @property
    def isolated_table_count(self) -> int:
        return self.graph_stats(tuple(self.tables)).isolated_table_count

    def graph_stats(self, selected_tables: Sequence[str]) -> SchemaGraphStats:
        return compute_graph_stats(self.table_edges, selected_tables)


@dataclass(frozen=True)
class SqlQuality:
    parseable: bool
    unknown_table_count: int
    unknown_column_count: int
    reference_valid_ratio: float
    referenced_tables_connected: bool
    selected_schema_coverage: float
    sqlite_explain_valid: bool
    referenced_tables: tuple[str, ...] = ()
    referenced_columns: tuple[tuple[str, str], ...] = ()
    unknown_tables: tuple[str, ...] = ()
    unknown_columns: tuple[str, ...] = ()
    valid_reference_count: int = 0
    total_reference_count: int = 0


class StaticSchemaContextProvider:
    """Load immutable SQLite schema metadata without global DB state."""

    def __init__(self, db_root: str | Path, db_mode: str):
        self.db_root = Path(db_root)
        self.db_mode = str(db_mode)
        self._cache: dict[str, StaticSchemaContext] = {}

    def get(self, db_id: str) -> StaticSchemaContext:
        db_id = str(db_id)
        if db_id not in self._cache:
            self._cache[db_id] = self._load(db_id)
        return self._cache[db_id]

    def _database_directory(self, db_id: str) -> Path:
        mode_directory = f"{self.db_mode}_databases"
        if self.db_root.name.casefold() == mode_directory.casefold():
            return self.db_root / db_id
        return self.db_root / mode_directory / db_id

    def _load(self, db_id: str) -> StaticSchemaContext:
        db_directory = self._database_directory(db_id)
        db_path = db_directory / f"{db_id}.sqlite"
        if not db_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        descriptions = _load_descriptions(db_directory)
        with sqlite3.connect(_readonly_uri(db_path), uri=True) as connection:
            table_names = sorted(
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if not str(row[0]).casefold().startswith("sqlite_")
            )
            table_lookup = {name.casefold(): name for name in table_names}
            table_columns: dict[str, tuple[str, ...]] = {}
            columns: list[ColumnContext] = []
            table_edges: set[tuple[str, str]] = set()
            foreign_key_constraints: set[tuple[str, int]] = set()
            primary_key_count = 0

            for table in table_names:
                quoted_table = _quote_identifier(table)
                column_rows = connection.execute(
                    f"PRAGMA table_info({quoted_table})"
                ).fetchall()
                foreign_key_rows = connection.execute(
                    f"PRAGMA foreign_key_list({quoted_table})"
                ).fetchall()

                foreign_keys_by_column: dict[
                    str, list[tuple[str, str]]
                ] = {}
                for row in foreign_key_rows:
                    constraint_id = int(row[0])
                    target_raw = str(row[2])
                    source_column = str(row[3])
                    target_column = "" if row[4] is None else str(row[4])
                    target_table = table_lookup.get(
                        target_raw.casefold(), target_raw
                    )
                    foreign_key_constraints.add((table.casefold(), constraint_id))
                    foreign_keys_by_column.setdefault(
                        source_column.casefold(), []
                    ).append((target_table, target_column))

                    real_target = table_lookup.get(target_raw.casefold())
                    if real_target is not None and real_target != table:
                        table_edges.add(tuple(sorted((table, real_target))))

                names: list[str] = []
                for row in column_rows:
                    column_name = str(row[1])
                    is_primary_key = bool(row[5])
                    names.append(column_name)
                    primary_key_count += int(is_primary_key)
                    columns.append(
                        ColumnContext(
                            table=table,
                            name=column_name,
                            data_type=str(row[2] or ""),
                            description=descriptions.get(
                                table.casefold(), {}
                            ).get(column_name.casefold(), ""),
                            primary_key=is_primary_key,
                            foreign_keys=tuple(
                                foreign_keys_by_column.get(
                                    column_name.casefold(), ()
                                )
                            ),
                        )
                    )
                table_columns[table] = tuple(names)

        return StaticSchemaContext(
            db_id=db_id,
            db_path=db_path,
            columns=tuple(columns),
            tables=table_columns,
            table_edges=tuple(sorted(table_edges)),
            primary_key_count=primary_key_count,
            foreign_key_count=len(foreign_key_constraints),
        )


def _readonly_uri(db_path: str | Path) -> str:
    return f"{Path(db_path).resolve().as_uri()}?mode=ro"


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _load_descriptions(db_directory: Path) -> dict[str, dict[str, str]]:
    description_directory = db_directory / "database_description"
    if not description_directory.is_dir() or not any(
        description_directory.glob("*.csv")
    ):
        return {}
    loaded = load_tables_concatenated_description(
        str(db_directory), use_value_description=True
    )
    return {
        str(table).casefold(): {
            str(column).casefold(): str(description or "")
            for column, description in columns.items()
        }
        for table, columns in loaded.items()
    }


def compute_graph_stats(
    table_edges: Sequence[tuple[str, str]],
    selected_tables: Sequence[str],
) -> SchemaGraphStats:
    selected_by_key: dict[str, str] = {}
    for table in selected_tables:
        selected_by_key.setdefault(str(table).casefold(), str(table))
    selected = set(selected_by_key)

    graph: dict[str, set[str]] = {}
    normalized_edges: set[tuple[str, str]] = set()
    for left, right in table_edges:
        left_key = str(left).casefold()
        right_key = str(right).casefold()
        if left_key == right_key:
            continue
        edge = tuple(sorted((left_key, right_key)))
        if edge in normalized_edges:
            continue
        normalized_edges.add(edge)
        graph.setdefault(left_key, set()).add(right_key)
        graph.setdefault(right_key, set()).add(left_key)

    induced_edges = {
        edge
        for edge in normalized_edges
        if edge[0] in selected and edge[1] in selected
    }
    induced_graph = {table: set() for table in selected}
    for left, right in induced_edges:
        induced_graph[left].add(right)
        induced_graph[right].add(left)

    remaining = set(selected)
    component_count = 0
    while remaining:
        component_count += 1
        queue = [remaining.pop()]
        while queue:
            current = queue.pop()
            unseen = induced_graph[current] & remaining
            remaining.difference_update(unseen)
            queue.extend(unseen)

    table_count = len(selected)
    isolated_count = sum(not neighbors for neighbors in induced_graph.values())
    connected = table_count <= 1 or component_count == 1
    density = (
        2.0 * len(induced_edges) / (table_count * (table_count - 1))
        if table_count > 1
        else 0.0
    )
    min_connection_edges = _maximum_pairwise_distance(graph, selected)

    return SchemaGraphStats(
        table_count=table_count,
        edge_count=len(induced_edges),
        component_count=component_count,
        isolated_table_count=isolated_count,
        connected=connected,
        min_connection_edges=min_connection_edges,
        density=density,
    )


def _maximum_pairwise_distance(
    graph: Mapping[str, set[str]], selected: set[str]
) -> int:
    if len(selected) <= 1:
        return 0
    maximum = 0
    for source in selected:
        distances = {source: 0}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        for target in selected:
            if target not in distances:
                return 10
            maximum = max(maximum, distances[target])
    return maximum


def _validate_stage_idx(stage_idx: int) -> None:
    if stage_idx not in _STAGE_INDICES:
        raise ValueError(f"stage_idx must be in 0..5, got {stage_idx!r}")


def _full_schema(static: StaticSchemaContext) -> dict[str, list[str]]:
    return {
        table: list(columns) for table, columns in static.tables.items()
    }


def _filtered_schema(
    raw_schema: Any, static: StaticSchemaContext
) -> dict[str, list[str]]:
    if not isinstance(raw_schema, Mapping):
        return {}
    table_lookup = {table.casefold(): table for table in static.tables}
    filtered: dict[str, list[str]] = {}
    for raw_table, raw_columns in raw_schema.items():
        table = table_lookup.get(str(raw_table).casefold())
        if table is None:
            continue
        if isinstance(raw_columns, str):
            candidates = [raw_columns]
        elif isinstance(raw_columns, Sequence):
            candidates = raw_columns
        else:
            continue
        column_lookup = {
            column.casefold(): column for column in static.tables[table]
        }
        columns: list[str] = []
        for raw_column in candidates:
            column = column_lookup.get(str(raw_column).casefold())
            if column is not None and column not in columns:
                columns.append(column)
        filtered[table] = columns
    return filtered


def _schema_for_selected_tables(
    selected_tables: Any, static: StaticSchemaContext
) -> dict[str, list[str]]:
    if isinstance(selected_tables, str) or not isinstance(
        selected_tables, Sequence
    ):
        return {}
    table_lookup = {table.casefold(): table for table in static.tables}
    selected: dict[str, list[str]] = {}
    for raw_table in selected_tables:
        table = table_lookup.get(str(raw_table).casefold())
        if table is not None and table not in selected:
            selected[table] = list(static.tables[table])
    return selected


def active_schema_for_stage(
    payload: Any,
    stage_idx: int,
    static: StaticSchemaContext,
) -> dict[str, list[str]]:
    """Return only schema information available before a router decision."""
    _validate_stage_idx(stage_idx)
    if stage_idx in (0, 1):
        return _full_schema(static)

    tentative = _filtered_schema(
        payload_get(payload, "tentative_schema", {}), static
    )
    if tentative:
        return tentative
    selected = _schema_for_selected_tables(
        payload_get(payload, "selected_tables", ()), static
    )
    return selected or _full_schema(static)


def _static_descriptions(
    static: StaticSchemaContext,
) -> dict[str, dict[str, str]]:
    descriptions = {
        table: {column: "" for column in columns}
        for table, columns in static.tables.items()
    }
    for column in static.columns:
        descriptions[column.table][column.name] = column.description
    return descriptions


def _description_text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    fields = (
        "column_name",
        "column_description",
        "value_description",
        "data_format",
        "description",
    )
    parts: list[str] = []
    for field in fields:
        part = payload_get(value, field, "")
        if isinstance(part, str):
            normalized = " ".join(part.split())
            if normalized and normalized not in parts:
                parts.append(normalized)
    return ", ".join(parts)


def _merge_description(static_text: str, dynamic_text: str) -> str:
    if not dynamic_text or dynamic_text in static_text:
        return static_text
    if not static_text:
        return dynamic_text
    return f"{static_text}, {dynamic_text}"


def visible_descriptions_for_stage(
    payload: Any,
    stage_idx: int,
    static: StaticSchemaContext,
) -> dict[str, dict[str, str]]:
    """Return static descriptions plus currently visible S3 descriptions."""
    _validate_stage_idx(stage_idx)
    visible = _static_descriptions(static)
    if stage_idx == 0:
        return visible

    dynamic = payload_get(payload, "schema_with_descriptions", {})
    if not isinstance(dynamic, Mapping):
        return visible
    table_lookup = {table.casefold(): table for table in static.tables}
    for raw_table, raw_columns in dynamic.items():
        table = table_lookup.get(str(raw_table).casefold())
        if table is None or not isinstance(raw_columns, Mapping):
            continue
        column_lookup = {
            column.casefold(): column for column in static.tables[table]
        }
        for raw_column, value in raw_columns.items():
            column = column_lookup.get(str(raw_column).casefold())
            if column is None:
                continue
            visible[table][column] = _merge_description(
                visible[table][column], _description_text(value)
            )
    return visible


def candidate_sql_for_stage(payload: Any, stage_idx: int) -> str:
    """Expose the S7 candidate only to the S8 pre-decision state."""
    if stage_idx != 5:
        return ""
    candidates = payload_get(payload, "candidates", ())
    if isinstance(candidates, Sequence) and not isinstance(candidates, str):
        if candidates:
            candidate_sql = payload_get(candidates[0], "sql", "")
            if isinstance(candidate_sql, str) and candidate_sql.strip():
                return candidate_sql
    final_sql = payload_get(payload, "final_sql", "")
    return final_sql if isinstance(final_sql, str) and final_sql.strip() else ""


def sqlite_explain_valid(db_path: Path, sql: str) -> bool:
    """Validate a SELECT through read-only SQLite query planning only."""
    if not isinstance(sql, str) or not sql.strip():
        return False
    try:
        parsed = parse_one(sql, read="sqlite")
        if parsed is None or not any(parsed.find_all(exp.Select)):
            return False
        with sqlite3.connect(_readonly_uri(db_path), uri=True) as connection:
            connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        return True
    except (ParseError, sqlite3.Error):
        return False


def _source_for_qualifier(scope: Scope, qualifier: str) -> Any:
    current: Scope | None = scope
    key = qualifier.casefold()
    while current is not None:
        for alias, source in current.sources.items():
            if str(alias).casefold() == key:
                return source
        current = current.parent
    return None


def _physical_sources(scope: Scope) -> list[exp.Table]:
    unique: dict[int, exp.Table] = {}
    for source in scope.sources.values():
        if isinstance(source, exp.Table):
            unique[id(source)] = source
    return list(unique.values())


def _empty_sql_quality(parseable: bool = False) -> SqlQuality:
    return SqlQuality(
        parseable=parseable,
        unknown_table_count=0,
        unknown_column_count=0,
        reference_valid_ratio=0.0,
        referenced_tables_connected=True,
        selected_schema_coverage=0.0,
        sqlite_explain_valid=False,
    )


def analyze_sql_quality(
    sql: str,
    static: StaticSchemaContext,
    selected_schema: Mapping[str, Sequence[str]] | None = None,
) -> SqlQuality:
    """Check SQL identifiers against static schema without outcomes or execution."""
    if not isinstance(sql, str) or not sql.strip():
        return _empty_sql_quality()
    try:
        parsed = parse_one(sql, read="sqlite")
        scopes = list(traverse_scope(parsed))
    except (SqlglotError, ValueError, RecursionError):
        return _empty_sql_quality()

    table_lookup = {table.casefold(): table for table in static.tables}
    column_lookups = {
        table: {column.casefold(): column for column in columns}
        for table, columns in static.tables.items()
    }
    referenced_tables: set[str] = set()
    referenced_columns: set[tuple[str, str]] = set()
    unknown_tables: set[str] = set()
    unknown_columns: set[str] = set()
    valid_references = 0
    total_references = 0

    for scope in scopes:
        for source in _physical_sources(scope):
            total_references += 1
            real_table = table_lookup.get(source.name.casefold())
            if real_table is None:
                unknown_tables.add(source.name)
            else:
                valid_references += 1
                referenced_tables.add(real_table)

        projection_aliases = {
            expression.alias.casefold()
            for expression in getattr(scope.expression, "selects", ())
            if expression.alias
        }
        physical_sources = _physical_sources(scope)
        has_derived_source = any(
            isinstance(source, Scope) for source in scope.sources.values()
        )
        for column in scope.columns:
            if column.is_star:
                continue
            column_name = column.name
            qualifier = column.table
            if qualifier:
                source = _source_for_qualifier(scope, qualifier)
                if isinstance(source, Scope):
                    continue
                total_references += 1
                if not isinstance(source, exp.Table):
                    unknown_columns.add(f"{qualifier}.{column_name}")
                    continue
                real_table = table_lookup.get(source.name.casefold())
                if real_table is None:
                    unknown_columns.add(f"{source.name}.{column_name}")
                    continue
                real_column = column_lookups[real_table].get(
                    column_name.casefold()
                )
                if real_column is None:
                    unknown_columns.add(f"{real_table}.{column_name}")
                    continue
                valid_references += 1
                referenced_columns.add((real_table, real_column))
                continue

            if column_name.casefold() in projection_aliases:
                continue
            matching_columns: list[tuple[str, str]] = []
            for source in physical_sources:
                real_table = table_lookup.get(source.name.casefold())
                if real_table is None:
                    continue
                real_column = column_lookups[real_table].get(
                    column_name.casefold()
                )
                if real_column is not None:
                    matching_columns.append((real_table, real_column))
            if matching_columns:
                total_references += 1
                valid_references += 1
                if len(matching_columns) == 1:
                    referenced_columns.add(matching_columns[0])
            elif not has_derived_source:
                total_references += 1
                unknown_columns.add(column_name)

    reference_valid_ratio = (
        valid_references / total_references if total_references else 0.0
    )
    connected = not unknown_tables and static.graph_stats(
        tuple(referenced_tables)
    ).connected
    selected_columns: set[tuple[str, str]] = set()
    if selected_schema is not None:
        filtered_selected = _filtered_schema(selected_schema, static)
        selected_columns = {
            (table, column)
            for table, columns in filtered_selected.items()
            for column in columns
        }
    selected_coverage = len(referenced_columns & selected_columns) / max(
        len(selected_columns), 1
    )

    return SqlQuality(
        parseable=True,
        unknown_table_count=len(unknown_tables),
        unknown_column_count=len(unknown_columns),
        reference_valid_ratio=reference_valid_ratio,
        referenced_tables_connected=connected,
        selected_schema_coverage=selected_coverage,
        sqlite_explain_valid=sqlite_explain_valid(static.db_path, sql),
        referenced_tables=tuple(sorted(referenced_tables)),
        referenced_columns=tuple(sorted(referenced_columns)),
        unknown_tables=tuple(sorted(unknown_tables)),
        unknown_columns=tuple(sorted(unknown_columns)),
        valid_reference_count=valid_references,
        total_reference_count=total_references,
    )


compute_sql_quality = analyze_sql_quality
