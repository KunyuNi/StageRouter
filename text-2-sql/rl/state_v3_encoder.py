"""Deployable State V3 encoder with strict pre-decision visibility."""

from __future__ import annotations

import math
import re
import hashlib
import json
from collections import OrderedDict, namedtuple
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .base_state_encoder import BaseStateEncoder
from .semantic_config import DEFAULT_SEMANTIC_MODEL
from .state_schema import ARTIFACT_FEATURE_DIM, S7_STAGE_IDX
from .state_v3_context import (
    StaticSchemaContext,
    StaticSchemaContextProvider,
    active_schema_for_stage,
    analyze_sql_quality,
    candidate_sql_for_stage,
    payload_get,
    visible_descriptions_for_stage,
)
from .state_v3_schema import (
    ARTIFACT_V3_DIM,
    QUERY_DIM,
    QUESTION_SEMANTIC_DIM,
    ROUTE_COST_DIM,
    SCHEMA_CONTEXT_DIM,
    SCHEMA_SEMANTIC_DIM,
    STATE_V3_DIM,
    STATE_V3_VERSION,
)
from .state_v3_semantic import (
    A3ReadOnlyQuestionEmbeddingProvider,
    SchemaSemanticCache,
    StateV3TextEncoder,
)


_KEYWORD_TYPES = (
    "entity",
    "predicate",
    "aggregation",
    "condition_value",
    "ordering",
)
_SCORE_KEYS = ("embedding_similarity", "similarity", "score")
SqlQualityCacheKey = namedtuple(
    "SqlQualityCacheKey",
    (
        "db_id",
        "db_path",
        "db_file_sha256",
        "sql_sha256",
        "active_schema_sha256",
    ),
)


class StateV3Encoder:
    """Compose the six State V3 towers without reading outcomes or future stages."""

    state_version = STATE_V3_VERSION
    state_dim = STATE_V3_DIM

    def __init__(
        self,
        db_root: str | Path | None = None,
        *,
        db_mode: str = "train",
        plm_model_name: str | Path = DEFAULT_SEMANTIC_MODEL,
        device: str = "cpu",
        stage_cost_clips: Sequence[float] | None = None,
        semantic_cache_root: str | Path = "cache/state_v3_schema_embeddings",
        static_schema_provider: Any | None = None,
        schema_semantic_cache: Any | None = None,
        base_encoder: BaseStateEncoder | None = None,
        sql_quality_cache_size: int = 128,
        alignment_cache_size: int = 128,
    ) -> None:
        if isinstance(sql_quality_cache_size, bool) or int(sql_quality_cache_size) <= 0:
            raise ValueError("sql_quality_cache_size must be positive")
        if isinstance(alignment_cache_size, bool) or int(alignment_cache_size) <= 0:
            raise ValueError("alignment_cache_size must be positive")
        self.db_mode = str(db_mode)
        self.base_encoder = base_encoder or BaseStateEncoder(
            plm_model_name=str(plm_model_name),
            device=device,
            stage_cost_clips=stage_cost_clips,
        )

        if static_schema_provider is None:
            if db_root is None:
                raise ValueError(
                    "db_root is required when static_schema_provider is not injected"
                )
            static_schema_provider = StaticSchemaContextProvider(db_root, self.db_mode)
        if not callable(getattr(static_schema_provider, "get", None)):
            raise TypeError("static_schema_provider must provide get(db_id)")

        if schema_semantic_cache is None:
            text_encoder = StateV3TextEncoder(
                plm_model_name,
                device=device,
            )
            schema_semantic_cache = SchemaSemanticCache(
                encoder=text_encoder,
                db_mode=self.db_mode,
                cache_root=semantic_cache_root,
                alignment_lru_size=int(alignment_cache_size),
            )
        if not callable(getattr(schema_semantic_cache, "align", None)):
            raise TypeError("schema_semantic_cache must provide align(...)")
        if not callable(
            getattr(schema_semantic_cache, "encode_candidate_sql", None)
        ):
            raise TypeError(
                "schema_semantic_cache must provide encode_candidate_sql(...)"
            )
        text_encoder = getattr(schema_semantic_cache, "encoder", None)
        if not callable(getattr(text_encoder, "encode_texts", None)):
            raise TypeError(
                "schema_semantic_cache.encoder must provide encode_texts(texts)"
            )

        self.static_schema_provider = static_schema_provider
        self.schema_semantic_cache = schema_semantic_cache
        self._sql_quality_cache_size = int(sql_quality_cache_size)
        self._sql_quality_cache: OrderedDict[SqlQualityCacheKey, Any] = OrderedDict()
        self._sql_quality_source_by_db: dict[tuple[str, str], str] = {}
        self._db_file_hash_cache: dict[
            Path,
            tuple[tuple[int, int, int, int], str],
        ] = {}
        self._a3_question_embeddings: Mapping[
            tuple[str, str, int], np.ndarray
        ] | None = None
        self._a3_question_text_keys: Mapping[str, tuple[str, str, int]] | None = None
        self._a3_question_embedding_identity: Mapping[str, Any] | None = None

    @property
    def sql_quality_cache_keys(self) -> tuple[SqlQualityCacheKey, ...]:
        return tuple(self._sql_quality_cache)

    def encode_question(self, question: str, evidence: str = "") -> np.ndarray:
        """Encode exactly question and optional evidence with the shared local model."""
        question = str(question or "")
        evidence = str(evidence or "")
        text = f"{question} [SEP] {evidence}" if evidence else question
        if self._a3_question_embeddings is not None:
            assert self._a3_question_text_keys is not None
            key = self._a3_question_text_keys.get(text)
            if key is None:
                raise RuntimeError(
                    "read-only A3 question embedding cache miss"
                )
            return self._a3_question_embeddings[key].copy()
        matrix = np.asarray(
            self.schema_semantic_cache.encoder.encode_texts([text]),
            dtype=np.float32,
        )
        if matrix.shape != (1, QUESTION_SEMANTIC_DIM):
            raise ValueError(
                "question encoder output must have shape "
                f"(1, {QUESTION_SEMANTIC_DIM})"
            )
        vector = matrix[0]
        if not np.isfinite(vector).all():
            raise ValueError("question encoder output contains NaN or inf")
        return vector.copy()

    @property
    def a3_question_embedding_identity(self) -> Mapping[str, Any] | None:
        return self._a3_question_embedding_identity

    def precompute_a3_question_embeddings(
        self,
        question_map: Mapping[int, Mapping[str, Any]],
        *,
        model_fingerprint: str,
        question_data_sha256: str,
    ) -> Mapping[str, Any]:
        """Install one immutable, identity-bound embedding per formal A3 qid."""
        if self._a3_question_embeddings is not None:
            raise ValueError("A3 question embeddings are already installed")
        model_identity = str(model_fingerprint).strip()
        data_identity = str(question_data_sha256).strip().lower()
        if not model_identity or not re.fullmatch(r"[0-9a-f]{64}", data_identity):
            raise ValueError("A3 question embedding identity is invalid")
        normalized: list[tuple[int, str, str]] = []
        for raw_qid in sorted(question_map):
            qid = int(raw_qid)
            record = question_map[raw_qid]
            if not isinstance(record, Mapping):
                raise ValueError("A3 question map entries must be mappings")
            question = str(record.get("question", "") or "")
            evidence = str(record.get("evidence", "") or "")
            text = f"{question} [SEP] {evidence}" if evidence else question
            normalized.append((qid, question, text))
        if not normalized or len({qid for qid, _, _ in normalized}) != len(
            normalized
        ):
            raise ValueError("A3 question map must contain unique qids")
        matrix = np.asarray(
            self.schema_semantic_cache.encoder.encode_texts(
                [text for _, _, text in normalized]
                + [question for _, question, _ in normalized]
            ),
            dtype=np.float32,
        )
        if matrix.shape != (len(normalized) * 2, QUESTION_SEMANTIC_DIM):
            raise ValueError("A3 question embedding matrix shape is invalid")
        if not np.isfinite(matrix).all():
            raise ValueError("A3 question embeddings contain NaN or inf")
        embeddings: dict[tuple[str, str, int], np.ndarray] = {}
        text_keys: dict[str, tuple[str, str, int]] = {}
        question_evidence_digest = hashlib.sha256()
        provider_records: list[tuple[int, str, np.ndarray]] = []
        for index, (qid, question, text) in enumerate(normalized):
            key = (model_identity, data_identity, qid)
            vector = np.ascontiguousarray(matrix[index], dtype=np.float32)
            vector.setflags(write=False)
            embeddings[key] = vector
            previous = text_keys.setdefault(text, key)
            if not np.array_equal(embeddings[previous], vector):
                raise ValueError(
                    "duplicate A3 question text produced inconsistent embeddings"
                )
            question_evidence_digest.update(str(qid).encode("ascii"))
            question_evidence_digest.update(b"\0")
            question_evidence_digest.update(
                hashlib.sha256(text.encode("utf-8")).digest()
            )
            question_evidence_digest.update(vector.tobytes(order="C"))
            provider_records.append(
                (qid, question, matrix[len(normalized) + index])
            )
        provider = A3ReadOnlyQuestionEmbeddingProvider(
            provider_records,
            model_fingerprint=model_identity,
            question_data_sha256=data_identity,
        )
        install_provider = getattr(
            self.schema_semantic_cache,
            "install_a3_question_embedding_provider",
            None,
        )
        if not callable(install_provider):
            raise TypeError(
                "schema_semantic_cache must support A3 question providers"
            )
        install_provider(
            provider,
            model_fingerprint=model_identity,
            question_data_sha256=data_identity,
        )
        identity = {
            "format_version": 2,
            "model_fingerprint": model_identity,
            "question_data_sha256": data_identity,
            "qid_count": len(normalized),
            "question_evidence_embedding_sha256": (
                question_evidence_digest.hexdigest()
            ),
            "question_embedding_sha256": provider.embedding_sha256,
        }
        self._a3_question_embeddings = MappingProxyType(embeddings)
        self._a3_question_text_keys = MappingProxyType(text_keys)
        self._a3_question_embedding_identity = MappingProxyType(identity)
        return self._a3_question_embedding_identity

    def encode(
        self,
        payload: Any,
        stage_idx: int,
        budget_remaining: float,
        tier_history: Sequence[int],
        cost_history: Sequence[float],
        question_emb: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        stage_idx = _stage_index(stage_idx)
        db_id = payload_get(payload, "db_id", None)
        if db_id is None or not str(db_id).strip():
            raise ValueError("payload.db_id is required for State V3 encoding")
        static = self.static_schema_provider.get(str(db_id))
        if not isinstance(static, StaticSchemaContext):
            required = ("tables", "columns", "graph_stats", "db_path")
            if any(not hasattr(static, name) for name in required):
                raise TypeError("schema provider returned an invalid static context")

        question = str(payload_get(payload, "question", "") or "")
        evidence = str(payload_get(payload, "evidence", "") or "")
        full_schema = {
            table: list(columns) for table, columns in static.tables.items()
        }
        active_schema = active_schema_for_stage(payload, stage_idx, static)
        visible_descriptions = (
            None
            if stage_idx == 0
            else visible_descriptions_for_stage(payload, stage_idx, static)
        )
        alignment = self.schema_semantic_cache.align(
            question=question,
            active_schema=active_schema,
            static_context=static,
            visible_descriptions=visible_descriptions,
        )

        sanitized_query = {
            "question": question,
            "evidence": evidence,
            "schema_with_descriptions": full_schema,
            "foreign_keys": [None] * int(static.foreign_key_count),
        }
        query = self.base_encoder.encode_query_features(sanitized_query)
        route = self._encode_route(
            stage_idx=stage_idx,
            budget_remaining=budget_remaining,
            tier_history=tier_history,
            cost_history=cost_history,
            action_mask=action_mask,
        )
        schema_context = _encode_schema_context(static, active_schema, alignment)
        question_semantic = _embedding(question_emb, "question semantic")
        artifact = self._encode_artifact(
            payload=payload,
            stage_idx=stage_idx,
            static=static,
            active_schema=active_schema,
            question_semantic=question_semantic,
        )
        schema_semantic = _embedding(
            alignment.schema_embedding, "schema semantic"
        )

        _expect_shape(query, QUERY_DIM, "query")
        _expect_shape(schema_context, SCHEMA_CONTEXT_DIM, "schema context")
        _expect_shape(route, ROUTE_COST_DIM, "route cost")
        _expect_shape(artifact, ARTIFACT_V3_DIM, "artifact")
        state = np.concatenate(
            [
                query,
                schema_context,
                route,
                artifact,
                question_semantic,
                schema_semantic,
            ]
        ).astype(np.float32, copy=False)
        if state.shape != (STATE_V3_DIM,):
            raise ValueError(
                f"encoded State V3 shape {state.shape} != ({STATE_V3_DIM},)"
            )
        if not np.isfinite(state).all():
            raise ValueError("encoded State V3 contains NaN or inf")
        return state

    def _encode_route(
        self,
        *,
        stage_idx: int,
        budget_remaining: float,
        tier_history: Sequence[int],
        cost_history: Sequence[float],
        action_mask: np.ndarray | None,
    ) -> np.ndarray:
        visible_tiers = list(tier_history[:stage_idx])
        visible_costs = list(cost_history[:stage_idx])
        mask = None
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=np.float32)
            if mask.shape != (4,) or not np.isfinite(mask).all():
                raise ValueError("action_mask must contain four finite values")
            if np.any(mask < 0.0) or np.any(mask > 1.0):
                raise ValueError("action_mask values must be in [0,1]")
            if stage_idx == S7_STAGE_IDX and mask[3] != 0.0:
                raise ValueError("S7 action mask must forbid SKIP")
        return self.base_encoder.encode_route_cost_features(
            stage_idx=stage_idx,
            tier_history=visible_tiers,
            cost_history=visible_costs,
            action_mask=mask,
            budget_remaining=budget_remaining,
        )

    def _encode_artifact(
        self,
        *,
        payload: Any,
        stage_idx: int,
        static: StaticSchemaContext,
        active_schema: Mapping[str, Sequence[str]],
        question_semantic: np.ndarray,
    ) -> np.ndarray:
        full_table_count = len(static.tables)
        full_column_count = sum(len(columns) for columns in static.tables.values())
        selected_schema, has_tentative = _visible_selected_schema(
            payload, stage_idx, static
        )
        selected_tables = tuple(selected_schema)
        selected_column_count = sum(
            len(columns) for columns in selected_schema.values()
        )
        graph_tables = tuple(active_schema)
        graph = static.graph_stats(graph_tables)

        if stage_idx >= 1:
            keywords = _keyword_features(payload_get(payload, "keywords", ()))
            similar_value_count, similar_scores = _similar_value_features(
                payload_get(payload, "similar_values", {})
            )
            similar_column_count, similar_column_tables = _similar_column_features(
                payload_get(payload, "similar_columns", {}), static
            )
            context_count = _dynamic_context_count(
                payload_get(payload, "schema_with_descriptions", {}), static
            )
        else:
            keywords = (0, {name: 0 for name in _KEYWORD_TYPES}, ())
            similar_value_count, similar_scores = 0, ()
            similar_column_count, similar_column_tables = 0, 0
            context_count = 0

        entity_count = similar_value_count + similar_column_count
        keyword_count, keyword_type_counts, confidences = keywords
        keyword_denominator = max(keyword_count, 1)
        score_mean = float(np.mean(similar_scores)) if similar_scores else 0.0
        score_max = max(similar_scores, default=0.0)

        sql, sql_source = _visible_candidate_sql(payload, stage_idx)
        sql_quality = (
            self._get_sql_quality(sql, static, active_schema) if sql else None
        )
        sql_upper = sql.upper() if sql else ""
        if sql:
            candidate_embedding = self.schema_semantic_cache.encode_candidate_sql(
                static, sql
            )
            question_sql_similarity = _mapped_cosine(
                question_semantic, candidate_embedding
            )
        else:
            question_sql_similarity = 0.0

        base = np.asarray(
            [
                float(has_tentative),
                _clip(len(selected_tables) / 20.0),
                _clip(selected_column_count / 100.0),
                _clip(graph.edge_count / 50.0),
                _ratio(len(selected_tables), full_table_count),
                _ratio(selected_column_count, full_column_count),
                float(graph.connected),
                _clip(graph.min_connection_edges / 10.0),
                _ratio(graph.component_count, len(graph_tables)),
                float(not selected_schema),
                float(selected_column_count > 60),
                float(entity_count > 0),
                _clip(entity_count / 100.0),
                float(context_count > 0),
                _clip(context_count / 100.0),
                float(bool(payload_get(payload, "question", None))),
                float(bool(payload_get(payload, "evidence", None))),
                float(bool(static.tables)),
                float(has_tentative),
                float(sql_source == "candidate"),
                float(sql_source == "final"),
                0.0,
                0.0,
                float(bool(sql)),
                _clip(len(sql.split()) / 200.0) if sql else 0.0,
                _sql_has(sql_upper, "WHERE"),
                _sql_has(sql_upper, "JOIN"),
                _sql_has(sql_upper, "GROUP BY"),
                _sql_has(sql_upper, "ORDER BY"),
                _sql_has(sql_upper, "LIMIT"),
                float(
                    bool(sql)
                    and any(
                        function in sql_upper
                        for function in ("COUNT(", "SUM(", "AVG(", "MAX(", "MIN(")
                    )
                ),
                _clip(len(re.findall(r"\b(?:FROM|JOIN)\b", sql_upper)) / 20.0)
                if sql
                else 0.0,
                float(sql_quality.parseable) if sql_quality is not None else 0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        _expect_shape(base, ARTIFACT_FEATURE_DIM, "base artifact")

        tentative_table_reduction = (
            _clip(1.0 - len(selected_tables) / max(full_table_count, 1))
            if has_tentative
            else 0.0
        )
        tentative_column_reduction = (
            _clip(1.0 - selected_column_count / max(full_column_count, 1))
            if has_tentative
            else 0.0
        )
        extra = np.asarray(
            [
                _clip(keyword_count / 20.0),
                _ratio(keyword_type_counts["entity"], keyword_denominator),
                _ratio(keyword_type_counts["predicate"], keyword_denominator),
                _ratio(keyword_type_counts["aggregation"], keyword_denominator),
                _ratio(
                    keyword_type_counts["condition_value"], keyword_denominator
                ),
                _ratio(keyword_type_counts["ordering"], keyword_denominator),
                _clip(float(np.mean(confidences))) if confidences else 0.0,
                _clip(similar_value_count / 100.0),
                _clip(score_mean),
                _clip(score_max),
                _clip(similar_column_count / 100.0),
                _ratio(similar_column_tables, full_table_count),
                tentative_table_reduction,
                tentative_column_reduction,
                _clip(sql_quality.unknown_table_count / 10.0)
                if sql_quality is not None
                else 0.0,
                _clip(sql_quality.unknown_column_count / 50.0)
                if sql_quality is not None
                else 0.0,
                _clip(sql_quality.reference_valid_ratio)
                if sql_quality is not None
                else 0.0,
                float(sql_quality.referenced_tables_connected)
                if sql_quality is not None
                else 0.0,
                _clip(sql_quality.selected_schema_coverage)
                if sql_quality is not None
                else 0.0,
                question_sql_similarity,
                float(sql_quality.sqlite_explain_valid)
                if sql_quality is not None
                else 0.0,
            ],
            dtype=np.float32,
        )
        artifact = np.concatenate([base, extra]).astype(np.float32, copy=False)
        _expect_shape(artifact, ARTIFACT_V3_DIM, "artifact")
        if np.any(artifact < 0.0) or np.any(artifact > 1.0):
            raise ValueError("artifact values must be in [0,1]")
        return artifact

    def _get_sql_quality(
        self,
        sql: str,
        static: StaticSchemaContext,
        active_schema: Mapping[str, Sequence[str]],
    ):
        db_path = str(Path(static.db_path).resolve())
        db_identity = (str(static.db_id), db_path)
        db_file_sha256 = self._database_file_sha256(static.db_path)
        previous_source = self._sql_quality_source_by_db.get(db_identity)
        if previous_source is not None and previous_source != db_file_sha256:
            for cached_key in tuple(self._sql_quality_cache):
                if (
                    cached_key.db_id == db_identity[0]
                    and cached_key.db_path == db_identity[1]
                ):
                    del self._sql_quality_cache[cached_key]
        self._sql_quality_source_by_db[db_identity] = db_file_sha256

        key = SqlQualityCacheKey(
            db_id=db_identity[0],
            db_path=db_identity[1],
            db_file_sha256=db_file_sha256,
            sql_sha256=_sha256_text(sql),
            active_schema_sha256=_active_schema_sha256(active_schema),
        )
        try:
            quality = self._sql_quality_cache.pop(key)
        except KeyError:
            quality = analyze_sql_quality(sql, static, active_schema)
            self._sql_quality_cache[key] = quality
            while len(self._sql_quality_cache) > self._sql_quality_cache_size:
                self._sql_quality_cache.popitem(last=False)
            return quality
        self._sql_quality_cache[key] = quality
        return quality

    def _database_file_sha256(self, db_path: str | Path) -> str:
        path = Path(db_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        stat = path.stat()
        snapshot = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            int(getattr(stat, "st_ino", 0)),
        )
        cached = self._db_file_hash_cache.get(path)
        if cached is not None and cached[0] == snapshot:
            return cached[1]
        fingerprint = _file_sha256(path)
        self._db_file_hash_cache[path] = (snapshot, fingerprint)
        return fingerprint


def _encode_schema_context(
    static: StaticSchemaContext,
    active_schema: Mapping[str, Sequence[str]],
    alignment: Any,
) -> np.ndarray:
    table_count = len(static.tables)
    column_count = sum(len(columns) for columns in static.tables.values())
    widths = [len(columns) for columns in static.tables.values()]
    graph = static.graph_stats(tuple(static.tables))
    active_item_count = len(active_schema) + sum(
        len(columns) for columns in active_schema.values()
    )
    context = np.asarray(
        [
            _log_norm(table_count, 20),
            _log_norm(column_count, 200),
            _log_norm(static.primary_key_count, 100),
            _log_norm(static.foreign_key_count, 100),
            _clip(graph.density),
            _ratio(graph.component_count, table_count),
            _ratio(graph.isolated_table_count, table_count),
            _ratio(sum(width >= 20 for width in widths), table_count),
            _clip((column_count / max(table_count, 1)) / 30.0),
            _clip(max(widths, default=0) / 80.0),
            _log_norm(active_item_count, 220),
            _clip(alignment.table_top1),
            _clip(alignment.table_top3_mean),
            _clip(alignment.table_margin),
            _clip(alignment.table_entropy),
            _clip(alignment.column_top1),
            _clip(alignment.column_top3_mean),
            _clip(alignment.column_margin),
            _clip(alignment.column_entropy),
            _clip(alignment.token_coverage),
            _clip(alignment.active_table_ratio),
            _clip(alignment.active_column_ratio),
            _clip((float(alignment.active_alignment_delta_from_full) + 1.0) / 2.0),
            float(bool(alignment.available)),
        ],
        dtype=np.float32,
    )
    _expect_shape(context, SCHEMA_CONTEXT_DIM, "schema context")
    if np.any(context < 0.0) or np.any(context > 1.0):
        raise ValueError("schema context values must be in [0,1]")
    return context


def _visible_selected_schema(
    payload: Any,
    stage_idx: int,
    static: StaticSchemaContext,
) -> tuple[dict[str, list[str]], bool]:
    if stage_idx < 2:
        return {}, False
    tentative = _filter_schema(payload_get(payload, "tentative_schema", {}), static)
    if tentative:
        return tentative, True
    selected = payload_get(payload, "selected_tables", ())
    if isinstance(selected, str) or not isinstance(selected, Sequence):
        return {}, False
    table_lookup = {table.casefold(): table for table in static.tables}
    selected_schema: dict[str, list[str]] = {}
    for raw_table in selected:
        table = table_lookup.get(str(raw_table).casefold())
        if table is not None and table not in selected_schema:
            selected_schema[table] = list(static.tables[table])
    return selected_schema, False


def _filter_schema(
    raw_schema: Any, static: StaticSchemaContext
) -> dict[str, list[str]]:
    if not isinstance(raw_schema, Mapping):
        return {}
    table_lookup = {table.casefold(): table for table in static.tables}
    output: dict[str, list[str]] = {}
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
        output[table] = columns
    return output


def _keyword_features(raw_keywords: Any):
    if isinstance(raw_keywords, Mapping):
        if "value" in raw_keywords or "type" in raw_keywords:
            items = [raw_keywords]
        else:
            items = list(raw_keywords.values())
    elif isinstance(raw_keywords, Sequence) and not isinstance(
        raw_keywords, (str, bytes)
    ):
        items = list(raw_keywords)
    elif raw_keywords is None:
        items = []
    else:
        items = [raw_keywords]

    counts = {name: 0 for name in _KEYWORD_TYPES}
    confidences: list[float] = []
    for item in items:
        keyword_type = payload_get(item, "type", None)
        normalized_type = (
            keyword_type.strip().casefold()
            if isinstance(keyword_type, str)
            else ""
        )
        if normalized_type not in counts:
            continue
        counts[normalized_type] += 1
        confidence = _valid_unit_number(payload_get(item, "confidence", None))
        if confidence is not None:
            confidences.append(confidence)
    return len(items), counts, tuple(confidences)


def _similar_value_features(raw: Any) -> tuple[int, tuple[float, ...]]:
    items = tuple(_iter_similar_value_items(raw))
    scores = tuple(
        score
        for item in items
        if (score := _similar_value_score(item)) is not None
    )
    return len(items), scores


def _iter_similar_value_items(value: Any):
    if isinstance(value, Mapping):
        if any(key in value for key in _SCORE_KEYS) or any(
            key in value for key in ("value", "column_value", "text")
        ):
            yield value
            return
        for child in value.values():
            yield from _iter_similar_value_items(child)
        return
    if isinstance(value, tuple):
        yield value
        return
    if isinstance(value, list):
        if _is_flat_list_candidate(value):
            yield value
            return
        for child in value:
            yield from _iter_similar_value_items(child)
        return
    if isinstance(value, set):
        for child in value:
            yield from _iter_similar_value_items(child)
        return
    if value is not None and not isinstance(value, (int, float, complex, bool)):
        yield value


def _is_flat_list_candidate(value: list[Any]) -> bool:
    if len(value) < 2 or any(
        isinstance(item, (Mapping, list, tuple, set)) for item in value
    ):
        return False
    if _is_numeric_scalar(value[0]):
        return False
    return any(_is_numeric_scalar(item) for item in value[1:])


def _is_numeric_scalar(value: Any) -> bool:
    if isinstance(value, (str, bytes, bool, np.bool_)):
        return False
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _similar_value_score(item: Any) -> float | None:
    if isinstance(item, Mapping):
        for key in _SCORE_KEYS:
            score = _valid_unit_number(item.get(key))
            if score is not None:
                return score
        return None
    if not isinstance(item, (str, bytes, list, tuple)):
        for key in _SCORE_KEYS:
            score = _valid_unit_number(getattr(item, key, None))
            if score is not None:
                return score
        return None
    if isinstance(item, (list, tuple)):
        for value in reversed(item):
            score = _valid_unit_number(value)
            if score is not None:
                return score
    return None


def _similar_column_features(
    raw: Any, static: StaticSchemaContext
) -> tuple[int, int]:
    if not isinstance(raw, Mapping):
        return _nested_item_count(raw), 0
    real_tables = {table.casefold() for table in static.tables}
    count = 0
    covered: set[str] = set()
    for raw_table, columns in raw.items():
        item_count = _nested_item_count(columns)
        count += item_count
        table_key = str(raw_table).casefold()
        if item_count and table_key in real_tables:
            covered.add(table_key)
    return count, len(covered)


def _nested_item_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return sum(_nested_item_count(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 1


def _dynamic_context_count(raw: Any, static: StaticSchemaContext) -> int:
    if not isinstance(raw, Mapping):
        return 0
    table_lookup = {table.casefold(): table for table in static.tables}
    count = 0
    for raw_table, raw_columns in raw.items():
        table = table_lookup.get(str(raw_table).casefold())
        if table is None or not isinstance(raw_columns, Mapping):
            continue
        real_columns = {column.casefold() for column in static.tables[table]}
        count += sum(
            str(raw_column).casefold() in real_columns
            for raw_column in raw_columns
        )
    return count


def _visible_candidate_sql(payload: Any, stage_idx: int) -> tuple[str, str]:
    if stage_idx != 5:
        return "", ""
    candidates = payload_get(payload, "candidates", ())
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        if candidates:
            sql = payload_get(candidates[0], "sql", "")
            if isinstance(sql, str) and sql.strip():
                return sql, "candidate"
    if _has_revision_output(payload):
        return "", ""
    sql = candidate_sql_for_stage(payload, stage_idx)
    return (sql, "final") if sql else ("", "")


def _has_revision_output(payload: Any) -> bool:
    revision_count = payload_get(payload, "revision_count", 0)
    try:
        if int(revision_count or 0) > 0:
            return True
    except (TypeError, ValueError, OverflowError):
        return bool(revision_count)
    return bool(
        payload_get(payload, "revision_history", None)
        or payload_get(payload, "errors_fixed", None)
    )


def _mapped_cosine(left: Any, right: Any) -> float:
    left_vector = _embedding(left, "question semantic")
    right_vector = _embedding(right, "candidate SQL semantic")
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= 0.0:
        return 0.0
    cosine = float(np.dot(left_vector, right_vector) / denominator)
    return _clip((float(np.clip(cosine, -1.0, 1.0)) + 1.0) / 2.0)


def _active_schema_sha256(
    active_schema: Mapping[str, Sequence[str]],
) -> str:
    rows = [
        [
            str(table),
            sorted((str(column) for column in columns), key=str.casefold),
        ]
        for table, columns in active_schema.items()
    ]
    rows.sort(key=lambda row: row[0].casefold())
    return _sha256_text(
        json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _embedding(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (SCHEMA_SEMANTIC_DIM,):
        raise ValueError(
            f"{label} shape {vector.shape} != ({SCHEMA_SEMANTIC_DIM},)"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} contains NaN or inf")
    return vector


def _stage_index(stage_idx: Any) -> int:
    try:
        normalized = int(stage_idx)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"stage_idx must be 0..5, got {stage_idx!r}") from exc
    if normalized != stage_idx or not 0 <= normalized < 6:
        raise ValueError(f"stage_idx must be 0..5, got {stage_idx!r}")
    return normalized


def _valid_unit_number(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _sql_has(sql_upper: str, token: str) -> float:
    return float(bool(sql_upper) and f" {token} " in f" {sql_upper} ")


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return _clip(float(numerator) / float(denominator))


def _clip(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _log_norm(value: float, maximum: float) -> float:
    return _clip(math.log1p(max(float(value), 0.0)) / math.log1p(maximum))


def _expect_shape(array: Any, dimension: int, label: str) -> None:
    if np.asarray(array).shape != (dimension,):
        raise ValueError(
            f"{label} shape {np.asarray(array).shape} != ({dimension},)"
        )
