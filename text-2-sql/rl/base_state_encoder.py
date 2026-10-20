"""Shared tabular feature encoder for routing states.

The encoder returns a flat vector with schema-defined tower slices:
    [query(22) | route_cost(57) | artifact(41) | semantic(1024)]

It intentionally does not include gold SQL, gold answer, EX, or final reward.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .stage_mapping import legal_decision_actions
from .state_schema import (
    ACTION_DIM,
    ARTIFACT_FEATURE_DIM,
    ARTIFACT_SLICE,
    QUERY_FEATURE_DIM,
    QUERY_SLICE,
    ROUTE_COST_FEATURE_DIM,
    ROUTE_COST_SLICE,
    SEMANTIC_DIM,
    SEMANTIC_SLICE,
    S7_STAGE_IDX,
    STAGE_COUNT,
    STATE_DIM,
)
from .semantic_config import DEFAULT_SEMANTIC_MODEL

DIM_QUESTION_EMB = SEMANTIC_DIM


class BaseStateEncoder:
    """Pure NumPy feature encoder for replay-time state reconstruction."""

    def __init__(
        self,
        plm_model_name: str = DEFAULT_SEMANTIC_MODEL,
        device: str = "cpu",
        question_encoder: Any = None,
        stage_cost_clips: Sequence[float] | None = None,
    ):
        self.plm_model_name = plm_model_name
        self.device = device
        self.question_encoder = question_encoder
        self.stage_cost_clips = _validate_stage_cost_clips(stage_cost_clips)

    def encode_question(self, question: str, evidence: str = "") -> np.ndarray:
        if self.question_encoder is not None:
            emb = self.question_encoder.encode(question, evidence)
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            return self._encode_semantic_features(emb)
        return np.zeros(SEMANTIC_DIM, dtype=np.float32)

    def encode(
        self,
        payload: Any,
        stage_idx: int,
        budget_remaining: float,
        tier_history: List[int],
        cost_history: List[float],
        question_emb: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        query = self.encode_query_features(payload)
        route = self.encode_route_cost_features(
            stage_idx,
            tier_history,
            cost_history,
            action_mask,
            budget_remaining=budget_remaining,
        )
        artifact = self.encode_artifact_features(payload, stage_idx)
        semantic = self._encode_semantic_features(question_emb)

        state = np.concatenate([query, route, artifact, semantic]).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"encoded state shape {state.shape} != ({STATE_DIM},)")
        if not np.isfinite(state).all():
            raise ValueError("encoded state contains NaN or inf")
        return state

    def encode_query_features(self, payload: Any) -> np.ndarray:
        question = str(_get(payload, "question", "") or "")
        evidence = str(_get(payload, "evidence", "") or "")
        schema = _schema_dict(_get(payload, "schema_with_descriptions", {}) or {})
        table_count = len(schema)
        column_count = sum(len(cols) for cols in schema.values())
        fk_count = _foreign_key_count(payload)
        avg_cols = column_count / max(table_count, 1)
        max_cols = max((len(cols) for cols in schema.values()), default=0)
        density_den = max(table_count * max(table_count - 1, 1), 1)

        base_features = np.array(
            [
                _clip(len(question.split()) / 40.0),
                _clip(len(evidence.split()) / 80.0),
                1.0 if re.search(r"\d", question) else 0.0,
                1.0 if _has_date_or_time(question + " " + evidence) else 0.0,
                1.0 if _has_any(question, ("most", "least", "highest", "lowest", "first", "last", "maximum", "minimum", "latest", "earliest")) else 0.0,
                1.0 if _has_any(question, ("not", "no", "without", "except", "never", "exclude")) else 0.0,
                _log_norm(table_count, 20),
                _log_norm(column_count, 200),
                _log_norm(fk_count, 100),
                _clip(fk_count / density_den),
                _clip(avg_cols / 30.0),
                _clip(max_cols / 80.0),
                1.0 if fk_count > 0 else 0.0,
                1.0 if table_count >= 8 else 0.0,
            ],
            dtype=np.float32,
        )
        features = np.concatenate(
            [base_features, _query_complexity_features(question, evidence)]
        ).astype(np.float32)
        _assert_dim(features, QUERY_FEATURE_DIM, "query_features")
        return features

    def encode_route_cost_features(
        self,
        stage_idx: int,
        tier_history: List[int],
        cost_history: List[float],
        action_mask: np.ndarray | None = None,
        budget_remaining: float = 0.0,
    ) -> np.ndarray:
        stage_onehot = np.zeros(STAGE_COUNT, dtype=np.float32)
        if not 0 <= int(stage_idx) < STAGE_COUNT:
            raise ValueError(f"stage_idx must be 0..{STAGE_COUNT - 1}, got {stage_idx}")
        stage_onehot[int(stage_idx)] = 1.0

        mask = self._default_action_mask(stage_idx) if action_mask is None else np.asarray(action_mask, dtype=np.float32)
        if mask.shape != (ACTION_DIM,):
            raise ValueError(f"action_mask shape {mask.shape} != ({ACTION_DIM},)")
        if mask.sum() <= 0:
            raise AssertionError("All-zero action mask detected")

        tier = _encode_tier_history_5way(tier_history, stage_idx)
        costs = _encode_cost_per_stage(cost_history, self.stage_cost_clips)
        cost_summary = _encode_cost_summary(
            cost_history,
            normalized_costs=costs if self.stage_cost_clips is not None else None,
        )
        action_counts = _encode_action_count_summary(tier_history, stage_idx)

        route = np.concatenate([stage_onehot, mask, tier, costs, cost_summary, action_counts]).astype(np.float32)
        _assert_dim(route, ROUTE_COST_FEATURE_DIM, "route_cost_features")
        return route

    def encode_artifact_features(self, payload: Any, stage_idx: int) -> np.ndarray:
        full_schema = _schema_dict(_get(payload, "schema_with_descriptions", {}) or {})
        tentative = _schema_dict(_get(payload, "tentative_schema", {}) or {})
        selected_tables = list(_get(payload, "selected_tables", []) or list(tentative.keys()))
        fk_count = _foreign_key_count(payload)
        selected_fk = min(fk_count, max(len(selected_tables) - 1, 0))
        full_table_count = max(len(full_schema), 1)
        full_col_count = max(sum(len(cols) for cols in full_schema.values()), 1)
        selected_col_count = sum(len(cols) for cols in tentative.values())
        entity_item_count = _entity_item_count(payload)
        context_item_count = _context_item_count(payload)

        generated_visible = int(stage_idx) >= S7_STAGE_IDX + 1
        final_sql = str(_get(payload, "final_sql", "") or "")
        sql_visible = generated_visible and bool(final_sql)
        sql_upper = final_sql.upper() if sql_visible else ""
        parsed = 1.0 if sql_visible and "SELECT" in sql_upper else 0.0

        error_type = _execution_error_type(payload) if generated_visible else ""
        has_revision_output = bool(_get(payload, "revision_history", None) or _get(payload, "revision_count", 0))

        artifact = np.array(
            [
                1.0 if tentative else 0.0,
                _clip(len(selected_tables) / 20.0),
                _clip(selected_col_count / 100.0),
                _clip(selected_fk / 50.0),
                _clip(len(selected_tables) / full_table_count),
                _clip(selected_col_count / full_col_count),
                1.0 if selected_fk >= max(len(selected_tables) - 1, 0) and len(selected_tables) > 1 else 0.0,
                _clip(max(len(selected_tables) - 1, 0) / 10.0),
                1.0 if len(selected_tables) > 1 and selected_fk < len(selected_tables) - 1 else 0.0,
                1.0 if not tentative else 0.0,
                1.0 if selected_col_count > 60 else 0.0,
                1.0 if entity_item_count > 0 else 0.0,
                _clip(entity_item_count / 100.0),
                1.0 if context_item_count > 0 else 0.0,
                _clip(context_item_count / 100.0),
                1.0 if _get(payload, "question", None) else 0.0,
                1.0 if _get(payload, "evidence", None) else 0.0,
                1.0 if full_schema else 0.0,
                1.0 if tentative else 0.0,
                1.0 if generated_visible and _get(payload, "candidates", None) else 0.0,
                1.0 if sql_visible else 0.0,
                1.0 if error_type else 0.0,
                1.0 if generated_visible and has_revision_output else 0.0,
                1.0 if sql_visible else 0.0,
                _clip(len(final_sql.split()) / 200.0) if sql_visible else 0.0,
                1.0 if " WHERE " in f" {sql_upper} " else 0.0,
                1.0 if " JOIN " in f" {sql_upper} " else 0.0,
                1.0 if " GROUP BY " in f" {sql_upper} " else 0.0,
                1.0 if " ORDER BY " in f" {sql_upper} " else 0.0,
                1.0 if " LIMIT " in f" {sql_upper} " else 0.0,
                1.0 if any(fn in sql_upper for fn in ("COUNT(", "SUM(", "AVG(", "MAX(", "MIN(")) else 0.0,
                _clip(_sql_table_count(sql_upper) / 20.0) if sql_visible else 0.0,
                parsed,
                1.0 if error_type else 0.0,
                0.0,
                1.0 if not error_type and sql_visible else 0.0,
                1.0 if error_type == "syntax" else 0.0,
                1.0 if error_type == "schema" else 0.0,
                1.0 if error_type == "runtime" else 0.0,
                1.0 if error_type == "empty_result" else 0.0,
                1.0 if error_type == "timeout" else 0.0,
            ],
            dtype=np.float32,
        )
        _assert_dim(artifact, ARTIFACT_FEATURE_DIM, "artifact_features")
        return artifact

    @staticmethod
    def _default_action_mask(stage_idx: int) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        mask[list(legal_decision_actions(stage_idx))] = 1.0
        return mask

    @staticmethod
    def _encode_semantic_features(question_emb: np.ndarray) -> np.ndarray:
        semantic = np.asarray(question_emb, dtype=np.float32).reshape(-1)
        if semantic.shape != (SEMANTIC_DIM,):
            raise ValueError(f"semantic embedding shape {semantic.shape} != ({SEMANTIC_DIM},)")
        if not np.isfinite(semantic).all():
            raise ValueError("semantic embedding contains NaN or inf")
        return semantic


def _encode_tier_history_5way(tier_history: List[int], current_stage_idx: int) -> np.ndarray:
    rows: List[List[float]] = []
    for i in range(STAGE_COUNT):
        if i >= int(current_stage_idx):
            rows.append([0, 0, 0, 0, 1])
            continue
        action = tier_history[i] if i < len(tier_history) else None
        if action == 0:
            rows.append([1, 0, 0, 0, 0])
        elif action == 1:
            rows.append([0, 1, 0, 0, 0])
        elif action == 2:
            rows.append([0, 0, 1, 0, 0])
        elif action == 3:
            rows.append([0, 0, 0, 1, 0])
        else:
            raise ValueError(f"invalid action in tier_history at stage {i}: {action}")
    return np.asarray(rows, dtype=np.float32).reshape(-1)


def _validate_stage_cost_clips(
    stage_cost_clips: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if stage_cost_clips is None:
        return None
    clips = tuple(float(value) for value in stage_cost_clips)
    if len(clips) != STAGE_COUNT or any(
        not math.isfinite(value) or value <= 0.0 for value in clips
    ):
        raise ValueError("stage_cost_clips must contain 6 finite positive values")
    return clips


def _encode_cost_per_stage(
    cost_history: Sequence[float],
    stage_cost_clips: Sequence[float] | None = None,
) -> np.ndarray:
    vec = np.zeros(STAGE_COUNT, dtype=np.float32)
    for i, cost in enumerate(cost_history[:STAGE_COUNT]):
        if stage_cost_clips is None:
            vec[i] = _norm_cost(cost)
        else:
            vec[i] = _clip(float(cost) / float(stage_cost_clips[i]))
    return vec


def _encode_cost_summary(
    cost_history: Sequence[float],
    normalized_costs: np.ndarray | None = None,
) -> np.ndarray:
    if normalized_costs is not None:
        executed_count = min(len(cost_history), STAGE_COUNT)
        executed = normalized_costs[:executed_count]
        recent = executed[-3:]
        trend = 0.0
        if len(recent) >= 2:
            trend = _clip(
                (float(recent[-1]) - float(recent[0])) / (len(recent) - 1),
                -1.0,
                1.0,
            )
        return np.asarray(
            [
                float(normalized_costs.sum()) / STAGE_COUNT,
                float(normalized_costs[S7_STAGE_IDX]),
                float(normalized_costs[S7_STAGE_IDX + 1]),
                float(normalized_costs.max()) if normalized_costs.size else 0.0,
                trend,
            ],
            dtype=np.float32,
        )

    costs = [float(c) for c in cost_history[:STAGE_COUNT]]
    total = sum(costs)
    recent = costs[-3:]
    trend = 0.0
    if len(recent) >= 2:
        trend = _clip((recent[-1] - recent[0]) / max(len(recent) - 1, 1) / 0.05, -1.0, 1.0)
    return np.asarray(
        [
            _norm_cost(total),
            _norm_cost(costs[S7_STAGE_IDX]) if len(costs) > S7_STAGE_IDX else 0.0,
            _norm_cost(costs[S7_STAGE_IDX + 1]) if len(costs) > S7_STAGE_IDX + 1 else 0.0,
            _norm_cost(max(costs)) if costs else 0.0,
            trend,
        ],
        dtype=np.float32,
    )


def _encode_action_count_summary(tier_history: Sequence[int], current_stage_idx: int) -> np.ndarray:
    hist = list(tier_history[: int(current_stage_idx)])
    denom = max(int(current_stage_idx), 1)
    skip_count = sum(1 for a in hist if a == 3)
    consecutive_skip = len(hist) >= 2 and hist[-1] == 3 and hist[-2] == 3
    return np.asarray(
        [
            hist.count(0) / denom,
            hist.count(1) / denom,
            hist.count(2) / denom,
            skip_count / denom,
            skip_count / denom,
            1.0 if consecutive_skip else 0.0,
        ],
        dtype=np.float32,
    )


def _schema_dict(raw: Any) -> Dict[str, List[str]]:
    if not isinstance(raw, dict):
        return {}
    schema: Dict[str, List[str]] = {}
    for table, columns in raw.items():
        if isinstance(columns, dict):
            schema[str(table)] = [str(c) for c in columns.keys()]
        elif isinstance(columns, (list, tuple, set)):
            schema[str(table)] = [str(c) for c in columns]
        else:
            schema[str(table)] = []
    return schema


def _foreign_key_count(payload: Any) -> int:
    fks = _get(payload, "foreign_keys", None)
    if isinstance(fks, dict):
        return len(fks)
    if isinstance(fks, (list, tuple, set)):
        return len(fks)
    schema = _get(payload, "schema_with_descriptions", {}) or {}
    if isinstance(schema, dict):
        nested = schema.get("foreign_keys") or schema.get("fks")
        if isinstance(nested, (list, tuple, set, dict)):
            return len(nested)
    return 0


def _entity_item_count(payload: Any) -> int:
    total = 0
    for key in ("similar_values", "similar_columns"):
        value = _get(payload, key, None)
        if isinstance(value, dict):
            total += _nested_item_count(value)
        elif isinstance(value, list):
            total += len(value)
    return total


def _context_item_count(payload: Any) -> int:
    schema = _schema_dict(_get(payload, "schema_with_descriptions", {}) or {})
    return sum(len(columns) for columns in schema.values())


def _nested_item_count(value: Any) -> int:
    if isinstance(value, dict):
        if not value:
            return 0
        return sum(_nested_item_count(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 1 if value is not None else 0


def _execution_error_type(payload: Any) -> str:
    evaluation = _get(payload, "evaluation", None)
    err = ""
    if isinstance(evaluation, dict):
        err = str(evaluation.get("error_type") or evaluation.get("error") or "").lower()
    if not err:
        history = _get(payload, "revision_history", None) or []
        if history and isinstance(history[-1], dict):
            err = str(history[-1].get("error") or "").lower()
    if "timeout" in err:
        return "timeout"
    if "syntax" in err or "parse" in err:
        return "syntax"
    if "schema" in err or "column" in err or "table" in err:
        return "schema"
    if "empty" in err:
        return "empty_result"
    if err:
        return "runtime"
    return ""


def _sql_table_count(sql_upper: str) -> int:
    if not sql_upper:
        return 0
    return len(re.findall(r"\bFROM\b|\bJOIN\b", sql_upper))


def _has_date_or_time(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(19|20)\d{2}\b", lowered)) or _has_any(
        lowered,
        ("date", "year", "month", "day", "before", "after", "during", "between", "latest", "earliest"),
    )


def _has_any(text: str, words: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words)


_AGGREGATION_HINTS = (
    "count",
    "number of",
    "average",
    "avg",
    "sum",
    "total",
    "max",
    "min",
    "maximum",
    "minimum",
)
_GROUPING_PATTERN_HINTS = ("each", "per", "for each", "grouped by", "group by", "by")
_COMPARISON_HINTS = (
    "more than",
    "less than",
    "at least",
    "at most",
    "greater than",
    "between",
    "before",
    "after",
)
_RANKING_HINTS = (
    "top",
    "highest",
    "lowest",
    "latest",
    "earliest",
    "first",
    "last",
    "most",
    "least",
)
_SET_PATTERN_HINTS = (
    "both",
    "either",
    "neither",
    "except",
    "excluding",
    "in common",
    "shared",
    "intersection",
    "union",
)
_NESTED_PATTERN_HINTS = (
    "whose",
    "that have",
    "which have",
    "among",
    "those with",
    "where exists",
    "belonging to",
)
_CONDITION_CONNECTORS = ("and", "or", "with", "without", "where", "when", "for", "in", "of", "from")


def _query_complexity_features(question: str, evidence: str) -> np.ndarray:
    """Deployable weak hints from question/evidence text only, not gold SQL."""
    text = f"{question} {evidence}".lower()
    question_text = str(question or "").lower()
    aggregation = _count_hints(text, _AGGREGATION_HINTS)
    grouping = _count_hints(text, _GROUPING_PATTERN_HINTS)
    comparison = _count_hints(text, _COMPARISON_HINTS)
    ranking = _count_hints(text, _RANKING_HINTS)
    set_pattern = _count_hints(text, _SET_PATTERN_HINTS)
    nested_pattern = _count_hints(text, _NESTED_PATTERN_HINTS)
    condition = _count_hints(question_text, _CONDITION_CONNECTORS)
    total_hints = aggregation + grouping + comparison + ranking + set_pattern + nested_pattern + condition
    word_count = max(len(re.findall(r"\b\w+\b", question_text)), 1)
    return np.asarray(
        [
            _clip(aggregation / 4.0),
            _clip(grouping / 3.0),
            _clip(comparison / 4.0),
            _clip(ranking / 3.0),
            _clip(set_pattern / 3.0),
            _clip(nested_pattern / 3.0),
            _clip(condition / 8.0),
            _clip((total_hints / word_count) * 5.0),
        ],
        dtype=np.float32,
    )


def _count_hints(text: str, hints: Iterable[str]) -> int:
    return sum(len(re.findall(rf"\b{re.escape(hint)}\b", text)) for hint in hints)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(float(value), low, high))


def _log_norm(value: float, max_value: float) -> float:
    return _clip(np.log1p(max(float(value), 0.0)) / np.log1p(max_value))


def _norm_cost(value: float) -> float:
    return _clip(float(value) / 0.10)


def _assert_dim(arr: np.ndarray, expected: int, label: str) -> None:
    if arr.shape != (expected,):
        raise ValueError(f"{label} shape {arr.shape} != ({expected},)")
