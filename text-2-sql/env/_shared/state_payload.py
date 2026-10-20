"""
统一 State Payload 协议与成本追踪器

设计原则:
  1. 所有 9 个阶段的 Tier 1 算子必须使用相同的 I/O 协议
  2. 阶段间通过标准化键名传递，支持 Tier 降级兼容
  3. CostTracker 实时记录 Token 消耗与延迟，供最终消融分析

Pricing is defined in model_config.py.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from .model_config import (
    EMBEDDING_PRICING_DOLLAR_PER_1M,
    LLM_PRICING_DOLLAR_PER_1M,
)

# ============================================================
# Qwen 定价常量
# ============================================================
QWEN_PRICING: Dict[str, Dict[str, float]] = LLM_PRICING_DOLLAR_PER_1M


def _embedding_price_dollar_per_1m(model: str) -> float:
    normalized = (model or "").upper().replace("-", "_").replace(".", "_")
    env_name = f"NL2SQL_EMBEDDING_{normalized}_PRICE_PER_1M_DOLLAR"
    raw = os.getenv(env_name)
    if raw:
        return float(raw)
    if model in EMBEDDING_PRICING_DOLLAR_PER_1M:
        return EMBEDDING_PRICING_DOLLAR_PER_1M[model]
    return 0.0


# ============================================================
# CostTracker — 跨阶段累积成本追踪
# ============================================================
@dataclass
class CostTracker:
    """实时追踪全链路 Token 消耗、延迟、资金花销。"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0

    # 按阶段/模型的明细
    stage_costs: List[Dict[str, Any]] = field(default_factory=list)

    def record_llm_call(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ):
        """记录一次 LLM 调用到累积统计。"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_ms += latency_ms

        pricing = QWEN_PRICING.get(model, QWEN_PRICING.get(model.lower(), None) if model else None)
        if pricing is None:
            import logging
            logging.warning("[CostTracker] Unknown model '%s' — cost will be 0. "
                          "Add pricing to LLM_PRICING_DOLLAR_PER_1M in _shared/model_config.py", model)
            pricing = {"input": 0.0, "output": 0.0}
        call_cost = (input_tokens / 1_000_000) * pricing["input"] + \
                    (output_tokens / 1_000_000) * pricing["output"]

        self.stage_costs.append({
            "stage": stage,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_dollar": call_cost,
        })

    def record_embedding_call(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        latency_ms: float,
    ):
        """记录一次 embedding API 调用到累积统计。"""
        self.total_input_tokens += input_tokens
        self.total_latency_ms += latency_ms

        input_price = _embedding_price_dollar_per_1m(model)
        call_cost = (input_tokens / 1_000_000) * input_price

        self.stage_costs.append({
            "stage": stage,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "latency_ms": latency_ms,
            "cost_dollar": call_cost,
            "call_type": "embedding",
        })

    def total_cost_dollar(self) -> float:
        """Compute the total monetary cost across workflow stages."""
        total = 0.0
        for entry in self.stage_costs:
            total += entry["cost_dollar"]
        return total

    def summary(self) -> Dict[str, Any]:
        """返回汇总统计字典，供消融测试输出。"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_latency_ms": self.total_latency_ms,
            "total_latency_s": round(self.total_latency_ms / 1000.0, 2),
            "total_cost_dollar": round(self.total_cost_dollar(), 6),
            "total_cost_dollar_per_1k": round(self.total_cost_dollar() * 1000, 4),
            "stage_breakdown": self.stage_costs,
        }


# ============================================================
# StatePayload — 统一跨阶段流通字典
# ============================================================
@dataclass
class StatePayload:
    """
    9 阶段 Tier 1 算子的统一 I/O 协议。
    每个阶段的算子接收并返回此结构，保证跨级兼容。

    字段按阶段 1→9 逐步填充，初始值为 None/空。
    """
    # --- 任务元信息 (输入，不变) ---
    question_id: Optional[int] = None
    db_id: Optional[str] = None
    question: Optional[str] = None
    evidence: Optional[str] = None
    gold_sql: Optional[str] = None

    # --- Stage 1: Keyword Extraction ---
    keywords: List[Dict[str, Any]] = field(default_factory=list)
    # [{"value": str, "type": "entity"|"predicate"|"aggregation"|"condition_value"|"ordering",
    #   "confidence": float}, ...]

    # --- Stage 2: Entity Retrieval ---
    similar_values: Dict[str, Dict[str, List[Dict]]] = field(default_factory=dict)
    # {table: {column: [{"value": str, "score": float, "method": str}, ...]}}

    similar_columns: Dict[str, List[str]] = field(default_factory=dict)
    # {table: [column_names]}  — 列名匹配结果 (get_similar_columns)

    # --- Stage 3: Context Retrieval ---
    schema_with_descriptions: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    # {table: {column: {column_description, value_description, data_format, ...}}}

    # --- Stage 4-6: Schema Linking ---
    tentative_schema: Dict[str, List[str]] = field(default_factory=dict)
    # {table: [column_names]}  — 逐步裁剪

    selected_tables: List[str] = field(default_factory=list)

    column_roles: Dict[str, str] = field(default_factory=dict)
    # {"table.column": "SELECT"|"WHERE"|"JOIN"|"GROUPBY"|"ORDERBY"|"HAVING"}

    # --- Stage 7: Candidate Generation ---
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    # [{"sql": str, "strategy": str, "confidence": float, "chain_of_thought": str, "token_cost": int}, ...]

    # --- Stage 8: Revision ---
    final_sql: Optional[str] = None
    revision_count: int = 0
    errors_fixed: List[str] = field(default_factory=list)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)

    # --- Stage 9: Evaluation ---
    evaluation: Optional[Dict[str, Any]] = None
    # {"exec_res": 0|1, "exec_err": str, "predicted_sql": str, "gold_sql": str}

    # --- 全局追踪 ---
    cost_tracker: CostTracker = field(default_factory=CostTracker)

    def to_json_dict(self) -> dict:
        """完整可逆序列化。返回纯 dict，所有字段可 JSON 序列化。"""
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "question": self.question,
            "evidence": self.evidence,
            "gold_sql": self.gold_sql,
            "keywords": self.keywords,
            "similar_values": self.similar_values,
            "similar_columns": self.similar_columns,
            "schema_with_descriptions": self.schema_with_descriptions,
            "tentative_schema": self.tentative_schema,
            "selected_tables": self.selected_tables,
            "column_roles": self.column_roles,
            "candidates": self.candidates,
            "final_sql": self.final_sql,
            "revision_count": self.revision_count,
            "errors_fixed": self.errors_fixed,
            "revision_history": self.revision_history,
            "evaluation": self.evaluation,
            "cost_tracker": {
                "total_input_tokens": self.cost_tracker.total_input_tokens,
                "total_output_tokens": self.cost_tracker.total_output_tokens,
                "total_latency_ms": self.cost_tracker.total_latency_ms,
                "stage_costs": self.cost_tracker.stage_costs,
            },
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "StatePayload":
        """从 to_json_dict 的输出还原 StatePayload。"""
        ct_data = d.get("cost_tracker", {})
        stage_costs = ct_data.get("stage_costs", [])
        if any(not isinstance(entry, dict) or "cost_dollar" not in entry for entry in stage_costs):
            raise ValueError("cached stage-cost entry lacks cost_dollar; rebuild the cache")
        payload = cls(
            question_id=d.get("question_id"),
            db_id=d.get("db_id"),
            question=d.get("question"),
            evidence=d.get("evidence"),
            gold_sql=d.get("gold_sql"),
            keywords=d.get("keywords", []),
            similar_values=d.get("similar_values", {}),
            similar_columns=d.get("similar_columns", {}),
            schema_with_descriptions=d.get("schema_with_descriptions", {}),
            tentative_schema=d.get("tentative_schema", {}),
            selected_tables=d.get("selected_tables", []),
            column_roles=d.get("column_roles", {}),
            candidates=d.get("candidates", []),
            final_sql=d.get("final_sql"),
            revision_count=d.get("revision_count", 0),
            errors_fixed=d.get("errors_fixed", []),
            revision_history=d.get("revision_history", []),
            evaluation=d.get("evaluation"),
            cost_tracker=CostTracker(
                total_input_tokens=ct_data.get("total_input_tokens", 0),
                total_output_tokens=ct_data.get("total_output_tokens", 0),
                total_latency_ms=ct_data.get("total_latency_ms", 0.0),
            ),
        )
        payload.cost_tracker.stage_costs = stage_costs
        return payload

    def snapshot(self) -> "StatePayload":
        """Create a deep copy of the current state for before/after comparison."""
        return StatePayload.from_json_dict(self.to_json_dict())

    def to_summary(self) -> Dict[str, Any]:
        """序列化为字典 (用于日志/调试)。"""
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "keywords_count": len(self.keywords),
            "similar_values_tables": list(self.similar_values.keys()),
            "schema_tables": list(self.tentative_schema.keys()),
            "selected_tables": self.selected_tables,
            "candidates_count": len(self.candidates),
            "final_sql": self.final_sql,
            "revision_count": self.revision_count,
            "evaluation": self.evaluation,
            "cost_summary": self.cost_tracker.summary(),
        }
