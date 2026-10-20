"""Serializable state exchanged by every RAG workflow stage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .model_config import EMBEDDING_PRICING_DOLLAR_PER_1M, LLM_PRICING_DOLLAR_PER_1M


@dataclass
class CostTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    stage_costs: List[Dict[str, Any]] = field(default_factory=list)

    def record_llm_call(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        pricing = LLM_PRICING_DOLLAR_PER_1M.get(
            model, {"input": 0.0, "output": 0.0}
        )
        cost_dollar = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1_000_000
        self.total_input_tokens += int(input_tokens)
        self.total_output_tokens += int(output_tokens)
        self.total_latency_ms += float(latency_ms)
        self.stage_costs.append(
            {
                "stage": stage,
                "model": model,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "latency_ms": float(latency_ms),
                "cost_dollar": cost_dollar,
                "call_type": "llm",
            }
        )

    def record_embedding_call(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record one fixed-model embedding request.

        Embeddings have no output-token charge, but their input usage and
        latency are part of the same cumulative tracker as LLM calls.  This
        deliberately mirrors the NL2SQL tracker contract while remaining
        local to the RAG environment.
        """
        input_tokens = int(input_tokens)
        latency_ms = float(latency_ms)
        price = float(EMBEDDING_PRICING_DOLLAR_PER_1M.get(model, 0.0))
        cost_dollar = input_tokens * price / 1_000_000
        self.total_input_tokens += input_tokens
        self.total_latency_ms += latency_ms
        self.stage_costs.append(
            {
                "stage": stage,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "latency_ms": latency_ms,
                "cost_dollar": cost_dollar,
                "call_type": "embedding",
            }
        )

    def total_cost_dollar(self) -> float:
        return sum(float(item.get("cost_dollar", 0.0)) for item in self.stage_costs)


@dataclass
class StatePayload:
    question_id: Optional[str] = None
    question: Optional[str] = None
    ground_truth: Optional[str] = None
    query_1: Optional[str] = None
    query_2: Optional[str] = None
    context: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cost_tracker: CostTracker = field(default_factory=CostTracker)

    def to_json_dict(self) -> Dict[str, Any]:
        return deepcopy(asdict(self))

    @classmethod
    def from_json_dict(cls, value: Dict[str, Any]) -> "StatePayload":
        data = deepcopy(value)
        tracker_data = data.pop("cost_tracker", {})
        if any(
            isinstance(entry, Mapping) and "cost_dollar" not in entry
            for entry in tracker_data.get("stage_costs", [])
        ):
            raise ValueError("cached stage-cost entry lacks cost_dollar; rebuild the cache with the current schema")
        tracker = CostTracker(**tracker_data)
        return cls(cost_tracker=tracker, **data)

    def snapshot(self) -> "StatePayload":
        return self.from_json_dict(self.to_json_dict())
