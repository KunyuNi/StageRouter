"""Stagewise cache-first execution for the four RAG decisions."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from RAG.env._shared.state_payload import StatePayload
from RAG.env._shared.stage_action_registry import SKIP_ACTION

from .env_wrapper import RAGEnvironmentWrapper, compute_stage_input_hash
from .stage_mapping import (
    FIXED_STAGE_DEFAULT_TIER,
    FIXED_STAGE_IDXS,
    NUM_DECISION_STAGES,
    NUM_PHYSICAL_STAGES,
    route_key,
)
from .transition_cache import TransitionCache, TransitionKey


DECISION_BUNDLES = ((0, 1), (2, 3), (4,), (5,))
LEGAL_MASKS = (
    (True, True, True, True),
    (True, True, True, True),
    (True, True, True, True),
    (True, True, True, False),
)


@dataclass(frozen=True)
class DecisionObservation:
    question_id: int
    decision_idx: int
    payload: StatePayload
    previous_decision_payload: StatePayload | None
    action_history: tuple[int, ...]
    bundled_cost_history: tuple[float, ...]
    physical_stage_costs: tuple[float, ...]
    physical_stage_latencies_ms: tuple[float, ...]


@dataclass(frozen=True)
class DecisionStepResult:
    decision_idx: int
    action: int
    physical_stages: tuple[int, ...]
    bundled_cost: float
    bundled_latency_ms: float
    cache_hits: tuple[bool, ...]
    transition_keys: tuple[TransitionKey, ...] = ()


class RAGDecisionSession:
    """Advance a RAG query one decision bundle at a time."""

    def __init__(
        self,
        *,
        payload: StatePayload,
        env: RAGEnvironmentWrapper,
        cache: TransitionCache,
        operator_version: int,
        cache_only: bool,
    ) -> None:
        if payload.question_id is None:
            raise ValueError("RAG rollout item requires question_id/qid")
        self.payload = payload
        self.question_id = int(payload.question_id)
        self.env = env
        self.cache = cache
        self.operator_version = int(operator_version)
        self.cache_only = bool(cache_only)
        self.decision_idx = 0
        self.actions: list[int] = []
        self.bundled_costs: list[float] = []
        self.bundled_latencies_ms: list[float] = []
        self.stage_costs = [0.0] * NUM_PHYSICAL_STAGES
        self.stage_latencies_ms = [0.0] * NUM_PHYSICAL_STAGES
        self.cache_hits = [False] * NUM_PHYSICAL_STAGES
        self._decision_inputs: list[StatePayload] = []
        self._final_result: dict[str, Any] | None = None

    @classmethod
    def start(
        cls,
        item: StatePayload | Mapping[str, Any],
        env: RAGEnvironmentWrapper,
        cache: TransitionCache,
        operator_version: int = 1,
        cache_only: bool = False,
    ) -> "RAGDecisionSession":
        return cls(
            payload=env.initial_payload(item),
            env=env,
            cache=cache,
            operator_version=operator_version,
            cache_only=cache_only,
        )

    def observation(self) -> DecisionObservation:
        if self.decision_idx >= NUM_DECISION_STAGES:
            raise RuntimeError("all RAG decisions have already been applied")
        previous = self._decision_inputs[-1].snapshot() if self._decision_inputs else None
        return DecisionObservation(
            question_id=self.question_id,
            decision_idx=self.decision_idx,
            payload=self.payload.snapshot(),
            previous_decision_payload=previous,
            action_history=tuple(self.actions),
            bundled_cost_history=tuple(self.bundled_costs),
            physical_stage_costs=tuple(self.stage_costs),
            physical_stage_latencies_ms=tuple(self.stage_latencies_ms),
        )

    def _run(
        self, stage_idx: int, tier: int, action: int
    ) -> tuple[bool, TransitionKey]:
        key = TransitionKey(
            question_id=self.question_id,
            stage_idx=stage_idx,
            tier=tier,
            input_hash=compute_stage_input_hash(self.payload, stage_idx),
            operator_version=self.operator_version,
        )
        result = self.env.run_physical_stage(
            payload=self.payload,
            question_id=self.question_id,
            stage_idx=stage_idx,
            tier=tier,
            cache=self.cache,
            operator_version=self.operator_version,
            cache_only=self.cache_only,
            decision_idx=self.decision_idx,
            decision_action=action,
        )
        if result is None:
            raise LookupError(
                f"cache_only rollout missing stage {stage_idx} for qid={self.question_id}"
            )
        self.payload, cost, latency, hit = result
        self.stage_costs[stage_idx] = float(cost)
        self.stage_latencies_ms[stage_idx] = float(latency)
        self.cache_hits[stage_idx] = bool(hit)
        return bool(hit), key

    def apply_action(self, action: int) -> DecisionStepResult:
        if self._final_result is not None:
            raise RuntimeError("RAG decision session is already finalized")
        if self.decision_idx >= NUM_DECISION_STAGES:
            raise RuntimeError("all RAG decisions have already been applied")
        action = int(action)
        if action not in (0, 1, 2, SKIP_ACTION):
            raise ValueError(f"RAG action must be in 0..3, got {action!r}")
        if not LEGAL_MASKS[self.decision_idx][action]:
            if self.decision_idx == 3 and action == SKIP_ACTION:
                raise ValueError("context_question_to_answer does not allow SKIP")
            raise ValueError(
                f"illegal action {action} at RAG decision {self.decision_idx}"
            )

        current_idx = self.decision_idx
        self._decision_inputs.append(self.payload.snapshot())
        if current_idx == 0:
            stages_and_tiers = (
                (0, FIXED_STAGE_DEFAULT_TIER if action == SKIP_ACTION else action + 1),
                (1, FIXED_STAGE_DEFAULT_TIER),
            )
        elif current_idx == 1:
            stages_and_tiers = (
                ((2, FIXED_STAGE_DEFAULT_TIER),)
                if action == SKIP_ACTION
                else ((2, action + 1), (3, FIXED_STAGE_DEFAULT_TIER))
            )
        elif current_idx == 2:
            stages_and_tiers = (
                (4, FIXED_STAGE_DEFAULT_TIER if action == SKIP_ACTION else action + 1),
            )
        else:
            stages_and_tiers = ((5, action + 1),)

        executed = tuple(
            self._run(stage_idx, tier, action) for stage_idx, tier in stages_and_tiers
        )
        hits = tuple(value[0] for value in executed)
        transition_keys = tuple(value[1] for value in executed)
        physical_stages = tuple(stage_idx for stage_idx, _ in stages_and_tiers)
        bundled_cost = sum(self.stage_costs[index] for index in physical_stages)
        bundled_latency = sum(self.stage_latencies_ms[index] for index in physical_stages)
        self.actions.append(action)
        self.bundled_costs.append(bundled_cost)
        self.bundled_latencies_ms.append(bundled_latency)
        self.decision_idx += 1
        return DecisionStepResult(
            decision_idx=current_idx,
            action=action,
            physical_stages=physical_stages,
            bundled_cost=bundled_cost,
            bundled_latency_ms=bundled_latency,
            cache_hits=hits,
            transition_keys=transition_keys,
        )

    def finalize(self) -> Mapping[str, Any]:
        if self._final_result is not None:
            return copy.deepcopy(self._final_result)
        if self.decision_idx != NUM_DECISION_STAGES:
            raise RuntimeError("all four RAG decisions must be applied before finalize")

        result = self.env.run_physical_stage(
            payload=self.payload,
            question_id=self.question_id,
            stage_idx=6,
            tier=FIXED_STAGE_DEFAULT_TIER,
            cache=self.cache,
            operator_version=self.operator_version,
            cache_only=self.cache_only,
        )
        if result is None:
            raise LookupError(
                f"cache_only rollout missing stage 6 for qid={self.question_id}"
            )
        self.payload, cost, latency, hit = result
        self.stage_costs[6] = float(cost)
        self.stage_latencies_ms[6] = float(latency)
        self.cache_hits[6] = bool(hit)

        evaluation = self.payload.evaluation or {}
        if "f1_score" not in evaluation:
            raise ValueError("qa_evaluation did not return f1_score")
        f1 = float(evaluation["f1_score"])
        decision_stage_idxs = (0, 2, 4, 5)
        decision_stage_costs = [self.stage_costs[index] for index in decision_stage_idxs]
        decision_stage_latencies = [
            self.stage_latencies_ms[index] for index in decision_stage_idxs
        ]
        fixed_stage_costs = [0.0] * NUM_PHYSICAL_STAGES
        fixed_stage_latencies = [0.0] * NUM_PHYSICAL_STAGES
        for index in FIXED_STAGE_IDXS:
            fixed_stage_costs[index] = self.stage_costs[index]
            fixed_stage_latencies[index] = self.stage_latencies_ms[index]
        index_identity = self.env.index_identity or self.env._identity_from_payload(
            self.payload
        )
        self._final_result = {
            "question_id": self.question_id,
            "actions": list(self.actions),
            "route_key": route_key(self.actions),
            "answer": self.payload.answer or "",
            "ground_truth": self.payload.ground_truth or "",
            "f1_score": f1,
            "reward": f1,
            "reward_version": "raw_f1_passthrough_v1",
            "stage_costs": list(self.stage_costs),
            "decision_stage_costs": decision_stage_costs,
            "bundled_decision_costs": list(self.bundled_costs),
            "fixed_stage_costs": fixed_stage_costs,
            "total_cost": sum(self.stage_costs),
            "stage_latencies_ms": list(self.stage_latencies_ms),
            "decision_stage_latencies_ms": decision_stage_latencies,
            "bundled_decision_latencies_ms": list(self.bundled_latencies_ms),
            "fixed_stage_latencies_ms": fixed_stage_latencies,
            "total_latency_ms": sum(self.stage_latencies_ms),
            "cache_hits": list(self.cache_hits),
            "fixed_stage_idxs": list(FIXED_STAGE_IDXS),
            "fixed_stage_default_tier": FIXED_STAGE_DEFAULT_TIER,
            "index_identity": index_identity,
            "metadata": copy.deepcopy(self.payload.metadata),
        }
        return copy.deepcopy(self._final_result)


__all__ = [
    "DECISION_BUNDLES",
    "LEGAL_MASKS",
    "DecisionObservation",
    "DecisionStepResult",
    "RAGDecisionSession",
]
