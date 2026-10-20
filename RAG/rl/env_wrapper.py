"""Cache-first execution wrapper for the seven-stage RAG workflow.

The wrapper is deliberately small and RAG-local.  It reuses the transition
cache boundary used by NL2SQL, but all stage semantics and operator lookup are
defined here so that a cached cumulative ``StatePayload`` can be replayed
without re-running an LLM or embedding request.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import time
from collections.abc import Mapping
from typing import Any, Callable

from RAG.env._shared.state_payload import CostTracker, StatePayload
from RAG.env._shared.stage_action_registry import SKIP_ACTION
from RAG.env._shared.tier_registry import TierRegistry

from .stage_mapping import (
    DECISION_TO_PHYSICAL,
    FIXED_STAGE_DEFAULT_TIER,
    FIXED_STAGE_IDXS,
    NUM_DECISION_STAGES,
    NUM_PHYSICAL_STAGES,
    PHYSICAL_STAGE_NAMES,
    route_key,
)
from .transition_cache import TransitionCache, TransitionKey


# These are the only fields which determine a physical operator's semantic
# input.  In particular, usage, cost, latency, timestamps, question_type and
# evidence_list never affect a transition key.
STAGE_INPUT_FIELDS: dict[int, tuple[str, ...]] = {
    0: ("question",),
    1: ("question", "query_1"),
    2: ("question", "query_1", "context", "metadata.retrieval1"),
    3: ("query_2", "context", "metadata.retrieval1"),
    4: ("question", "context", "metadata.retrieval1", "metadata.retrieval2"),
    5: ("question", "context", "metadata.rerank"),
    6: ("answer", "ground_truth"),
}


def _as_payload_dict(payload: StatePayload | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, StatePayload):
        return payload.to_json_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a StatePayload or mapping")
    return copy.deepcopy(dict(payload))


def _field_value(payload: Mapping[str, Any], field: str) -> Any:
    """Read a top-level or dotted field without including unrelated fields."""

    value: Any = payload
    for part in field.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def compute_stage_input_hash(
    payload_dict: StatePayload | Mapping[str, Any], stage_idx: int
) -> str:
    """Compute the exact SHA256 key material for one physical stage.

    The stage index is included in the canonical envelope even though it is
    also part of :class:`TransitionKey`; this makes the helper independently
    collision-resistant and gives callers a stable, inspectable hash.
    """

    try:
        stage_idx = int(stage_idx)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid RAG physical stage index: {stage_idx!r}") from exc
    if stage_idx not in STAGE_INPUT_FIELDS:
        raise ValueError(f"stage_idx {stage_idx} has no declared semantic inputs")
    data = _as_payload_dict(payload_dict)
    selected = {
        field: _field_value(data, field) for field in STAGE_INPUT_FIELDS[stage_idx]
    }
    canonical = json.dumps(
        {"stage_idx": stage_idx, "inputs": selected},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stage_entries(payload: StatePayload, stage_name: str) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(item)
        for item in payload.cost_tracker.stage_costs
        if isinstance(item, Mapping) and item.get("stage") == stage_name
    ]


def _cost(entries: list[Mapping[str, Any]]) -> float:
    return sum(float(item.get("cost_dollar", 0.0) or 0.0) for item in entries)


def _latency(entries: list[Mapping[str, Any]]) -> float:
    return sum(float(item.get("latency_ms", 0.0) or 0.0) for item in entries)


def _set_stage_latency(payload: StatePayload, stage_name: str, latency_ms: float) -> None:
    """Use measured wall time for this stage while preserving token costs."""

    entries = [
        item
        for item in payload.cost_tracker.stage_costs
        if isinstance(item, Mapping) and item.get("stage") == stage_name
    ]
    if entries:
        previous = sum(float(item.get("latency_ms", 0.0) or 0.0) for item in entries)
        if len(entries) == 1 or previous <= 0.0:
            entries[0]["latency_ms"] = float(latency_ms)
            for item in entries[1:]:
                item["latency_ms"] = 0.0
        else:
            ratio = float(latency_ms) / previous
            for item in entries:
                item["latency_ms"] = float(item.get("latency_ms", 0.0) or 0.0) * ratio
    payload.cost_tracker.total_latency_ms = sum(
        float(item.get("latency_ms", 0.0) or 0.0)
        for item in payload.cost_tracker.stage_costs
        if isinstance(item, Mapping)
    )


def _merge_stage_entries(
    current: CostTracker, entries: list[Mapping[str, Any]]
) -> CostTracker:
    merged = copy.deepcopy(current)
    for item in entries:
        entry = copy.deepcopy(dict(item))
        merged.total_input_tokens += int(entry.get("input_tokens", 0) or 0)
        merged.total_output_tokens += int(entry.get("output_tokens", 0) or 0)
        merged.total_latency_ms += float(entry.get("latency_ms", 0.0) or 0.0)
        merged.stage_costs.append(entry)
    return merged


def _copy_stage_semantics(current: StatePayload, cached: StatePayload, stage_idx: int) -> None:
    """Copy only the semantic output produced by ``stage_idx``.

    Cached payloads contain a cumulative tracker and all prior fields.  Copying
    the entire payload would accidentally restore the previous route's cost
    history, so this narrow mapping is intentional.
    """

    if stage_idx == 0:
        current.query_1 = cached.query_1
    elif stage_idx == 1:
        current.context = copy.deepcopy(cached.context)
        if "retrieval1" in cached.metadata:
            current.metadata["retrieval1"] = copy.deepcopy(cached.metadata["retrieval1"])
        else:
            current.metadata.pop("retrieval1", None)
    elif stage_idx == 2:
        current.query_2 = cached.query_2
    elif stage_idx == 3:
        current.context = copy.deepcopy(cached.context)
        if "retrieval2" in cached.metadata:
            current.metadata["retrieval2"] = copy.deepcopy(cached.metadata["retrieval2"])
        else:
            current.metadata.pop("retrieval2", None)
    elif stage_idx == 4:
        current.context = copy.deepcopy(cached.context)
        if "rerank" in cached.metadata:
            current.metadata["rerank"] = copy.deepcopy(cached.metadata["rerank"])
        else:
            current.metadata.pop("rerank", None)
    elif stage_idx == 5:
        current.answer = cached.answer
    elif stage_idx == 6:
        current.evaluation = copy.deepcopy(cached.evaluation)
    else:  # pragma: no cover - guarded by stage mapping
        raise ValueError(f"invalid physical stage index: {stage_idx}")


class RAGEnvironmentWrapper:
    """Execute one RAG route while caching every physical transition."""

    def __init__(
        self,
        operator_pkg: str = "RAG.env.0_old",
        adapter: Any = None,
        retriever: Any = None,
        operators: Mapping[Any, Callable[..., StatePayload]] | None = None,
        index_identity: str = "",
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.operator_pkg = operator_pkg
        self.adapter = adapter
        self.retriever = retriever
        self.operators = dict(operators or {})
        self.index_identity = str(index_identity or "")
        self.clock = clock or time.perf_counter
        self._operator_module: Any = None

    def _load_operator_module(self) -> Any:
        if self._operator_module is None:
            self._operator_module = importlib.import_module(self.operator_pkg)
        return self._operator_module

    def _operator(self, stage_idx: int, tier: int) -> Callable[..., StatePayload]:
        stage_name = PHYSICAL_STAGE_NAMES[stage_idx]
        # Accept either (stage_name, tier), stage_name, or the physical index as
        # keys in tests/injected deployments.
        for key in ((stage_name, int(tier)), stage_name, (stage_idx, int(tier)), stage_idx):
            operator = self.operators.get(key)
            if operator is not None:
                return operator
        module = self._load_operator_module()
        if stage_idx in FIXED_STAGE_IDXS:
            operator = getattr(module, "FIXED_OPERATORS", {}).get(stage_name)
            if operator is None:
                raise LookupError(f"No fixed RAG operator registered for {stage_name!r}")
            return operator
        return TierRegistry.require(stage_name, int(tier))

    def _call_operator(
        self, operator: Callable[..., StatePayload], payload: StatePayload, stage_idx: int
    ) -> StatePayload:
        """Inject only kwargs accepted by the operator or fake."""

        try:
            parameters = inspect.signature(operator).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, Any] = {}
        if "adapter" in parameters and self.adapter is not None:
            kwargs["adapter"] = self.adapter
        if stage_idx in (1, 3) and "retriever" in parameters and self.retriever is not None:
            kwargs["retriever"] = self.retriever
        if stage_idx in (1, 3) and "top_k" in parameters:
            kwargs["top_k"] = 3
        result = operator(payload, **kwargs)
        if not isinstance(result, StatePayload):
            raise TypeError(
                f"RAG operator {PHYSICAL_STAGE_NAMES[stage_idx]!r} returned "
                f"{type(result).__name__}, expected StatePayload"
            )
        return result

    def _run_stage(
        self,
        payload: StatePayload,
        question_id: int,
        stage_idx: int,
        tier: int,
        cache: TransitionCache,
        operator_version: int,
        cache_only: bool,
    ) -> tuple[StatePayload, float, float, bool] | None:
        stage_name = PHYSICAL_STAGE_NAMES[stage_idx]
        input_hash = compute_stage_input_hash(payload, stage_idx)
        key = TransitionKey(
            question_id=int(question_id),
            stage_idx=stage_idx,
            tier=int(tier),
            input_hash=input_hash,
            operator_version=int(operator_version),
        )
        cached_dict = cache.lookup(key)
        if cached_dict is not None:
            cached = StatePayload.from_json_dict(cached_dict)
            entries = _stage_entries(cached, stage_name)
            _copy_stage_semantics(payload, cached, stage_idx)
            payload.cost_tracker = _merge_stage_entries(payload.cost_tracker, entries)
            return payload, _cost(entries), _latency(entries), True
        if cache_only:
            return None

        # Internally, LLM tiers are numbered 1..3 while a skipped decision
        # stage is represented by tier 0.  Do not compare against
        # ``SKIP_ACTION`` here: action id 2 (T3) is translated to tier 3.
        # Cache the unchanged semantic payload so a later replay can still
        # report an exact transition hit.
        if stage_idx in DECISION_TO_PHYSICAL and int(tier) == FIXED_STAGE_DEFAULT_TIER:
            output = payload.snapshot()
            cache.store(key, output.to_json_dict(), source="skip")
            return output, 0.0, 0.0, False

        before_entries = _stage_entries(payload, stage_name)
        started = self.clock()
        operator = self._operator(stage_idx, tier)
        output = self._call_operator(operator, payload, stage_idx)
        elapsed_ms = max(0.0, (self.clock() - started) * 1000.0)
        after_entries = _stage_entries(output, stage_name)
        new_entries = after_entries[len(before_entries) :]
        # A normal operator snapshots the input.  If a very small injected
        # fake returns an output without prior entries, treat all its entries
        # as stage-local rather than silently losing accounting.
        if len(after_entries) < len(before_entries):
            new_entries = after_entries
        _set_stage_latency(output, stage_name, elapsed_ms)
        local_entries = _stage_entries(output, stage_name)
        if len(before_entries) and len(local_entries) > len(before_entries):
            local_entries = local_entries[len(before_entries) :]
        elif len(before_entries) and len(local_entries) == len(before_entries):
            local_entries = []
        output.cost_tracker.total_latency_ms = sum(
            float(item.get("latency_ms", 0.0) or 0.0)
            for item in output.cost_tracker.stage_costs
            if isinstance(item, Mapping)
        )
        cache.store(key, output.to_json_dict(), source="live")
        return output, _cost(local_entries), _latency(local_entries), False

    def _initial_payload(self, item: StatePayload | Mapping[str, Any]) -> StatePayload:
        if isinstance(item, StatePayload):
            return item.snapshot()
        if not isinstance(item, Mapping):
            raise TypeError("rollout item must be a StatePayload or mapping")
        question = item.get("question", item.get("query"))
        answer = item.get("answer", item.get("ground_truth"))
        metadata = dict(item.get("metadata") or {})
        for key in ("question_type", "evidence_list"):
            if key in item:
                metadata[key] = copy.deepcopy(item[key])
        return StatePayload(
            question_id=item.get("question_id", item.get("qid")),
            question=None if question is None else str(question),
            ground_truth=None if answer is None else str(answer),
            metadata=metadata,
        )

    def initial_payload(self, item: StatePayload | Mapping[str, Any]) -> StatePayload:
        """Create an isolated initial payload for stagewise execution."""

        return self._initial_payload(item)

    def run_physical_stage(
        self,
        payload: StatePayload,
        question_id: int,
        stage_idx: int,
        tier: int,
        cache: TransitionCache,
        operator_version: int = 1,
        cache_only: bool = False,
        decision_idx: int | None = None,
        decision_action: int | None = None,
    ) -> tuple[StatePayload, float, float, bool] | None:
        """Execute one cache-first physical stage with stable error context."""

        try:
            return self._run_stage(
                payload,
                int(question_id),
                int(stage_idx),
                int(tier),
                cache,
                int(operator_version),
                bool(cache_only),
            )
        except Exception as exc:
            owner_decision_idx = decision_idx
            if owner_decision_idx is None:
                owner_decision_idx = {1: 0, 3: 1}.get(int(stage_idx))
            tier_name = (
                "fixed"
                if int(stage_idx) in FIXED_STAGE_IDXS
                else ("SKIP" if int(tier) == 0 else f"T{int(tier)}")
            )
            context = {
                "question_id": int(question_id),
                "stage_index": int(stage_idx),
                "stage_name": PHYSICAL_STAGE_NAMES[int(stage_idx)],
                "decision_stage_index": owner_decision_idx,
                "decision_action": (
                    int(decision_action) if decision_action is not None else None
                ),
                "tier": int(tier),
                "tier_name": tier_name,
            }
            for name, value in context.items():
                if not hasattr(exc, name):
                    setattr(exc, name, value)
            raise

    @staticmethod
    def _identity_from_payload(payload: StatePayload) -> str:
        for name in ("retrieval2", "retrieval1"):
            stage = payload.metadata.get(name, {})
            if isinstance(stage, Mapping) and stage.get("index_identity"):
                return str(stage["index_identity"])
        return ""

    def rollout_with_cache(
        self,
        item: StatePayload | Mapping[str, Any],
        actions: list[int] | tuple[int, ...],
        cache: TransitionCache,
        operator_version: int = 1,
        cache_only: bool = False,
    ) -> dict[str, Any] | None:
        """Run actions ``[a0, a1, a2, a3]`` through physical stages 0..6."""

        actions = [int(action) for action in actions]
        if len(actions) != NUM_DECISION_STAGES:
            raise ValueError(f"RAG rollout requires {NUM_DECISION_STAGES} actions")
        if any(action not in (0, 1, 2, SKIP_ACTION) for action in actions):
            raise ValueError(f"RAG actions must be in 0..3, got {actions!r}")
        payload = self._initial_payload(item)
        question_id = payload.question_id
        if question_id is None:
            raise ValueError("RAG rollout item requires question_id/qid")
        question_id = int(question_id)

        stage_costs = [0.0] * NUM_PHYSICAL_STAGES
        stage_latencies = [0.0] * NUM_PHYSICAL_STAGES
        decision_stage_costs = [0.0] * NUM_DECISION_STAGES
        decision_stage_latencies = [0.0] * NUM_DECISION_STAGES
        fixed_stage_costs = [0.0] * NUM_PHYSICAL_STAGES
        fixed_stage_latencies = [0.0] * NUM_PHYSICAL_STAGES
        cache_hits = [False] * NUM_PHYSICAL_STAGES

        def execute(stage_idx: int, tier: int, decision_idx: int | None = None) -> bool:
            nonlocal payload
            result = self.run_physical_stage(
                payload=payload,
                question_id=question_id,
                stage_idx=stage_idx,
                tier=tier,
                cache=cache,
                operator_version=operator_version,
                cache_only=cache_only,
                decision_idx=decision_idx,
                decision_action=(
                    actions[decision_idx] if decision_idx is not None else None
                ),
            )
            if result is None:
                raise LookupError(
                    f"cache_only rollout missing stage {stage_idx} for qid={question_id}"
                )
            payload, cost, latency, hit = result
            stage_costs[stage_idx] = cost
            stage_latencies[stage_idx] = latency
            cache_hits[stage_idx] = hit
            if decision_idx is not None:
                decision_stage_costs[decision_idx] = cost
                decision_stage_latencies[decision_idx] = latency
            elif stage_idx in FIXED_STAGE_IDXS:
                fixed_stage_costs[stage_idx] = cost
                fixed_stage_latencies[stage_idx] = latency
            return hit

        # Decision 0 and Retrieval 1 are always coupled.  A skipped rewrite
        # still permits retrieval1 to use the original question fallback.
        if actions[0] == SKIP_ACTION:
            execute(0, FIXED_STAGE_DEFAULT_TIER, 0)
        else:
            execute(0, actions[0] + 1, 0)
        execute(1, FIXED_STAGE_DEFAULT_TIER)

        # Second-hop SKIP bypasses both the rewrite and retrieval2.
        if actions[1] == SKIP_ACTION:
            execute(2, FIXED_STAGE_DEFAULT_TIER, 1)
        else:
            execute(2, actions[1] + 1, 1)
            execute(3, FIXED_STAGE_DEFAULT_TIER)

        if actions[2] == SKIP_ACTION:
            execute(4, FIXED_STAGE_DEFAULT_TIER, 2)
        else:
            execute(4, actions[2] + 1, 2)

        # Answer generation is mandatory; action 3 is intentionally rejected.
        if actions[3] == SKIP_ACTION:
            raise ValueError("context_question_to_answer does not allow SKIP")
        execute(5, actions[3] + 1, 3)
        execute(6, FIXED_STAGE_DEFAULT_TIER)

        evaluation = payload.evaluation or {}
        if "f1_score" not in evaluation:
            raise ValueError("qa_evaluation did not return f1_score")
        f1 = float(evaluation["f1_score"])
        index_identity = self.index_identity or self._identity_from_payload(payload)
        total_cost = sum(stage_costs)
        total_latency = sum(stage_latencies)
        return {
            "question_id": question_id,
            "actions": actions,
            "route_key": route_key(actions),
            "answer": payload.answer or "",
            "ground_truth": payload.ground_truth or "",
            "f1_score": f1,
            "reward": f1,
            "reward_version": "raw_f1_passthrough_v1",
            "stage_costs": stage_costs,
            "decision_stage_costs": decision_stage_costs,
            "fixed_stage_costs": fixed_stage_costs,
            "total_cost": total_cost,
            "stage_latencies_ms": stage_latencies,
            "decision_stage_latencies_ms": decision_stage_latencies,
            "fixed_stage_latencies_ms": fixed_stage_latencies,
            "total_latency_ms": total_latency,
            "cache_hits": cache_hits,
            "fixed_stage_idxs": list(FIXED_STAGE_IDXS),
            "fixed_stage_default_tier": FIXED_STAGE_DEFAULT_TIER,
            "index_identity": index_identity,
            "metadata": copy.deepcopy(payload.metadata),
        }


def rollout_with_cache(
    item: StatePayload | Mapping[str, Any],
    actions: list[int] | tuple[int, ...],
    cache: TransitionCache,
    env: RAGEnvironmentWrapper | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Module-level convenience wrapper mirroring the NL2SQL API."""

    runner = env or RAGEnvironmentWrapper()
    return runner.rollout_with_cache(item, actions, cache, **kwargs)


__all__ = [
    "RAGEnvironmentWrapper",
    "STAGE_INPUT_FIELDS",
    "compute_stage_input_hash",
    "rollout_with_cache",
]
