"""Two-level cache-first Dev/Test policy evaluation."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .decision_session import RAGDecisionSession
from .error_policy import is_content_safety_error, is_fatal_collection_error
from .failure_queue import EvaluationFailureQueue
from .stage_mapping import route_key
from .trajectory_buffer import TrajectoryBuffer, TrajectoryRecord


@dataclass(frozen=True)
class PolicyPrediction:
    question_id: int
    policy_id: str
    actions: tuple[int, int, int, int]
    route_key: str
    f1_score: float
    total_cost: float
    total_latency_ms: float
    transition_hits: int
    transition_misses: int
    route_hit: bool
    incremental_api_cost: float
    operator_calls: int
    qa_evaluation_calls: int
    split: str
    operator_version: int
    workflow_identity: str

    @property
    def route_identity(self) -> tuple[str, int, str, int, str]:
        return (
            self.split,
            self.question_id,
            self.route_key,
            self.operator_version,
            self.workflow_identity,
        )


class TrajectoryBufferWriter:
    def __init__(self, buffer: TrajectoryBuffer, *, autosave_every: int = 10) -> None:
        self.buffer = buffer
        self.autosave_every = max(1, int(autosave_every))
        self._pending = 0
        self._lock = threading.RLock()

    def write(self, record: TrajectoryRecord) -> None:
        with self._lock:
            self.buffer.upsert_by_question_route(record)
            self._pending += 1
            if self._pending >= self.autosave_every:
                self.buffer.save()
                self._pending = 0

    def flush(self) -> None:
        with self._lock:
            if self._pending:
                self.buffer.save()
                self._pending = 0


def _select_action(
    *,
    model: Any,
    policy: Any,
    observation: Any,
    state_encoder: Any,
    inference_lock: threading.RLock,
) -> int:
    if hasattr(model, "select_action"):
        with inference_lock:
            return int(model.select_action(observation))
    if state_encoder is None or policy is None:
        raise ValueError("DQN evaluation requires state_encoder and policy")
    with inference_lock:
        encoded = state_encoder.encode(observation)
        state = torch.from_numpy(encoded.vector).reshape(1, -1)
        legal = torch.from_numpy(encoded.legal_action_mask).reshape(1, -1)
        device = next(model.parameters()).device
        state, legal = state.to(device), legal.to(device)
        with torch.inference_mode():
            values = model(state, legal)
        prior_skip_t1_t2_t3 = np.asarray(
            state_encoder.cost_priors.means[observation.decision_idx],
            dtype=np.float32,
        )
        empirical = torch.tensor(
            prior_skip_t1_t2_t3[[1, 2, 3, 0]], device=device
        )
        return int(
            policy.select(
                q=values[0], legal_mask=legal[0], empirical_cost=empirical
            )
        )


def _validate_route_hit(
    record: TrajectoryRecord,
    *,
    session: RAGDecisionSession,
    actions: Sequence[int],
    workflow_identity: str,
    operator_version: int,
) -> None:
    if list(record.actions) != [int(value) for value in actions]:
        raise ValueError("route cache actions mismatch")
    if record.answer != (session.payload.answer or ""):
        raise ValueError("route cache answer mismatch")
    if record.workflow_identity != workflow_identity:
        raise ValueError("route cache workflow identity mismatch")
    if int(record.operator_version) != int(operator_version):
        raise ValueError("route cache operator version mismatch")
    expected_index = session.env.index_identity or session.env._identity_from_payload(
        session.payload
    )
    if record.index_identity and expected_index and record.index_identity != expected_index:
        raise ValueError("route cache index identity mismatch")
    if abs(float(record.total_cost) - sum(session.stage_costs)) > 1e-9:
        raise ValueError("route cache total cost mismatch")


def evaluate_query(
    *,
    model: Any,
    policy: Any,
    item: Mapping[str, Any],
    env: Any,
    transition_cache: Any,
    trajectory_buffer: TrajectoryBuffer,
    writer: TrajectoryBufferWriter,
    failure_queue: EvaluationFailureQueue,
    operator_version: int,
    workflow_identity: str,
    state_encoder: Any = None,
    split: str = "test",
    policy_id: str | None = None,
    cache_only: bool = False,
    inference_lock: threading.RLock | None = None,
) -> PolicyPrediction | None:
    policy_id = str(policy_id or failure_queue.policy_id)
    lock = inference_lock or threading.RLock()
    session = RAGDecisionSession.start(
        item=item,
        env=env,
        cache=transition_cache,
        operator_version=operator_version,
        cache_only=cache_only,
    )
    actions: list[int] = []
    transition_hits = 0
    transition_misses = 0
    incremental_cost = 0.0
    operator_calls = 0
    try:
        for _ in range(4):
            observation = session.observation()
            action = _select_action(
                model=model,
                policy=policy,
                observation=observation,
                state_encoder=state_encoder,
                inference_lock=lock,
            )
            actions.append(action)
            step = session.apply_action(action)
            transition_hits += sum(step.cache_hits)
            transition_misses += sum(not value for value in step.cache_hits)
            for stage_idx, hit in zip(step.physical_stages, step.cache_hits):
                if not hit:
                    incremental_cost += session.stage_costs[stage_idx]
                    if not (action == 3 and stage_idx in (0, 2, 4)):
                        operator_calls += 1

        key = route_key(actions)
        cached = trajectory_buffer.get_exact(
            split,
            session.question_id,
            key,
            operator_version,
            workflow_identity,
        )
        if cached is not None:
            _validate_route_hit(
                cached,
                session=session,
                actions=actions,
                workflow_identity=workflow_identity,
                operator_version=operator_version,
            )
            failure_queue.resolve(session.question_id)
            return PolicyPrediction(
                question_id=session.question_id,
                policy_id=policy_id,
                actions=tuple(actions),  # type: ignore[arg-type]
                route_key=key,
                f1_score=float(cached.f1_score),
                total_cost=float(cached.total_cost),
                total_latency_ms=float(cached.total_latency_ms),
                transition_hits=transition_hits,
                transition_misses=transition_misses,
                route_hit=True,
                incremental_api_cost=incremental_cost,
                operator_calls=operator_calls,
                qa_evaluation_calls=0,
                split=split,
                operator_version=operator_version,
                workflow_identity=workflow_identity,
            )

        result = session.finalize()
        stage6_hit = bool(result["cache_hits"][6])
        transition_hits += int(stage6_hit)
        transition_misses += int(not stage6_hit)
        incremental_cost += 0.0 if stage6_hit else float(result["stage_costs"][6])
        index_identity = str(result["index_identity"])
        record = TrajectoryRecord(
            question_id=session.question_id,
            split=split,
            route_name=f"policy:{policy_id}",
            route_key=key,
            actions=list(actions),
            f1_score=float(result["f1_score"]),
            reward=float(result["reward"]),
            answer=str(result["answer"]),
            ground_truth=str(result["ground_truth"]),
            total_cost=float(result["total_cost"]),
            stage_costs=list(result["stage_costs"]),
            decision_stage_costs=list(result["decision_stage_costs"]),
            fixed_stage_costs=list(result["fixed_stage_costs"]),
            total_latency_ms=float(result["total_latency_ms"]),
            stage_latencies_ms=list(result["stage_latencies_ms"]),
            decision_stage_latencies_ms=list(
                result["decision_stage_latencies_ms"]
            ),
            fixed_stage_latencies_ms=list(result["fixed_stage_latencies_ms"]),
            metadata={**copy.deepcopy(result["metadata"]), "policy_id": policy_id},
            operator_version=operator_version,
            workflow_identity=workflow_identity,
            index_identity=index_identity,
        )
        writer.write(record)
        failure_queue.resolve(session.question_id)
        return PolicyPrediction(
            question_id=session.question_id,
            policy_id=policy_id,
            actions=tuple(actions),  # type: ignore[arg-type]
            route_key=key,
            f1_score=float(record.f1_score),
            total_cost=float(record.total_cost),
            total_latency_ms=float(record.total_latency_ms),
            transition_hits=transition_hits,
            transition_misses=transition_misses,
            route_hit=False,
            incremental_api_cost=incremental_cost,
            operator_calls=operator_calls,
            qa_evaluation_calls=int(not stage6_hit),
            split=split,
            operator_version=operator_version,
            workflow_identity=workflow_identity,
        )
    except Exception as exc:
        if is_fatal_collection_error(exc):
            raise
        failure_queue.record_failure(
            {
                "question_id": session.question_id,
                "policy_id": policy_id,
                "operator_version": operator_version,
                "workflow_identity": workflow_identity,
                "selected_prefix": actions,
                "route_key": "-".join(str(value) for value in actions),
                "stage_index": getattr(exc, "stage_index", None),
                "stage_name": getattr(exc, "stage_name", None),
                "decision_stage_index": getattr(
                    exc, "decision_stage_index", session.decision_idx
                ),
                "decision_action": getattr(
                    exc, "decision_action", actions[-1] if actions else None
                ),
                "tier": getattr(exc, "tier", None),
                "tier_name": getattr(exc, "tier_name", None),
                "error": exc,
                "error_type": type(exc).__name__,
                "error_category": (
                    "content_safety" if is_content_safety_error(exc) else None
                ),
            }
        )
        if is_content_safety_error(exc):
            failure_queue.exclude_question(session.question_id)
        return None


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    completed: int
    failed: int
    predictions: tuple[PolicyPrediction, ...]


def evaluate_policy(
    *,
    items: Sequence[Mapping[str, Any]],
    n_workers: int = 64,
    writer: TrajectoryBufferWriter,
    **kwargs: Any,
) -> PolicyEvaluationSummary:
    predictions: list[PolicyPrediction] = []
    failed = 0
    inference_lock = threading.RLock()
    with ThreadPoolExecutor(max_workers=int(n_workers)) as executor:
        futures = [
            executor.submit(
                evaluate_query,
                item=item,
                writer=writer,
                inference_lock=inference_lock,
                **kwargs,
            )
            for item in items
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                failed += 1
            else:
                predictions.append(result)
    writer.flush()
    predictions.sort(key=lambda row: row.question_id)
    return PolicyEvaluationSummary(len(predictions), failed, tuple(predictions))


def build_safety_union(policy_safety_qids: Mapping[str, set[int]]) -> set[int]:
    result: set[int] = set()
    for values in policy_safety_qids.values():
        result.update(int(value) for value in values)
    return result


__all__ = [
    "PolicyEvaluationSummary",
    "PolicyPrediction",
    "TrajectoryBufferWriter",
    "build_safety_union",
    "evaluate_policy",
    "evaluate_query",
]
