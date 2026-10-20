"""Workflow execution and route caching for dev evaluation."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from env._shared.model_config import DEFAULT_LLM_MODEL, EMBEDDING_MODEL, TIER_LLM_MODELS
from env._shared.state_payload import CostTracker, StatePayload
from rl.api_guard import ApiQuotaExhaustedError
from rl.env_wrapper import retryable_infrastructure_identity
from rl.reward import RewardConfig, RewardEvaluator
from rl.route_ladder import route_key, route_name
from rl.stage_cost_normalization import load_global_stage_cost_normalizer
from rl.stage_mapping import (
    CANDIDATE_GENERATION_DECISION_IDX,
    FIXED_PIPELINE_STAGE_IDXS,
    NUM_PIPELINE_STAGES,
    NUM_RL_STAGES,
    RL_DECISION_TO_PIPELINE_STAGE_IDXS,
    decision_to_pipeline_stage_idx,
)
from rl.trajectory_buffer import TrajectoryBuffer, TrajectoryRecord

DECISION_STAGE_LABELS = ("S1", "S4", "S5", "S6", "S7", "S8")


def default_model_config_version() -> str:
    return (
        f"t1={TIER_LLM_MODELS[1]}|t2={TIER_LLM_MODELS[2]}|"
        f"t3={TIER_LLM_MODELS[3]}|embed={EMBEDDING_MODEL}"
    )


def dev_environment_kwargs(
    config: Mapping[str, Any],
    *,
    db_manager_factory: Any,
    total_budget: float,
    cache_identity: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    if config.get("exact_replay_only") is not True:
        raise ValueError("State V26 requires an exact transition cache")
    return {
        "operator_pkg": str(config["operator_pkg"]),
        "db_manager_factory": db_manager_factory,
        "total_budget": total_budget,
        "exact_replay_only": False,
        "exact_only": True,
        "cache_identity": cache_identity,
        "model": DEFAULT_LLM_MODEL,
    }


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _cost_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    return [_finite_nonnegative(item, name) for item in value]


def _action_mask(valid_actions: Sequence[Any], decision_idx: int) -> np.ndarray:
    mask = np.zeros(4, dtype=np.float32)
    for value in valid_actions:
        if isinstance(value, bool) or int(value) != value or not 0 <= int(value) < 4:
            raise ValueError("environment action mask contains an invalid index")
        mask[int(value)] = 1.0
    if decision_idx == CANDIDATE_GENERATION_DECISION_IDX:
        mask[3] = 0.0
    if not mask.any():
        raise ValueError("environment action mask has no legal actions")
    return mask


def _reward_evaluator(records: Sequence[TrajectoryRecord], normalizer_path: str) -> RewardEvaluator:
    if not records:
        raise ValueError("dev trajectory buffer must not be empty")
    metadata = records[0].metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("dev trajectory buffer reward metadata is missing")
    config = RewardConfig(
        version=str(metadata.get("reward_version", "")),
        beta=float(metadata.get("reward_beta")),
        success_bonus=float(metadata.get("reward_success_bonus")),
        cost_weight=float(metadata.get("reward_cost_weight")),
    ).normalized()
    normalizer = load_global_stage_cost_normalizer(normalizer_path)
    if metadata.get("reward_normalizer_sha256") != normalizer.artifact_sha256:
        raise ValueError("dev trajectory buffer reward normalizer SHA mismatch")
    return RewardEvaluator(config, normalizer=normalizer)


class DevTrajectoryCache:
    """Read and append the existing dev trajectory-buffer format."""

    def __init__(self, path: str | Path, *, normalizer_path: str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.buffer = TrajectoryBuffer(capacity=100_000)
        self.buffer.load(str(self.path))
        self.reward_evaluator = _reward_evaluator(self.buffer.records, normalizer_path)
        self._index: dict[tuple[int, str], TrajectoryRecord] = {}
        for record in self.buffer.records:
            key = (
                int(record.question_id),
                str((record.metadata or {}).get("route_key") or route_key(record.actions)),
            )
            if key in self._index:
                raise ValueError("dev trajectory buffer has a duplicate qid-route pair")
            self._index[key] = record

    def get(self, *, qid: int, route_key: str, db_id: str | None = None, **_kwargs: Any) -> dict[str, Any] | None:
        record = self._index.get((int(qid), str(route_key)))
        if record is None:
            return None
        metadata = dict(record.metadata or {})
        return {
            "qid": int(record.question_id),
            "db_id": str(db_id or metadata.get("db_id", "")),
            "actions": [int(value) for value in record.actions],
            "route_key": str(route_key),
            "exec_res": int(record.exec_res),
            "total_cost": float(record.total_cost),
            "stage_costs": list(record.stage_costs),
            "decision_stage_costs": list(record.decision_stage_costs),
            "fixed_stage_costs": list(record.fixed_stage_costs),
            "total_latency_ms": float(record.total_latency_ms),
            "stage_latencies_ms": list(record.stage_latencies_ms),
            "decision_stage_latencies_ms": list(record.decision_stage_latencies_ms),
            "fixed_stage_latencies_ms": list(record.fixed_stage_latencies_ms),
            "final_sql": str(metadata.get("final_sql", "")),
            "error": str(metadata.get("error", "") or ""),
        }

    def append(self, record: Mapping[str, Any]) -> None:
        actions = [int(value) for value in record["actions"]]
        key = (int(record["qid"]), route_key(actions))
        if key in self._index:
            raise ValueError("dev trajectory buffer already contains qid-route pair")
        reward_result = self.reward_evaluator.evaluate(
            exec_res=int(record["exec_res"]),
            total_cost=float(record["total_cost"]),
            decision_stage_costs=list(record["decision_stage_costs"]),
        )
        metadata = {
            "cache_hits": list(record.get("transition_cache_hits", [])),
            "skip_stages": [
                DECISION_STAGE_LABELS[index]
                for index, action in enumerate(actions)
                if action == 3
            ],
            "action_masks": list(record.get("action_masks", [])),
            "decision_stage_costs": list(record["decision_stage_costs"]),
            "decision_stage_latencies_ms": list(record["decision_stage_latencies_ms"]),
            "fixed_stage_costs": list(record["fixed_stage_costs"]),
            "fixed_stage_latencies_ms": list(record["fixed_stage_latencies_ms"]),
            "fixed_stage_idxs": list(FIXED_PIPELINE_STAGE_IDXS),
            "fixed_stage_default_tier": 0,
            "route_key": key[1],
            "route_name": str(record.get("route_name", route_name(actions))),
            **self.reward_evaluator.metadata(reward_result),
            "operator_pkg": str(record["operator_pkg"]),
            "operator_version": int(record["operator_version"]),
            "model_config_version": str(record["model_config_version"]),
            "collection_policy": "dqn_dev_eval",
            "policy_logprob": None,
            "policy_logprob_source": "not_scored",
            "collection_logprob": None,
            "trajectory_idx": 0,
            "trajectory_outcome": "success" if int(record["exec_res"]) == 1 else "failed",
            "source": str(record.get("source", "dqn_policy_materialized")),
            "run_name": str(record.get("run_name", "")),
            "checkpoint_sha256": str(record.get("checkpoint_sha256", "")),
            "db_id": str(record.get("db_id", "")),
            "final_sql": str(record.get("final_sql", "") or ""),
            "error": str(record.get("error", "") or ""),
        }
        stored = TrajectoryRecord(
            question_id=key[0],
            actions=actions,
            reward=float(reward_result.reward),
            exec_res=int(record["exec_res"]),
            total_cost=float(record["total_cost"]),
            stage_costs=list(record["stage_costs"]),
            decision_stage_costs=list(record["decision_stage_costs"]),
            fixed_stage_costs=list(record["fixed_stage_costs"]),
            total_latency_ms=float(record["total_latency_ms"]),
            stage_latencies_ms=list(record["stage_latencies_ms"]),
            decision_stage_latencies_ms=list(record["decision_stage_latencies_ms"]),
            fixed_stage_latencies_ms=list(record["fixed_stage_latencies_ms"]),
            metadata=metadata,
        )
        self.buffer.upsert_by_question_route(stored)
        self.buffer.save(str(self.path))
        self._index[key] = stored


def _validated_cached(
    record: Mapping[str, Any] | None,
    *,
    qid: int,
    db_id: str,
    actions: list[int],
) -> dict[str, Any] | None:
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValueError("cached route record must be a mapping")
    if (
        record.get("qid") != qid
        or record.get("db_id") != db_id
        or list(record.get("actions", ())) != actions
        or record.get("route_key") != route_key(actions)
        or type(record.get("exec_res")) is not int
        or record["exec_res"] not in (0, 1)
    ):
        raise ValueError("dev route cache identity or outcome mismatch")
    stage_costs = _cost_vector(record.get("stage_costs"), NUM_PIPELINE_STAGES, "stage_costs")
    decision_costs = _cost_vector(record.get("decision_stage_costs"), NUM_RL_STAGES, "decision_stage_costs")
    fixed_costs = _cost_vector(record.get("fixed_stage_costs"), NUM_PIPELINE_STAGES, "fixed_stage_costs")
    stage_latencies = _cost_vector(record.get("stage_latencies_ms"), NUM_PIPELINE_STAGES, "stage_latencies_ms")
    decision_latencies = _cost_vector(record.get("decision_stage_latencies_ms"), NUM_RL_STAGES, "decision_stage_latencies_ms")
    fixed_latencies = _cost_vector(record.get("fixed_stage_latencies_ms"), NUM_PIPELINE_STAGES, "fixed_stage_latencies_ms")
    total_cost = _finite_nonnegative(record["total_cost"], "cached total cost")
    if not math.isclose(total_cost, math.fsum(stage_costs), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("dev route cache cost mismatch")
    total_latency = _finite_nonnegative(record["total_latency_ms"], "cached total latency")
    if not math.isclose(total_latency, math.fsum(stage_latencies), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("dev route cache latency mismatch")
    if decision_costs != [stage_costs[i] for i in RL_DECISION_TO_PIPELINE_STAGE_IDXS]:
        raise ValueError("dev route cache decision costs mismatch")
    if decision_latencies != [stage_latencies[i] for i in RL_DECISION_TO_PIPELINE_STAGE_IDXS]:
        raise ValueError("dev route cache decision latencies mismatch")
    expected_fixed_costs = [stage_costs[i] if i in FIXED_PIPELINE_STAGE_IDXS else 0.0 for i in range(NUM_PIPELINE_STAGES)]
    expected_fixed_latencies = [stage_latencies[i] if i in FIXED_PIPELINE_STAGE_IDXS else 0.0 for i in range(NUM_PIPELINE_STAGES)]
    if fixed_costs != expected_fixed_costs or fixed_latencies != expected_fixed_latencies:
        raise ValueError("dev route cache fixed-stage vectors mismatch")
    if not isinstance(record.get("final_sql"), str):
        raise ValueError("dev route cache final_sql must be a string")
    result = dict(record)
    result.update(
        stage_costs=stage_costs,
        decision_stage_costs=decision_costs,
        fixed_stage_costs=fixed_costs,
        stage_latencies_ms=stage_latencies,
        decision_stage_latencies_ms=decision_latencies,
        fixed_stage_latencies_ms=fixed_latencies,
        total_cost=total_cost,
        total_latency_ms=total_latency,
    )
    return result


def _prediction(
    *,
    run_name: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    data_sha256: str,
    operator_pkg: str,
    operator_version: int,
    model_config_version: str,
    split: str,
    qid: int,
    db_id: str,
    actions: list[int],
    source: Mapping[str, Any],
    route_hit: bool,
    cache_hits: list[bool],
    cache_accessed: list[bool],
) -> dict[str, Any]:
    hits = sum(hit and accessed for hit, accessed in zip(cache_hits, cache_accessed))
    misses = sum(not hit and accessed for hit, accessed in zip(cache_hits, cache_accessed))
    return {
        "run_name": run_name,
        "checkpoint_id": run_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "data_sha256": data_sha256,
        "operator_pkg": operator_pkg,
        "operator_version": operator_version,
        "model_config_version": model_config_version,
        "split": split,
        "qid": qid,
        "db_id": db_id,
        "actions": actions,
        "route_key": route_key(actions),
        "route_name": route_name(actions),
        "status": "ok",
        "exec_res": int(source["exec_res"]),
        "EX": int(source["exec_res"]),
        "total_cost": float(source["total_cost"]),
        "controllable_cost": float(sum(source["decision_stage_costs"])),
        "stage_costs": list(source["stage_costs"]),
        "decision_stage_costs": list(source["decision_stage_costs"]),
        "fixed_stage_costs": list(source["fixed_stage_costs"]),
        "total_latency_ms": float(source["total_latency_ms"]),
        "stage_latencies_ms": list(source["stage_latencies_ms"]),
        "decision_stage_latencies_ms": list(source["decision_stage_latencies_ms"]),
        "fixed_stage_latencies_ms": list(source["fixed_stage_latencies_ms"]),
        "route_result_hit": route_hit,
        "route_result_cache": "hit" if route_hit else "miss",
        "transition_cache_hits": cache_hits,
        "transition_cache_accessed": cache_accessed,
        "transition_cache": {"hits": int(hits), "misses": int(misses)},
        "final_sql": str(source.get("final_sql", "") or ""),
        "error": str(source.get("error", "") or ""),
        "system_error": "",
    }


def evaluate_query(
    *,
    item: Mapping[str, Any],
    run_name: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    data_sha256: str,
    split: str,
    route_cache: Any,
    env: Any,
    transition_cache: Any,
    state_encoder: Any,
    policy: Any,
    selection_config: Any,
    operator_pkg: str,
    operator_version: int,
    model_config_version: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if selection_config is not None:
        raise ValueError("State V26 does not use an action-selection override")
    qid = int(item["question_id"])
    db_id = str(item.get("db_id", "") or "")
    payload = StatePayload(
        question_id=qid,
        db_id=db_id,
        question=str(item.get("question", "") or ""),
        evidence=str(item.get("evidence", "") or ""),
        gold_sql=str(item.get("SQL", "") or ""),
        cost_tracker=CostTracker(),
    )
    actions: list[int] = []
    cache_hits = [False] * NUM_PIPELINE_STAGES
    cache_accessed = [False] * NUM_PIPELINE_STAGES
    stage_costs = [0.0] * NUM_PIPELINE_STAGES
    fixed_costs = [0.0] * NUM_PIPELINE_STAGES
    stage_latencies = [0.0] * NUM_PIPELINE_STAGES
    fixed_latencies = [0.0] * NUM_PIPELINE_STAGES
    decision_costs: list[float] = []
    decision_latencies: list[float] = []
    action_masks: list[list[float]] = []
    total_cost = 0.0
    total_latency = 0.0
    manager = env._get_db_manager(db_id)
    question_embedding = state_encoder.encode_question(payload.question or "", payload.evidence or "")

    for decision_idx in range(NUM_RL_STAGES):
        budget = max(0.0, float(getattr(env, "total_budget", 10.0)) - total_cost)
        mask = _action_mask(env.get_action_mask(decision_idx, payload, budget), decision_idx)
        action_masks.append(mask.tolist())
        state = state_encoder.encode(
            payload,
            decision_idx,
            budget,
            actions[:],
            decision_costs[:],
            question_embedding,
            mask,
        )
        action = int(policy.select_action(state, mask))
        if mask[action] != 1.0:
            raise ValueError("State V26 selected a forbidden action")
        actions.append(action)

        physical = decision_to_pipeline_stage_idx(decision_idx)
        cache_accessed[physical] = action != 3
        payload, raw_cost, hit = env.run_physical_stage(
            stage_idx=physical,
            tier=action,
            payload=payload,
            cache=transition_cache,
            question_id=qid,
            db_manager=manager,
            operator_version=operator_version,
            cache_only=False,
        )
        cost = _finite_nonnegative(raw_cost, "decision stage cost")
        latency = _finite_nonnegative(getattr(env, "_last_step_latency_ms", 0.0), "decision stage latency")
        if cache_accessed[physical]:
            cache_hits[physical] = bool(hit)
        stage_costs[physical] = cost
        stage_latencies[physical] = latency
        decision_costs.append(cost)
        decision_latencies.append(latency)
        total_cost += cost
        total_latency += latency

        payload, fixed_results = env.run_fixed_stages_after_decision(
            decision_idx=decision_idx,
            payload=payload,
            cache=transition_cache,
            question_id=qid,
            db_manager=manager,
            operator_version=operator_version,
            cache_only=False,
        )
        for fixed_idx, raw_fixed_cost, fixed_hit, raw_fixed_latency in fixed_results:
            fixed_idx = int(fixed_idx)
            if not 0 <= fixed_idx < NUM_PIPELINE_STAGES:
                raise ValueError("fixed stage index is out of range")
            fixed_cost = _finite_nonnegative(raw_fixed_cost, "fixed stage cost")
            fixed_latency = _finite_nonnegative(raw_fixed_latency, "fixed stage latency")
            cache_accessed[fixed_idx] = True
            cache_hits[fixed_idx] = bool(fixed_hit)
            stage_costs[fixed_idx] = fixed_cost
            fixed_costs[fixed_idx] = fixed_cost
            stage_latencies[fixed_idx] = fixed_latency
            fixed_latencies[fixed_idx] = fixed_latency
            total_cost += fixed_cost
            total_latency += fixed_latency

    key = route_key(actions)
    cached = route_cache.get(
        split=split,
        qid=qid,
        db_id=db_id,
        route_key=key,
        operator_pkg=operator_pkg,
        operator_version=operator_version,
        model_config_version=model_config_version,
    )
    source = _validated_cached(cached, qid=qid, db_id=db_id, actions=actions)
    route_hit = source is not None
    created = None
    if source is None:
        if manager is None:
            raise ValueError("database manager is unavailable")
        try:
            outcome = env.step_s9(payload, manager)
        except Exception as exc:
            if isinstance(exc, ApiQuotaExhaustedError) or retryable_infrastructure_identity(exc):
                raise
            if type(exc) not in (sqlite3.OperationalError, sqlite3.ProgrammingError, ValueError, RuntimeError):
                raise
            outcome = {"exec_res": 0, "exec_err": str(exc)}
        if not isinstance(outcome, Mapping):
            raise ValueError("terminal evaluation must return a mapping")
        exec_res = outcome.get("exec_res")
        if type(exec_res) is not int or exec_res not in (0, 1):
            raise ValueError("terminal exec_res must be 0 or 1")
        error = outcome.get("exec_err")
        if not isinstance(error, str):
            raise ValueError("terminal exec_err must be a string")
        source = {
            "qid": qid,
            "db_id": db_id,
            "actions": actions,
            "route_key": key,
            "route_name": route_name(actions),
            "operator_pkg": operator_pkg,
            "operator_version": operator_version,
            "model_config_version": model_config_version,
            "exec_res": exec_res,
            "total_cost": total_cost,
            "stage_costs": stage_costs,
            "decision_stage_costs": decision_costs,
            "fixed_stage_costs": fixed_costs,
            "total_latency_ms": total_latency,
            "stage_latencies_ms": stage_latencies,
            "decision_stage_latencies_ms": decision_latencies,
            "fixed_stage_latencies_ms": fixed_latencies,
            "final_sql": str(getattr(payload, "final_sql", "") or ""),
            "error": error,
            "transition_cache_hits": cache_hits,
            "action_masks": action_masks,
            "source": "dqn_policy_materialized",
            "run_name": run_name,
            "checkpoint_sha256": checkpoint_sha256,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        route_cache.append(source)
        created = source

    prediction = _prediction(
        run_name=run_name,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        data_sha256=data_sha256,
        operator_pkg=operator_pkg,
        operator_version=operator_version,
        model_config_version=model_config_version,
        split=split,
        qid=qid,
        db_id=db_id,
        actions=actions,
        source=source,
        route_hit=route_hit,
        cache_hits=cache_hits,
        cache_accessed=cache_accessed,
    )
    return prediction, created
