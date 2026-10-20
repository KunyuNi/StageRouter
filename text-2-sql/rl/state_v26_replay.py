"""Deterministic route schedule and replay batch types for State V26."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _qid(record: Any) -> int:
    if isinstance(record, Mapping):
        return int(record.get("question_id", record.get("qid")))
    if hasattr(record, "question_id"):
        return int(record.question_id)
    return int(record.qid)


@dataclass(frozen=True)
class FrozenCycleSchedule:
    route_indices: np.ndarray
    canonical_sha256: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.route_indices)
        if matrix.ndim != 2 or matrix.dtype != np.int32:
            raise ValueError("route schedule must be a two-dimensional int32 matrix")
        digest = hashlib.sha256(matrix.astype("<i4", copy=False).tobytes()).hexdigest()
        if digest != self.canonical_sha256:
            raise ValueError("route schedule canonical SHA mismatch")


@dataclass(frozen=True)
class StateV26ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    return_to_go: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    action_masks: np.ndarray
    next_action_masks: np.ndarray
    stage_idxs: np.ndarray


def build_cycle_schedule(
    records: Sequence[Any],
    *,
    steps: int = 7500,
    qids_per_step: int = 16,
    routes_per_qid: int = 4,
    seed: int = 1145,
) -> FrozenCycleSchedule:
    if steps <= 0 or qids_per_step <= 0 or routes_per_qid <= 0:
        raise ValueError("schedule dimensions must be positive")
    by_qid: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        by_qid.setdefault(_qid(record), []).append(index)
    if len(by_qid) < qids_per_step:
        raise ValueError("not enough unique qids for one schedule step")
    if any(len(indices) < routes_per_qid for indices in by_qid.values()):
        raise ValueError("every qid must have at least routes_per_qid routes")

    rng = np.random.Generator(np.random.PCG64(seed))
    qids = np.asarray(sorted(by_qid), dtype=np.int64)
    qid_queue: list[int] = []
    route_queues: dict[int, list[int]] = {qid: [] for qid in by_qid}
    matrix = np.empty(
        (steps, qids_per_step * routes_per_qid), dtype=np.int32
    )

    def refill_qids() -> None:
        qid_queue.extend(int(value) for value in rng.permutation(qids))

    def take_qids() -> list[int]:
        selected: list[int] = []
        selected_set: set[int] = set()
        while len(selected) < qids_per_step:
            if not qid_queue:
                refill_qids()
            qid = qid_queue.pop(0)
            if qid in selected_set:
                qid_queue.append(qid)
                continue
            selected.append(qid)
            selected_set.add(qid)
        return selected

    def take_routes(qid: int) -> list[int]:
        queue = route_queues[qid]
        chosen: list[int] = []
        while len(chosen) < routes_per_qid:
            if not queue:
                queue.extend(
                    int(value)
                    for value in rng.permutation(np.asarray(by_qid[qid]))
                )
            route = queue.pop(0)
            if route in chosen:
                queue.append(route)
                continue
            chosen.append(route)
        return chosen

    for step in range(steps):
        row = [route for qid in take_qids() for route in take_routes(qid)]
        matrix[step] = np.asarray(row, dtype=np.int32)
    digest = hashlib.sha256(matrix.astype("<i4", copy=False).tobytes()).hexdigest()
    return FrozenCycleSchedule(matrix, digest)


class StateV26ReplayBuilder:
    """Materialize the route indices fixed for a global training step."""

    def __init__(
        self,
        records: Sequence[Any],
        schedule: FrozenCycleSchedule,
        materialize_route: Callable[[Any], StateV26ReplayBatch],
    ) -> None:
        self.records = records
        self.schedule = schedule
        self.materialize_route = materialize_route

    def sample_batch(self, global_step: int) -> StateV26ReplayBatch:
        if not 0 <= global_step < len(self.schedule.route_indices):
            raise IndexError("global_step is outside the frozen replay schedule")
        batches = [
            self.materialize_route(self.records[int(index)])
            for index in self.schedule.route_indices[global_step]
        ]
        fields = StateV26ReplayBatch.__dataclass_fields__
        return StateV26ReplayBatch(
            **{
                field: np.concatenate(
                    [np.asarray(getattr(batch, field)) for batch in batches], axis=0
                )
                for field in fields
            }
        )


class CachedStateV26RouteMaterializer:
    """Replay one frozen trajectory and attach scalar rewards and MC returns."""

    def __init__(
        self,
        env: Any,
        cache: Any,
        encoder: Any,
        *,
        lambda_cost: float,
        c_ref: float,
    ) -> None:
        self.env = env
        self.cache = cache
        self.encoder = encoder
        self.lambda_cost = float(lambda_cost)
        self.c_ref = float(c_ref)

    def __call__(self, record: Any) -> StateV26ReplayBatch:
        from .state_v26_objective import return_to_go, scalar_transition_rewards

        qid = _qid(record)
        actions = tuple(int(value) for value in (
            record["actions"] if isinstance(record, Mapping) else record.actions
        ))
        final_ex = int(
            record["exec_res"] if isinstance(record, Mapping) else record.exec_res
        )
        costs = np.asarray(
            record["decision_stage_costs"]
            if isinstance(record, Mapping)
            else record.decision_stage_costs,
            dtype=np.float64,
        )
        fixed_costs = np.asarray(
            record["fixed_stage_costs"]
            if isinstance(record, Mapping)
            else record.fixed_stage_costs,
            dtype=np.float64,
        )
        total_cost = float(
            record["total_cost"] if isinstance(record, Mapping) else record.total_cost
        )
        if (
            costs.shape != (6,)
            or fixed_costs.shape != (8,)
            or not np.isfinite(costs).all()
            or not np.isfinite(fixed_costs).all()
            or np.any(costs < 0)
            or np.any(fixed_costs < 0)
            or not math.isfinite(total_cost)
            or not math.isclose(
                total_cost, float(costs.sum() + fixed_costs.sum()),
                rel_tol=1e-9, abs_tol=1e-9,
            )
        ):
            raise ValueError("replay stage costs do not match total route cost")
        if np.any(fixed_costs[[0, 3, 4, 5, 6, 7]] != 0):
            raise ValueError("fixed costs must belong to the intervening retrieval stages")
        costs = costs.copy()
        costs[0] += fixed_costs[1] + fixed_costs[2]
        replayed = self.env.rollout_with_cache(
            question_id=qid,
            action_sequence=actions,
            cache=self.cache,
            state_encoder=self.encoder,
            cache_only=True,
            evaluate_terminal=False,
        )
        if not isinstance(replayed, Mapping):
            raise RuntimeError(f"cache-only State V26 replay failed for qid={qid}")
        states = np.asarray(replayed["states"], dtype=np.float32)
        masks = np.asarray(replayed["action_masks"], dtype=bool)
        if states.shape != (7, 5256) or masks.shape != (6, 4):
            raise ValueError("State V26 replay has invalid state or mask shape")
        if not np.array_equal(masks[1], np.asarray([1, 1, 1, 0], dtype=bool)):
            raise ValueError("State V26 S4 mask must be exactly {T1,T2,T3}")
        rewards = scalar_transition_rewards(
            costs,
            final_ex=final_ex,
            lambda_cost=self.lambda_cost,
            c_ref=self.c_ref,
        )
        returns = return_to_go(rewards)
        next_masks = np.vstack(
            (masks[1:], np.asarray([[1, 0, 0, 0]], dtype=bool))
        )
        return StateV26ReplayBatch(
            states=states[:-1],
            actions=np.asarray(actions, dtype=np.int64),
            rewards=rewards.astype(np.float32),
            return_to_go=returns.astype(np.float32),
            next_states=states[1:],
            dones=np.asarray([0, 0, 0, 0, 0, 1], dtype=np.float32),
            action_masks=masks,
            next_action_masks=next_masks,
            stage_idxs=np.arange(6, dtype=np.int64),
        )
