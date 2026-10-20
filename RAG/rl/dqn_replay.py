"""Offline transition records and deterministic qid/route scheduling."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .decision_session import RAGDecisionSession
from .dqn_objective import dual_mc_returns, dual_rewards, scalar_mc_returns, scalar_rewards
from .state_v1_encoder import EncodedState, StateV1Encoder


@dataclass(frozen=True)
class OfflineTransition:
    qid: int
    route_key: str
    trajectory_identity: str
    decision_idx: int
    action: int
    state: EncodedState
    next_state: EncodedState | None
    reward_scalar: float
    reward_f1: float
    reward_cost: float
    mc_scalar: float
    mc_f1: float
    mc_cost: float
    terminal: bool
    final_f1: float
    remaining_cost: float
    actions: tuple[int, int, int, int]


def reconstruct_offline_transitions(
    *,
    item: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    env: Any,
    cache: Any,
    encoder: StateV1Encoder,
    operator_version: int = 1,
    c_ref: float = 0.00333879,
    lambda_cost: float = 0.01,
) -> tuple[OfflineTransition, ...]:
    actions = tuple(int(value) for value in trajectory["actions"])
    if len(actions) != 4:
        raise ValueError("complete RAG trajectory must have four actions")
    if not math.isfinite(float(trajectory["f1_score"])):
        raise ValueError("trajectory F1 must be finite")
    session = RAGDecisionSession.start(
        item=item,
        env=env,
        cache=cache,
        operator_version=operator_version,
        cache_only=True,
    )
    states: list[EncodedState] = []
    next_states: list[EncodedState | None] = []
    for decision_idx, action in enumerate(actions):
        states.append(encoder.encode(session.observation()))
        session.apply_action(action)
        next_states.append(
            encoder.encode(session.observation()) if decision_idx < 3 else None
        )
    result = session.finalize()
    final_f1 = float(result["f1_score"])
    stored_f1 = float(trajectory["f1_score"])
    if abs(final_f1 - stored_f1) > 1e-12:
        raise ValueError("reconstructed F1 does not match stored raw F1")
    costs = tuple(float(value) for value in result["bundled_decision_costs"])
    stored_total = float(trajectory.get("total_cost", sum(costs)))
    if abs(sum(costs) - stored_total) > 1e-9:
        raise ValueError("reconstructed bundled costs do not match route total")
    if not all(math.isfinite(value) and value >= 0.0 for value in costs):
        raise ValueError("trajectory decision costs must be finite and nonnegative")
    scalar = scalar_rewards(final_f1, costs, c_ref=c_ref, lambda_cost=lambda_cost)
    quality, cost = dual_rewards(final_f1, costs, c_ref=c_ref)
    mc_scalar = scalar_mc_returns(
        final_f1, costs, c_ref=c_ref, lambda_cost=lambda_cost
    )
    mc_quality, mc_cost = dual_mc_returns(final_f1, costs, c_ref=c_ref)
    route = str(result["route_key"])
    qid = int(result["question_id"])
    trajectory_identity = hashlib.sha256(
        (
            f"{trajectory.get('split', 'train')}:{qid}:{route}:"
            f"{operator_version}:{trajectory.get('workflow_identity', '')}"
        ).encode("utf-8")
    ).hexdigest()
    transitions = []
    for index in range(4):
        transitions.append(
            OfflineTransition(
                qid=qid,
                route_key=route,
                trajectory_identity=trajectory_identity,
                decision_idx=index,
                action=actions[index],
                state=states[index],
                next_state=next_states[index],
                reward_scalar=scalar[index],
                reward_f1=quality[index],
                reward_cost=cost[index],
                mc_scalar=mc_scalar[index],
                mc_f1=mc_quality[index],
                mc_cost=mc_cost[index],
                terminal=index == 3,
                final_f1=final_f1,
                remaining_cost=sum(costs[index:]),
                actions=actions,  # type: ignore[arg-type]
            )
        )
    return tuple(transitions)


def _jsonable_random_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_random_state(item) for item in value]
    return value


def _tuple_random_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_random_state(item) for item in value)
    return value


class DeterministicRouteSchedule:
    def __init__(
        self,
        routes_by_qid: Mapping[int, Sequence[str]],
        *,
        seed: int,
        qids_per_step: int = 16,
        routes_per_qid: int = 4,
    ) -> None:
        self.routes_by_qid = {
            int(qid): tuple(str(route) for route in routes)
            for qid, routes in routes_by_qid.items()
        }
        self.seed = int(seed)
        self.qids_per_step = int(qids_per_step)
        self.routes_per_qid = int(routes_per_qid)
        if len(self.routes_by_qid) < self.qids_per_step:
            raise ValueError("not enough distinct qids for one replay step")
        if any(len(set(routes)) < self.routes_per_qid for routes in self.routes_by_qid.values()):
            raise ValueError("every qid must provide enough distinct routes")
        self._source_identity = self._compute_source_identity()
        self._rng = random.Random(self.seed)
        self.qid_permutation = sorted(self.routes_by_qid)
        self._rng.shuffle(self.qid_permutation)
        self.qid_cursor = 0
        self.qid_reshuffles = 0
        self.route_permutations: dict[int, list[str]] = {}
        self.route_cursors: dict[int, int] = {}
        self.route_reshuffles: dict[int, int] = {}
        for qid, routes in self.routes_by_qid.items():
            permutation = list(routes)
            self._rng.shuffle(permutation)
            self.route_permutations[qid] = permutation
            self.route_cursors[qid] = 0
            self.route_reshuffles[qid] = 0

    def _compute_source_identity(self) -> str:
        canonical = {
            "routes_by_qid": {
                str(qid): list(routes)
                for qid, routes in sorted(self.routes_by_qid.items())
            },
            "qids_per_step": self.qids_per_step,
            "routes_per_qid": self.routes_per_qid,
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _reshuffle_qids(self) -> None:
        self.qid_permutation = sorted(self.routes_by_qid)
        self._rng.shuffle(self.qid_permutation)
        self.qid_cursor = 0
        self.qid_reshuffles += 1

    def _next_qids(self) -> list[int]:
        chosen: list[int] = []
        chosen_set: set[int] = set()
        while len(chosen) < self.qids_per_step:
            if self.qid_cursor >= len(self.qid_permutation):
                self._reshuffle_qids()
            qid = self.qid_permutation[self.qid_cursor]
            self.qid_cursor += 1
            if qid in chosen_set:
                continue
            chosen.append(qid)
            chosen_set.add(qid)
        return chosen

    def _reshuffle_routes(self, qid: int) -> None:
        permutation = list(self.routes_by_qid[qid])
        self._rng.shuffle(permutation)
        self.route_permutations[qid] = permutation
        self.route_cursors[qid] = 0
        self.route_reshuffles[qid] += 1

    def _next_routes(self, qid: int) -> list[str]:
        chosen: list[str] = []
        chosen_set: set[str] = set()
        while len(chosen) < self.routes_per_qid:
            cursor = self.route_cursors[qid]
            permutation = self.route_permutations[qid]
            if cursor >= len(permutation):
                self._reshuffle_routes(qid)
                continue
            route = permutation[cursor]
            self.route_cursors[qid] = cursor + 1
            if route in chosen_set:
                continue
            chosen.append(route)
            chosen_set.add(route)
        return chosen

    def next_batch(self) -> tuple[tuple[int, str], ...]:
        return tuple(
            (qid, route)
            for qid in self._next_qids()
            for route in self._next_routes(qid)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "source_identity": self._source_identity,
            "seed": self.seed,
            "qids_per_step": self.qids_per_step,
            "routes_per_qid": self.routes_per_qid,
            "qid_permutation": list(self.qid_permutation),
            "qid_cursor": self.qid_cursor,
            "qid_reshuffles": self.qid_reshuffles,
            "route_permutations": {
                str(qid): list(values)
                for qid, values in self.route_permutations.items()
            },
            "route_cursors": {
                str(qid): value for qid, value in self.route_cursors.items()
            },
            "route_reshuffles": {
                str(qid): value for qid, value in self.route_reshuffles.items()
            },
            "rng_state": _jsonable_random_state(self._rng.getstate()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if str(state.get("source_identity")) != self._source_identity:
            raise ValueError("replay schedule source identity mismatch")
        if int(state["qids_per_step"]) != self.qids_per_step or int(
            state["routes_per_qid"]
        ) != self.routes_per_qid:
            raise ValueError("replay schedule batch contract mismatch")
        self.seed = int(state["seed"])
        self.qid_permutation = [int(value) for value in state["qid_permutation"]]
        self.qid_cursor = int(state["qid_cursor"])
        self.qid_reshuffles = int(state["qid_reshuffles"])
        self.route_permutations = {
            int(qid): [str(value) for value in values]
            for qid, values in state["route_permutations"].items()
        }
        self.route_cursors = {
            int(qid): int(value) for qid, value in state["route_cursors"].items()
        }
        self.route_reshuffles = {
            int(qid): int(value)
            for qid, value in state["route_reshuffles"].items()
        }
        self._rng.setstate(_tuple_random_state(state["rng_state"]))

    @property
    def identity(self) -> str:
        canonical = json.dumps(
            self.state_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OfflineReplayDataset:
    """Complete four-transition trajectories addressed by schedule identity."""

    def __init__(
        self,
        trajectories: Mapping[tuple[int, str], Sequence[OfflineTransition]],
        schedule: DeterministicRouteSchedule,
    ) -> None:
        self.trajectories = {
            (int(qid), str(route)): tuple(values)
            for (qid, route), values in trajectories.items()
        }
        self.schedule = schedule
        for identity, values in self.trajectories.items():
            if len(values) != 4 or tuple(row.decision_idx for row in values) != (0, 1, 2, 3):
                raise ValueError(f"trajectory {identity} is not a complete four-step route")
        self.last_batch_identity = ""

    def next_batch(self) -> tuple[OfflineTransition, ...]:
        selected = self.schedule.next_batch()
        rows: list[OfflineTransition] = []
        for identity in selected:
            if identity not in self.trajectories:
                raise KeyError(f"scheduled trajectory is missing: {identity}")
            rows.extend(self.trajectories[identity])
        self.last_batch_identity = hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return tuple(rows)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.state_dict(),
            "last_batch_identity": self.last_batch_identity,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.schedule.load_state_dict(state["schedule"])
        self.last_batch_identity = str(state.get("last_batch_identity", ""))


__all__ = [
    "DeterministicRouteSchedule",
    "OfflineReplayDataset",
    "OfflineTransition",
    "reconstruct_offline_transitions",
]
