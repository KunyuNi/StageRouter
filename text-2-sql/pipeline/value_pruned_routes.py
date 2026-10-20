"""Construct legal diagnostic routes from homogeneous-anchor cost traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Mapping, Sequence

from rl.route_ladder import RouteTemplate


@dataclass(frozen=True)
class RoutePruningResult:
    routes: tuple[RouteTemplate, ...]
    mean_cost_shares: tuple[float, ...]
    high_cost_stages: tuple[int, ...]
    anchor_requests: int


def _anchor_profile(
    traces: Sequence[Mapping[str, Any]], model_count: int, stage_count: int
) -> tuple[tuple[float, ...], int]:
    shares: list[list[float]] = [[] for _ in range(stage_count)]
    seen: set[tuple[int, int]] = set()
    requests: set[int] = set()
    for record in traces:
        qid = int(record["question_id"])
        actions = tuple(int(action) for action in record["actions"])
        if len(actions) != stage_count or len(set(actions)) != 1 or not 0 <= actions[0] < model_count:
            raise ValueError("cost profiling requires one homogeneous route per model and request")
        key = (qid, actions[0])
        if key in seen:
            raise ValueError("duplicate homogeneous-anchor trace")
        seen.add(key)
        requests.add(qid)
        costs = tuple(float(cost) for cost in record["decision_stage_costs"])
        total = float(record["total_cost"])
        if (
            len(costs) != stage_count
            or not math.isfinite(total)
            or total <= 0
            or any(not math.isfinite(cost) or cost < 0 for cost in costs)
            or sum(costs) > total + 1e-9
        ):
            raise ValueError("anchor costs must be finite, non-negative, and consistent with total cost")
        for stage, cost in enumerate(costs):
            shares[stage].append(cost / total)
    if not requests or len(seen) != len(requests) * model_count:
        raise ValueError("every selected request needs all homogeneous anchors")
    return tuple(sum(values) / len(values) for values in shares), len(requests)


def build_value_pruned_routes(
    anchor_traces: Sequence[Mapping[str, Any]],
    *,
    model_count: int,
    stage_count: int,
    skippable_stages: Sequence[int],
    cost_share_threshold: float = 0.10,
) -> RoutePruningResult:
    """Apply the paper's anchors, isolations, ablations, and high-cost pairs.

    Stage indices are zero-based positions among routable stages. The SKIP
    action is encoded as ``model_count``. Cost profiling uses executed anchors;
    constructing the remaining routes makes no LLM calls.
    """
    if model_count < 2 or stage_count < 1:
        raise ValueError("at least two models and one routable stage are required")
    if not math.isfinite(cost_share_threshold) or not 0 <= cost_share_threshold < 1:
        raise ValueError("cost-share threshold must be in [0, 1)")
    optional = tuple(sorted(set(int(stage) for stage in skippable_stages)))
    if any(stage < 0 or stage >= stage_count for stage in optional):
        raise ValueError("skippable stage index is outside the workflow")
    shares, request_count = _anchor_profile(anchor_traces, model_count, stage_count)
    high_cost = tuple(stage for stage, share in enumerate(shares) if share > cost_share_threshold)
    routes: list[RouteTemplate] = []
    seen: set[tuple[int, ...]] = set()

    def insert(name: str, actions: Sequence[int]) -> None:
        route = tuple(actions)
        if any(action not in range(model_count) and not (action == model_count and stage in optional)
               for stage, action in enumerate(route)):
            raise ValueError("generated route contains an illegal stage action")
        if route not in seen:
            seen.add(route)
            routes.append(RouteTemplate(name=name, actions=list(route), rank=len(routes)))

    for model in range(model_count):
        insert(f"anchor_t{model + 1}", [model] * stage_count)
    for model in range(model_count):
        anchor = [model] * stage_count
        for stage in high_cost:
            for alternative in range(model_count):
                if alternative != model:
                    route = anchor.copy()
                    route[stage] = alternative
                    insert(f"isolate_t{model + 1}_s{stage + 1}_t{alternative + 1}", route)
        for stage in optional:
            route = anchor.copy()
            route[stage] = model_count
            insert(f"ablate_t{model + 1}_s{stage + 1}", route)
        for upstream, downstream in combinations(high_cost, 2):
            for first, second in product(range(model_count), repeat=2):
                route = anchor.copy()
                route[upstream], route[downstream] = first, second
                insert(
                    f"couple_t{model + 1}_s{upstream + 1}_s{downstream + 1}_t{first + 1}_t{second + 1}",
                    route,
                )
    return RoutePruningResult(tuple(routes), shares, high_cost, request_count)
