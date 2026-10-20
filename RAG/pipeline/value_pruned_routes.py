"""Construct RAG diagnostic routes from homogeneous-anchor cost shares."""

from __future__ import annotations

import math
from itertools import combinations, product

from RAG.rl.counterfactual_route_ladder import ANCHOR_PROFILE, RouteProfile, RouteTemplate, SKIP


MODEL_ACTIONS = (0, 1, 2)
STAGE_COUNT = 4
SKIPPABLE_STAGES = (0, 1, 2)
PLAN_SCHEMA = "rag_observed_route_plan_v3"


def routes_from_cost_shares(shares: tuple[float, ...], threshold: float = 0.10) -> tuple[RouteProfile, tuple[int, ...]]:
    if len(shares) != STAGE_COUNT or not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("invalid cost shares or threshold")
    if any(not math.isfinite(value) or value < 0 for value in shares):
        raise ValueError("cost shares must be finite and nonnegative")
    high_cost = tuple(stage for stage, share in enumerate(shares) if share > threshold)
    routes: list[RouteTemplate] = []
    seen: set[tuple[int, ...]] = set()

    def insert(name: str, actions: tuple[int, ...]) -> None:
        if len(actions) != STAGE_COUNT or any(
            action not in MODEL_ACTIONS and not (action == SKIP and stage in SKIPPABLE_STAGES)
            for stage, action in enumerate(actions)
        ):
            raise ValueError("generated route contains an illegal action")
        if actions not in seen:
            seen.add(actions)
            routes.append(RouteTemplate(len(routes), name, actions))

    for anchor in ANCHOR_PROFILE.routes:
        insert(anchor.name, anchor.actions)
    for anchor in ANCHOR_PROFILE.routes:
        model = anchor.actions[0]
        for stage in high_cost:
            for alternative in MODEL_ACTIONS:
                if alternative != model:
                    actions = list(anchor.actions)
                    actions[stage] = alternative
                    insert(f"isolate_t{model + 1}_s{stage + 1}_t{alternative + 1}", tuple(actions))
        for stage in SKIPPABLE_STAGES:
            actions = list(anchor.actions)
            actions[stage] = SKIP
            insert(f"ablate_t{model + 1}_s{stage + 1}", tuple(actions))
        for first_stage, second_stage in combinations(high_cost, 2):
            for first_model, second_model in product(MODEL_ACTIONS, repeat=2):
                actions = list(anchor.actions)
                actions[first_stage], actions[second_stage] = first_model, second_model
                insert(
                    f"couple_t{model + 1}_s{first_stage + 1}_s{second_stage + 1}"
                    f"_t{first_model + 1}_t{second_model + 1}",
                    tuple(actions),
                )
    return RouteProfile("rag_cost_pruned_route_library_v3", tuple(routes)), high_cost
