from __future__ import annotations

import pytest

from RAG.rl.counterfactual_route_ladder import (
    ANCHOR_PROFILE,
    SKIP,
    build_random_route_profile,
    enumerate_legal_routes,
    get_route_profile,
)
from RAG.pipeline.value_pruned_routes import routes_from_cost_shares


def test_anchor_and_cost_pruned_templates_are_legal():
    profile, high_cost = routes_from_cost_shares((0.4, 0.05, 0.4, 0.15))
    routes = profile.routes
    assert high_cost == (0, 2, 3)
    assert tuple(route.rank for route in routes) == tuple(range(len(routes)))
    assert tuple((route.rank, route.name, route.actions) for route in routes[:3]) == (
        (0, "all_t3", (2, 2, 2, 2)),
        (1, "all_t2", (1, 1, 1, 1)),
        (2, "all_t1", (0, 0, 0, 0)),
    )
    assert len({route.name for route in routes}) == len(routes)
    assert len({route.actions for route in routes}) == len(routes)


def test_cost_pruning_limits_model_substitutions_but_keeps_skip_ablations():
    profile, _ = routes_from_cost_shares((0.4, 0.05, 0.4, 0.15))
    assert not any(route.name.startswith("isolate_") and "_s2_" in route.name for route in profile.routes)
    assert any(route.name.startswith("ablate_") and "_s2" in route.name for route in profile.routes)
    assert any(route.name.startswith("couple_") for route in profile.routes)


def test_answer_never_skips_and_all_profile_aliases_resolve():
    profile, _ = routes_from_cost_shares((0.4, 0.05, 0.4, 0.15))
    assert all(route.actions[3] != 3 for route in profile.routes)
    assert get_route_profile("anchors") is ANCHOR_PROFILE
    assert get_route_profile("fixed_anchors_v1") is ANCHOR_PROFILE
    with pytest.raises(ValueError, match="Unknown route profile"):
        get_route_profile("obsolete")


def test_random_route_profile_is_deterministic_legal_and_unique():
    first = build_random_route_profile(count=36, seed=1145)
    second = build_random_route_profile(count=36, seed=1145)

    assert first == second
    assert len(enumerate_legal_routes()) == 192
    assert len(first.routes) == 36
    assert len({route.name for route in first.routes}) == 36
    assert len({route.actions for route in first.routes}) == 36
    assert all(route.actions[3] != SKIP for route in first.routes)

    assert all(route.name.startswith("random36_") for route in first.routes)
