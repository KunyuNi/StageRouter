"""Legal RAG homogeneous anchors and random-route ablation templates."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


T1 = 0
T2 = 1
T3 = 2
SKIP = 3


@dataclass(frozen=True)
class RouteTemplate:
    rank: int
    name: str
    actions: tuple[int, int, int, int]


@dataclass(frozen=True)
class RouteProfile:
    name: str
    routes: tuple[RouteTemplate, ...]

    @property
    def route_map(self) -> dict[str, tuple[int, int, int, int]]:
        return {route.name: route.actions for route in self.routes}

    @property
    def route_order(self) -> tuple[str, ...]:
        return tuple(route.name for route in self.routes)

    @property
    def sha256(self) -> str:
        rows = [
            {"rank": route.rank, "name": route.name, "actions": route.actions}
            for route in self.routes
        ]
        payload = json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


_ANCHOR_ROUTES = (
    RouteTemplate(0, "all_t3", (T3, T3, T3, T3)),
    RouteTemplate(1, "all_t2", (T2, T2, T2, T2)),
    RouteTemplate(2, "all_t1", (T1, T1, T1, T1)),
)


ANCHOR_PROFILE = RouteProfile("fixed_anchors_v1", _ANCHOR_ROUTES)


def enumerate_legal_routes() -> tuple[tuple[int, int, int, int], ...]:
    """Return the complete legal four-decision RAG route universe."""

    return tuple(
        (query, second_hop, rerank, answer)
        for query in (T1, T2, T3, SKIP)
        for second_hop in (T1, T2, T3, SKIP)
        for rerank in (T1, T2, T3, SKIP)
        for answer in (T1, T2, T3)
    )


def build_random_route_profile(
    *, count: int = 36, seed: int = 1145
) -> RouteProfile:
    """Build one globally shared deterministic random route profile."""

    legal_routes = enumerate_legal_routes()
    route_count = int(count)
    if route_count <= 0 or route_count > len(legal_routes):
        raise ValueError(
            f"count must be in 1..{len(legal_routes)}, got {route_count}"
        )
    selected = sorted(random.Random(int(seed)).sample(legal_routes, route_count))
    routes = tuple(
        RouteTemplate(
            rank=rank,
            name=f"random{route_count}_{rank:02d}",
            actions=actions,
        )
        for rank, actions in enumerate(selected)
    )
    return RouteProfile(f"rag_random{route_count}_seed{int(seed)}_v1", routes)


def get_route_profile(name: str) -> RouteProfile:
    profiles = {
        "anchors": ANCHOR_PROFILE,
        "fixed_anchors_v1": ANCHOR_PROFILE,
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"Unknown route profile: {name}") from exc


__all__ = [
    "ANCHOR_PROFILE",
    "build_random_route_profile",
    "enumerate_legal_routes",
    "RouteProfile",
    "RouteTemplate",
    "SKIP",
    "T1",
    "T2",
    "T3",
    "get_route_profile",
]
