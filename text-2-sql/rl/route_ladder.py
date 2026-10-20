"""Shared route representation and legality helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class RouteTemplate:
    name: str
    actions: List[int]
    rank: int


def route_key(actions: Sequence[int]) -> str:
    return "-".join(str(int(action)) for action in actions)


def route_name(actions: Sequence[int]) -> str:
    return route_key(actions)


def valid_route_for_stage_masks(actions: Sequence[int], masks: Iterable[Sequence[float]]) -> bool:
    masks_list = list(masks)
    if len(actions) != len(masks_list):
        return False
    for action, mask in zip(actions, masks_list):
        action_i = int(action)
        if action_i < 0 or action_i >= len(mask):
            return False
        if float(mask[action_i]) <= 0.0:
            return False
    return True
