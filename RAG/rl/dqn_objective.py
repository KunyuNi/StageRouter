"""Scalar DQN objective and legacy prepared-replay label helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def bundled_decision_costs(
    physical_stage_costs: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(physical_stage_costs) != 7:
        raise ValueError("expected seven physical stage costs")
    values = tuple(float(value) for value in physical_stage_costs)
    return (values[0] + values[1], values[2] + values[3], values[4], values[5])


def scalar_rewards(
    f1: float,
    costs: Sequence[float],
    *,
    c_ref: float,
    lambda_cost: float,
) -> tuple[float, float, float, float]:
    if len(costs) != 4:
        raise ValueError("scalar rewards require four decision costs")
    if c_ref <= 0.0:
        raise ValueError("c_ref must be positive")
    rewards = [-float(lambda_cost) * float(cost) / float(c_ref) for cost in costs]
    rewards[-1] += float(f1)
    return tuple(rewards)  # type: ignore[return-value]


def dual_rewards(
    f1: float,
    costs: Sequence[float],
    *,
    c_ref: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    if len(costs) != 4:
        raise ValueError("dual rewards require four decision costs")
    if c_ref <= 0.0:
        raise ValueError("c_ref must be positive")
    quality = (0.0, 0.0, 0.0, float(f1))
    positive_cost = tuple(float(cost) / float(c_ref) for cost in costs)
    return quality, positive_cost  # type: ignore[return-value]


def _suffix_returns(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("MC returns require four rewards")
    result = [0.0] * 4
    running = 0.0
    for index in range(3, -1, -1):
        running += float(values[index])
        result[index] = running
    return tuple(result)  # type: ignore[return-value]


def scalar_mc_returns(
    f1: float,
    costs: Sequence[float],
    *,
    c_ref: float,
    lambda_cost: float,
) -> tuple[float, float, float, float]:
    return _suffix_returns(
        scalar_rewards(f1, costs, c_ref=c_ref, lambda_cost=lambda_cost)
    )


def dual_mc_returns(
    f1: float,
    costs: Sequence[float],
    *,
    c_ref: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    quality, cost = dual_rewards(f1, costs, c_ref=c_ref)
    return _suffix_returns(quality), _suffix_returns(cost)


_TIE_PRIORITY = (3, 2, 1, 0)  # SKIP, T3, T2, T1


def legal_argmax(
    scores: torch.Tensor,
    legal_mask: torch.Tensor,
    empirical_cost: torch.Tensor | None = None,
) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[-1] != 4:
        raise ValueError("scores must have shape [batch,4]")
    if legal_mask.shape != scores.shape:
        raise ValueError("legal_mask must match scores")
    legal_mask = legal_mask.to(dtype=torch.bool, device=scores.device)
    if not torch.all(legal_mask.any(dim=1)):
        raise ValueError("every row must contain a legal action")
    if empirical_cost is None:
        empirical_cost = torch.zeros_like(scores)
    if empirical_cost.shape != scores.shape:
        raise ValueError("empirical_cost must match scores")
    selected: list[int] = []
    for row in range(scores.shape[0]):
        legal = [action for action in range(4) if bool(legal_mask[row, action])]
        best_score = max(float(scores[row, action]) for action in legal)
        tied = [
            action
            for action in legal
            if float(scores[row, action]) == best_score
        ]
        best_cost = min(float(empirical_cost[row, action]) for action in tied)
        tied = [
            action
            for action in tied
            if float(empirical_cost[row, action]) == best_cost
        ]
        selected.append(next(action for action in _TIE_PRIORITY if action in tied))
    return torch.tensor(selected, dtype=torch.long, device=scores.device)


def scalar_double_dqn_targets(
    *,
    rewards: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    legal_mask: torch.Tensor,
    terminal: torch.Tensor,
    empirical_cost: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    actions = legal_argmax(online_next_q, legal_mask, empirical_cost)
    bootstrap = target_next_q.gather(1, actions.unsqueeze(1)).squeeze(1)
    return rewards + (~terminal.to(torch.bool)).to(rewards.dtype) * float(gamma) * bootstrap


__all__ = [
    "bundled_decision_costs",
    "dual_mc_returns",
    "dual_rewards",
    "legal_argmax",
    "scalar_double_dqn_targets",
    "scalar_mc_returns",
    "scalar_rewards",
]
