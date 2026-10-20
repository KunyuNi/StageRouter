"""Scalar rewards and auxiliary objectives for State V26."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StateV26ObjectiveConfig:
    lambda_cost: float
    c_ref: float
    gamma: float = 1.0
    mc_weight: float = 0.3

    def validate(self) -> None:
        if not math.isfinite(self.lambda_cost) or self.lambda_cost <= 0:
            raise ValueError("lambda_cost must be positive and finite")
        if not math.isfinite(self.c_ref) or self.c_ref <= 0:
            raise ValueError("c_ref must be positive and finite")
        if self.gamma != 1.0:
            raise ValueError("gamma must equal one for undiscounted workflow return")
        if not math.isfinite(self.mc_weight) or self.mc_weight < 0:
            raise ValueError("mc_weight must be non-negative and finite")


def scalar_transition_rewards(
    stage_costs: np.ndarray,
    *,
    final_ex: int,
    lambda_cost: float,
    c_ref: float,
) -> np.ndarray:
    costs = np.asarray(stage_costs, dtype=np.float64)
    if (
        costs.shape != (6,)
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
        or final_ex not in (0, 1)
        or not math.isfinite(lambda_cost)
        or lambda_cost <= 0
        or not math.isfinite(c_ref)
        or c_ref <= 0
    ):
        raise ValueError("invalid scalar reward inputs")
    reward = -lambda_cost * costs / c_ref
    reward[-1] += float(final_ex)
    return reward


def return_to_go(rewards: np.ndarray, *, gamma: float = 1.0) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all() or gamma != 1.0:
        raise ValueError("State V26 return-to-go requires six rewards and gamma=1")
    result = np.empty_like(values)
    running = 0.0
    for index in range(5, -1, -1):
        running = values[index] + running
        result[index] = running
    return result


def mc_huber_loss(chosen_q: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(chosen_q, returns)


def _global_grad_norm(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
    terms = [grad.square().sum() for grad in grads if grad is not None]
    if not terms:
        return torch.tensor(0.0)
    return torch.sqrt(torch.stack(terms).sum())


def weighted_auxiliary_gradient_ratio(
    td_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    weight: float,
) -> float:
    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    td_norm = _global_grad_norm(
        torch.autograd.grad(td_loss, params, retain_graph=True, allow_unused=True)
    )
    auxiliary_norm = _global_grad_norm(
        torch.autograd.grad(
            auxiliary_loss, params, retain_graph=True, allow_unused=True
        )
    )
    if (
        td_norm <= 0
        or auxiliary_norm <= 0
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("gradient-ratio inputs must be positive and finite")
    return float((weight * auxiliary_norm / td_norm).detach().cpu())
