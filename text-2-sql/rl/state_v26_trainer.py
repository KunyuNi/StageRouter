"""Offline masked Double DQN trainer for State V26."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .state_v26_objective import (
    StateV26ObjectiveConfig,
    mc_huber_loss,
    weighted_auxiliary_gradient_ratio,
)
from .state_v26_replay import StateV26ReplayBatch


def _masked(q: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if q.shape != masks.shape or not masks.any(dim=1).all().item():
        raise ValueError("invalid action masks")
    return q.masked_fill(~masks, -torch.inf)


@torch.no_grad()
def compute_state_v26_double_dqn_target(
    online: torch.nn.Module,
    target: torch.nn.Module,
    next_states: torch.Tensor,
    next_masks: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    nonterminal = dones == 0
    target_q = torch.zeros_like(rewards)
    if nonterminal.any().item():
        active_states = next_states[nonterminal]
        active_masks = next_masks[nonterminal]
        online_q = _masked(online(active_states, active_masks), active_masks)
        next_actions = online_q.argmax(dim=1)
        target_q[nonterminal] = target(active_states, active_masks).gather(
            1, next_actions[:, None]
        ).squeeze(1)
    return rewards + gamma * target_q


class OfflineStateV26Trainer:
    def __init__(
        self,
        online: torch.nn.Module,
        target: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        objective: StateV26ObjectiveConfig,
        *,
        target_update_interval: int = 500,
        grad_clip: float = 1.0,
        identity: Mapping[str, Any] | None = None,
    ) -> None:
        objective.validate()
        self.online = online
        self.target = target
        self.optimizer = optimizer
        self.objective = objective
        self.target_update_interval = int(target_update_interval)
        self.grad_clip = float(grad_clip)
        self.identity = dict(identity or {})
        self.epoch = 0
        self.global_step = 0
        self.target.load_state_dict(self.online.state_dict(), strict=True)

    @property
    def device(self) -> torch.device:
        return next(self.online.parameters()).device

    def _tensor(self, value: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def train_batch(
        self,
        batch: StateV26ReplayBatch,
    ) -> dict[str, float]:
        states = self._tensor(batch.states, torch.float32)
        actions = self._tensor(batch.actions, torch.long)
        rewards = self._tensor(batch.rewards, torch.float32)
        returns = self._tensor(batch.return_to_go, torch.float32)
        next_states = self._tensor(batch.next_states, torch.float32)
        dones = self._tensor(batch.dones, torch.float32)
        masks = self._tensor(batch.action_masks, torch.bool)
        next_masks = self._tensor(batch.next_action_masks, torch.bool)
        q = self.online(states, masks)
        chosen_q = q.gather(1, actions[:, None]).squeeze(1)
        td_target = compute_state_v26_double_dqn_target(
            self.online,
            self.target,
            next_states,
            next_masks,
            rewards,
            dones,
            self.objective.gamma,
        )
        td_loss = F.smooth_l1_loss(chosen_q, td_target)
        zero = chosen_q.sum() * 0.0
        mc_loss = (
            mc_huber_loss(chosen_q, returns)
            if self.objective.mc_weight > 0
            else zero
        )
        loss = td_loss + self.objective.mc_weight * mc_loss
        if not torch.isfinite(loss).item():
            raise ValueError("State V26 training loss is nonfinite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.grad_clip
        )
        self.optimizer.step()
        self.global_step += 1
        if self.global_step % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict(), strict=True)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "td_loss": float(td_loss.detach().cpu()),
            "mc_loss": float(mc_loss.detach().cpu()),
            "mc_weighted_loss": float(
                (self.objective.mc_weight * mc_loss).detach().cpu()
            ),
            "q_mean": float(chosen_q.detach().mean().cpu()),
            "target_mean": float(td_target.detach().mean().cpu()),
            "return_mean": float(returns.detach().mean().cpu()),
            "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "epoch": float(self.epoch),
            "global_step": float(self.global_step),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("State V26 metrics are nonfinite")
        return metrics

    def auxiliary_gradient_ratios(
        self,
        batch: StateV26ReplayBatch,
    ) -> dict[str, float]:
        states = self._tensor(batch.states, torch.float32)
        actions = self._tensor(batch.actions, torch.long)
        rewards = self._tensor(batch.rewards, torch.float32)
        returns = self._tensor(batch.return_to_go, torch.float32)
        next_states = self._tensor(batch.next_states, torch.float32)
        dones = self._tensor(batch.dones, torch.float32)
        masks = self._tensor(batch.action_masks, torch.bool)
        next_masks = self._tensor(batch.next_action_masks, torch.bool)
        chosen_q = self.online(states, masks).gather(1, actions[:, None]).squeeze(1)
        td_target = compute_state_v26_double_dqn_target(
            self.online,
            self.target,
            next_states,
            next_masks,
            rewards,
            dones,
            self.objective.gamma,
        )
        td_loss = F.smooth_l1_loss(chosen_q, td_target)
        result: dict[str, float] = {}
        if self.objective.mc_weight > 0:
            result["mc_to_td"] = weighted_auxiliary_gradient_ratio(
                td_loss,
                mc_huber_loss(chosen_q, returns),
                self.online.parameters(),
                self.objective.mc_weight,
            )
        if not all(math.isfinite(value) and value > 0 for value in result.values()):
            raise ValueError("State V26 auxiliary gradient ratio is invalid")
        self.optimizer.zero_grad(set_to_none=True)
        return result

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "algorithm": "state_v26_scalar_double_dqn",
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "objective": dict(self.objective.__dict__),
            "identity": self.identity,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("format_version") != 1 or payload.get(
            "algorithm"
        ) != "state_v26_scalar_double_dqn":
            raise ValueError("State V26 checkpoint format mismatch")
        if payload.get("identity") != self.identity:
            raise ValueError("State V26 checkpoint identity mismatch")
        if payload.get("objective") != dict(self.objective.__dict__):
            raise ValueError("State V26 checkpoint objective mismatch")
        self.online.load_state_dict(payload["online_state_dict"], strict=True)
        self.target.load_state_dict(payload["target_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epoch = int(payload["epoch"])
        self.global_step = int(payload["global_step"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
