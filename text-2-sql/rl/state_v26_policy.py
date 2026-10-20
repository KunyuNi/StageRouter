"""Greedy action selection for the locked State V26 Q-network."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class StateV26Policy:
    def __init__(self, network: torch.nn.Module, device: str | torch.device) -> None:
        self.network = network
        self.device = torch.device(device)
        self.network.to(self.device).eval()

    def _q_tensor(
        self, state: np.ndarray | torch.Tensor, action_mask: Sequence[int | float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        with torch.no_grad():
            q = self.network(state_tensor, mask)[0]
        return q, mask

    def q_values(
        self, state: np.ndarray | torch.Tensor, action_mask: Sequence[int | float]
    ) -> np.ndarray:
        q, _ = self._q_tensor(state, action_mask)
        return q.detach().cpu().numpy().astype(np.float64, copy=False)

    def select_action(
        self, state: np.ndarray | torch.Tensor, action_mask: Sequence[int | float]
    ) -> int:
        q, mask = self._q_tensor(state, action_mask)
        return int(q.masked_fill(~mask[0], -torch.inf).argmax())
