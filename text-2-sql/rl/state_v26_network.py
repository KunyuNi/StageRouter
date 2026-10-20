"""Three-view value-advantage network for stage-level routing."""

from __future__ import annotations

import torch
import torch.nn as nn

from .state_v25_schema import BLOCK_SLICES, STATE_V25_DIM
from .state_v26_schema import STATE_V26_DIM


NUM_ACTIONS = 4
VIEW_DIM = 128
FUSED_DIM = 512


def _view_encoder(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, VIEW_DIM),
        nn.SiLU(),
        nn.LayerNorm(VIEW_DIM),
    )


class StateV26ScalarDuelingNetwork(nn.Module):
    requires_action_mask = True

    def __init__(self) -> None:
        super().__init__()
        self.state_dim = STATE_V26_DIM
        self.action_dim = NUM_ACTIONS
        self.hidden_dim = FUSED_DIM
        self.request_slice = slice(BLOCK_SLICES["task"].start, BLOCK_SLICES["database"].stop)
        self.artifact_slice = slice(BLOCK_SLICES["retrieval"].start, BLOCK_SLICES["candidate_sql"].stop)
        self.trajectory_slice = BLOCK_SLICES["control"]
        self.request_encoder = _view_encoder(self.request_slice.stop - self.request_slice.start)
        self.artifact_encoder = _view_encoder(self.artifact_slice.stop - self.artifact_slice.start)
        self.trajectory_encoder = _view_encoder(self.trajectory_slice.stop - self.trajectory_slice.start)
        self.fusion = nn.Sequential(
            nn.Linear(3 * VIEW_DIM, FUSED_DIM),
            nn.SiLU(),
            nn.LayerNorm(FUSED_DIM),
        )
        self.value_head = nn.Linear(FUSED_DIM, 1)
        self.advantage_head = nn.Linear(FUSED_DIM, NUM_ACTIONS)

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        if not isinstance(state, torch.Tensor) or state.dim() not in (1, 2) or state.shape[-1] != STATE_V26_DIM:
            raise ValueError(f"state must have shape [{STATE_V26_DIM}] or [batch,{STATE_V26_DIM}]")
        parameter = next(self.parameters())
        state = state.to(device=parameter.device, dtype=parameter.dtype)
        if not torch.isfinite(state).all().item():
            raise ValueError("state must contain only finite values")
        if not (state[..., STATE_V25_DIM:] == 0).all().item():
            raise ValueError("inactive State V26 context dimensions must be zero")
        if state.dim() == 1:
            state = state.unsqueeze(0)
        views = (
            self.request_encoder(state[:, self.request_slice]),
            self.artifact_encoder(state[:, self.artifact_slice]),
            self.trajectory_encoder(state[:, self.trajectory_slice]),
        )
        return self.fusion(torch.cat(views, dim=-1))

    @staticmethod
    def _legal_mask(mask: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor) or mask.device != hidden.device:
            raise ValueError("action mask must be a tensor on the state device")
        if mask.dim() == 1:
            if mask.shape != (NUM_ACTIONS,):
                raise ValueError("action mask shape is invalid")
            mask = mask.unsqueeze(0).expand(hidden.shape[0], -1)
        elif mask.shape != (hidden.shape[0], NUM_ACTIONS):
            raise ValueError("action mask shape is invalid")
        if mask.is_floating_point() and not torch.isfinite(mask).all().item():
            raise ValueError("action mask must be finite")
        if not ((mask == 0) | (mask == 1)).all().item() or not (mask.sum(dim=1) > 0).all().item():
            raise ValueError("each action mask must contain at least one legal binary action")
        return mask.to(dtype=hidden.dtype)

    def _components(
        self, state: torch.Tensor, action_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self._encode(state)
        legal = self._legal_mask(action_mask, hidden)
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        mean_advantage = (advantage * legal).sum(dim=1, keepdim=True) / legal.sum(dim=1, keepdim=True)
        return value, advantage, value + advantage - mean_advantage

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self._components(state, action_mask)[2]

    def debug_value_advantage(
        self, state: torch.Tensor, action_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, advantage, _ = self._components(state, action_mask)
        return value, advantage
