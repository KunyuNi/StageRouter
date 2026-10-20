"""Deterministic legal action policy for the scalar Dueling Q network."""

from __future__ import annotations

import torch

from .dqn_objective import legal_argmax


class ScalarGreedyPolicy:
    def select(
        self,
        *,
        q: torch.Tensor,
        legal_mask: torch.Tensor,
        empirical_cost: torch.Tensor | None = None,
    ) -> int:
        return int(
            legal_argmax(
                q.reshape(1, 4),
                legal_mask.reshape(1, 4),
                None if empirical_cost is None else empirical_cost.reshape(1, 4),
            )[0]
        )


__all__ = ["ScalarGreedyPolicy"]
