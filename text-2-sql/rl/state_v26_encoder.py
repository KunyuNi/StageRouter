"""Composition of State V25 with frozen State V26 context features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .state_v25_schema import STATE_V25_DIM
from .state_v26_schema import CONTEXT_SLICES, STATE_V26_DIM, STATE_V26_VERSION


@dataclass(frozen=True)
class StateV26FeatureFlags:
    cost_context: bool
    s4_difficulty_context: bool


class StateV26Encoder:
    state_version = STATE_V26_VERSION
    state_dim = STATE_V26_DIM

    def __init__(self, v25: Any, context_provider: Any, flags: StateV26FeatureFlags):
        self.v25 = v25
        self.context_provider = context_provider
        self.flags = flags

    def encode_question(self, question: str, evidence: str = "") -> Any:
        return self.v25.encode_question(question, evidence)

    def encode(
        self,
        payload: Any,
        stage_idx: int,
        budget_remaining: float,
        tier_history: Sequence[int],
        cost_history: Sequence[float],
        question_embedding: Any,
        action_mask: Any = None,
    ) -> np.ndarray:
        prefix = self.v25.encode(
            payload,
            stage_idx,
            budget_remaining,
            tier_history,
            cost_history,
            question_embedding,
            action_mask,
        )
        prefix = np.asarray(prefix, dtype=np.float32)
        if prefix.shape != (STATE_V25_DIM,):
            raise ValueError("State V25 prefix has the wrong shape")
        state = np.zeros(STATE_V26_DIM, dtype=np.float32)
        state[:STATE_V25_DIM] = prefix
        if self.flags.cost_context or self.flags.s4_difficulty_context:
            context = np.asarray(
                self.context_provider.get(payload, stage_idx, tier_history),
                dtype=np.float32,
            )
            if context.shape != (19,) or not np.isfinite(context).all():
                raise ValueError("State V26 context must be finite float32[19]")
            if self.flags.cost_context:
                state[CONTEXT_SLICES["cost_context"]] = context[:11]
            if self.flags.s4_difficulty_context:
                state[CONTEXT_SLICES["s4_difficulty_context"]] = context[11:]
        if not np.isfinite(state).all():
            raise ValueError("State V26 encoder produced nonfinite values")
        return state

    @staticmethod
    def terminal_state() -> np.ndarray:
        return np.zeros(STATE_V26_DIM, dtype=np.float32)
