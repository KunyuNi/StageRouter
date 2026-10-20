"""State V25 vector encoder with strict causal semantic lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rl.semantic_config import DEFAULT_SEMANTIC_MODEL
from rl.state_v3_context import StaticSchemaContextProvider, payload_get
from rl.state_v25_context import build_state_v25_context, normalize_text
from rl.state_v25_schema import (
    BLOCK_SLICES,
    SEMANTIC_DIMS,
    SEMANTIC_SLICES,
    STATE_V25_DIM,
    STATE_V25_VERSION,
    STRUCTURED_SLICES,
)
from rl.state_v25_semantic import StateV25SemanticCache, StateV25TextEncoder


class StateV25Encoder:
    """Concatenate six causal blocks without outcomes, latency, or sunk cost."""

    state_version = STATE_V25_VERSION
    state_dim = STATE_V25_DIM

    def __init__(
        self,
        db_root: str | Path | None = None,
        *,
        db_mode: str = "train",
        plm_model_name: str | Path = DEFAULT_SEMANTIC_MODEL,
        device: str = "cpu",
        semantic_cache_path: str | Path = "cache/state_v25_semantics.db",
        semantic_cache_writable: bool = False,
        static_schema_provider: Any | None = None,
        semantic_cache: StateV25SemanticCache | None = None,
        text_encoder: Any | None = None,
    ) -> None:
        if static_schema_provider is None:
            if db_root is None:
                raise ValueError(
                    "db_root is required when static_schema_provider is not injected"
                )
            static_schema_provider = StaticSchemaContextProvider(db_root, db_mode)
        if not callable(getattr(static_schema_provider, "get", None)):
            raise TypeError("static_schema_provider must provide get(db_id)")
        if text_encoder is None:
            text_encoder = StateV25TextEncoder(
                plm_model_name,
                device=device,
                max_length=8192,
            )
        if not callable(getattr(text_encoder, "token_count", None)):
            raise TypeError("State V25 text encoder must provide token_count(text)")
        if semantic_cache is None:
            semantic_cache = StateV25SemanticCache(
                semantic_cache_path,
                text_encoder,
                writable=semantic_cache_writable,
            )
        if not callable(getattr(semantic_cache, "vector", None)):
            raise TypeError("semantic_cache must provide vector(role, text)")
        self.db_mode = str(db_mode)
        self.static_schema_provider = static_schema_provider
        self.text_encoder = text_encoder
        self.semantic_cache = semantic_cache

    @staticmethod
    def _task_text(question: str, evidence: str) -> str:
        question = normalize_text(question)
        evidence = normalize_text(evidence)
        combined = question if not evidence else f"{question} [SEP] {evidence}"
        return f"task\n{combined}" if combined else ""

    def encode_question(self, question: str, evidence: str = "") -> np.ndarray:
        vector = self.semantic_cache.vector(
            "task", self._task_text(question, evidence)
        )
        return self._semantic_vector(vector, "task")

    @staticmethod
    def _semantic_vector(value: Any, role: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (SEMANTIC_DIMS[role],) or not np.isfinite(vector).all():
            raise ValueError(
                f"State V25 {role} semantic vector must be finite 1024D"
            )
        return vector

    def encode(
        self,
        payload: Any,
        stage_idx: int,
        budget_remaining: float,
        tier_history,
        cost_history,
        question_embedding: np.ndarray | None,
        action_mask=None,
    ) -> np.ndarray:
        del budget_remaining, cost_history, action_mask
        db_id = str(payload_get(payload, "db_id", "") or "")
        if not db_id:
            raise ValueError("State V25 payload requires db_id")
        static = self.static_schema_provider.get(db_id)
        context = build_state_v25_context(
            payload,
            stage_idx,
            tier_history,
            static,
            token_count=self.text_encoder.token_count,
        )
        state = np.zeros(STATE_V25_DIM, dtype=np.float32)
        for block in BLOCK_SLICES:
            state[STRUCTURED_SLICES[block]] = context.structured[block]
            if SEMANTIC_DIMS[block] == 0:
                continue
            if block == "task" and question_embedding is not None:
                vector = self._semantic_vector(question_embedding, "task")
            else:
                vector = self.semantic_cache.vector(
                    block, context.semantic_texts[block]
                )
                vector = self._semantic_vector(vector, block)
            state[SEMANTIC_SLICES[block]] = vector
        if state.shape != (STATE_V25_DIM,) or not np.isfinite(state).all():
            raise ValueError("State V25 encoder produced an invalid state")
        return state

    @staticmethod
    def terminal_state() -> np.ndarray:
        return np.zeros(STATE_V25_DIM, dtype=np.float32)
