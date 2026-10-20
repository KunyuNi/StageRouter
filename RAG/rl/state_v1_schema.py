"""Immutable block layout for the 9,516-value RAG State V1 tensor."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


STATE_V1_BLOCK_SHAPES = (
    ("question_semantic", (1024,)),
    ("question_structured", (24,)),
    ("question_type", (5,)),
    ("query_semantics", (2, 1024)),
    ("query_features", (2, 8)),
    ("passage_semantics", (6, 1024)),
    ("passage_features", (6, 20)),
    ("passage_mask", (6,)),
    ("retrieval_summary", (32,)),
    ("transition_delta", (24,)),
    ("route_control", (32,)),
    ("cost_history", (17,)),
    ("availability", (8,)),
    ("difficulty_context", (16,)),
)
STATE_V1_DIM = 9516
STATE_V1_SCHEMA_ID = "rag_state_v1_9516"


@dataclass
class StateV1Blocks:
    question_semantic: np.ndarray
    question_structured: np.ndarray
    question_type: np.ndarray
    query_semantics: np.ndarray
    query_features: np.ndarray
    passage_semantics: np.ndarray
    passage_features: np.ndarray
    passage_mask: np.ndarray
    retrieval_summary: np.ndarray
    transition_delta: np.ndarray
    route_control: np.ndarray
    cost_history: np.ndarray
    availability: np.ndarray
    difficulty_context: np.ndarray

    @classmethod
    def zeros(cls) -> "StateV1Blocks":
        return cls(
            **{
                name: np.zeros(shape, dtype=np.float32)
                for name, shape in STATE_V1_BLOCK_SHAPES
            }
        )


class StateV1Schema:
    def __init__(self) -> None:
        self.shapes = dict(STATE_V1_BLOCK_SHAPES)
        self.flat_sizes = {
            name: int(np.prod(shape, dtype=np.int64))
            for name, shape in STATE_V1_BLOCK_SHAPES
        }
        self.slices: dict[str, slice] = {}
        offset = 0
        for name, _shape in STATE_V1_BLOCK_SHAPES:
            size = self.flat_sizes[name]
            self.slices[name] = slice(offset, offset + size)
            offset += size
        if offset != STATE_V1_DIM:
            raise RuntimeError(
                f"State V1 block sizes total {offset}, expected {STATE_V1_DIM}"
            )

    def flatten(self, blocks: StateV1Blocks) -> np.ndarray:
        if not isinstance(blocks, StateV1Blocks):
            raise TypeError("blocks must be StateV1Blocks")
        expected_names = tuple(name for name, _shape in STATE_V1_BLOCK_SHAPES)
        actual_names = tuple(item.name for item in fields(blocks))
        if actual_names != expected_names:
            raise ValueError("StateV1Blocks field order does not match schema")
        flattened: list[np.ndarray] = []
        for name, shape in STATE_V1_BLOCK_SHAPES:
            value = np.asarray(getattr(blocks, name))
            if value.shape != shape:
                raise ValueError(
                    f"{name} has shape {value.shape}, expected {shape}"
                )
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError(f"{name} must be numeric")
            value = np.asarray(value, dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            flattened.append(value.reshape(-1))
        vector = np.concatenate(flattened).astype(np.float32, copy=False)
        if vector.shape != (STATE_V1_DIM,):
            raise RuntimeError("flattened State V1 has an invalid dimension")
        return vector

    def split(self, vector: np.ndarray) -> StateV1Blocks:
        value = np.asarray(vector)
        if value.shape != (STATE_V1_DIM,):
            raise ValueError(
                f"State V1 vector has shape {value.shape}, expected {(STATE_V1_DIM,)}"
            )
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError("State V1 vector contains non-finite values")
        return StateV1Blocks(
            **{
                name: value[self.slices[name]].reshape(shape).copy()
                for name, shape in STATE_V1_BLOCK_SHAPES
            }
        )


STATE_V1_SCHEMA = StateV1Schema()


def legal_action_mask(decision_idx: int) -> np.ndarray:
    decision_idx = int(decision_idx)
    if decision_idx not in range(4):
        raise ValueError(f"decision_idx must be in 0..3, got {decision_idx}")
    mask = [True, True, True, decision_idx != 3]
    return np.asarray(mask, dtype=np.bool_)


__all__ = [
    "STATE_V1_BLOCK_SHAPES",
    "STATE_V1_DIM",
    "STATE_V1_SCHEMA_ID",
    "STATE_V1_SCHEMA",
    "StateV1Blocks",
    "StateV1Schema",
    "legal_action_mask",
]
