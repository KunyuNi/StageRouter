"""Causal State V1 assembly, normalization, and reproducible hashes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from RAG.env._shared.state_payload import StatePayload

from .decision_session import DecisionObservation
from .state_v1_features import (
    DecisionCostPriors,
    availability_features,
    build_cost_history,
    difficulty_context_features,
    encode_question_type,
    extract_passages,
    passage_feature_rows,
    query_feature_rows,
    question_structured_features,
    retrieval_summary_features,
    route_control_features,
    transition_delta_features,
)
from .state_v1_schema import (
    STATE_V1_DIM,
    STATE_V1_SCHEMA,
    STATE_V1_SCHEMA_ID,
    StateV1Blocks,
    legal_action_mask,
)
from .state_v1_semantic import text_hash


def _normalization_mask() -> np.ndarray:
    mask = np.zeros(STATE_V1_DIM, dtype=np.bool_)

    def select(block: str, indices: Any) -> None:
        block_mask = mask[STATE_V1_SCHEMA.slices[block]].reshape(
            STATE_V1_SCHEMA.shapes[block]
        )
        block_mask[indices] = True

    select("question_structured", slice(0, 16))
    select("query_features", (slice(None), slice(1, 8)))
    select(
        "passage_features",
        (slice(None), [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 18, 19]),
    )
    select("retrieval_summary", slice(None))
    select("transition_delta", slice(0, 22))
    select("difficulty_context", [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    return mask


NORMALIZED_POSITIONS = _normalization_mask()


class StateV1Normalizer:
    def __init__(
        self,
        *,
        means: np.ndarray,
        stds: np.ndarray,
        count: int,
        identity: str | None = None,
    ) -> None:
        self.means = np.asarray(means, dtype=np.float64).reshape(-1)
        self.stds = np.asarray(stds, dtype=np.float64).reshape(-1)
        if self.means.shape != (STATE_V1_DIM,) or self.stds.shape != (STATE_V1_DIM,):
            raise ValueError("State V1 normalizer arrays must have length 9516")
        if not np.isfinite(self.means).all() or not np.isfinite(self.stds).all():
            raise ValueError("State V1 normalizer contains non-finite values")
        if np.any(self.stds <= 0.0):
            raise ValueError("State V1 normalizer standard deviations must be positive")
        self.count = int(count)
        self.identity = str(identity or self.compute_identity())

    @classmethod
    def identity_transform(cls) -> "StateV1Normalizer":
        return cls(
            means=np.zeros(STATE_V1_DIM, dtype=np.float64),
            stds=np.ones(STATE_V1_DIM, dtype=np.float64),
            count=0,
        )

    @classmethod
    def fit(cls, raw_blocks: Iterable[StateV1Blocks]) -> "StateV1Normalizer":
        count = 0
        means = np.zeros(STATE_V1_DIM, dtype=np.float64)
        m2 = np.zeros(STATE_V1_DIM, dtype=np.float64)
        selected = np.flatnonzero(NORMALIZED_POSITIONS)
        for blocks in raw_blocks:
            vector = STATE_V1_SCHEMA.flatten(blocks).astype(np.float64)
            count += 1
            delta = vector[selected] - means[selected]
            means[selected] += delta / count
            m2[selected] += delta * (vector[selected] - means[selected])
        if count == 0:
            raise ValueError("cannot fit State V1 normalizer on no states")
        variance = m2 / count
        stds = np.sqrt(np.maximum(variance, 0.0))
        stds = np.maximum(stds, 1e-8)
        means[~NORMALIZED_POSITIONS] = 0.0
        stds[~NORMALIZED_POSITIONS] = 1.0
        return cls(means=means, stds=stds, count=count)

    def transform(self, blocks: StateV1Blocks) -> StateV1Blocks:
        vector = STATE_V1_SCHEMA.flatten(blocks).astype(np.float64)
        transformed = vector.copy()
        transformed[NORMALIZED_POSITIONS] = np.clip(
            (vector[NORMALIZED_POSITIONS] - self.means[NORMALIZED_POSITIONS])
            / self.stds[NORMALIZED_POSITIONS],
            -5.0,
            5.0,
        )
        return STATE_V1_SCHEMA.split(transformed.astype(np.float32))

    def compute_identity(self) -> str:
        digest = hashlib.sha256()
        digest.update(STATE_V1_SCHEMA_ID.encode("utf-8"))
        digest.update(int(self.count).to_bytes(8, "little", signed=False))
        digest.update(np.asarray(self.means, dtype="<f8").tobytes())
        digest.update(np.asarray(self.stds, dtype="<f8").tobytes())
        digest.update(NORMALIZED_POSITIONS.tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_id": STATE_V1_SCHEMA_ID,
            "count": self.count,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "identity": self.identity,
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

    @classmethod
    def load(cls, path: str | Path) -> "StateV1Normalizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema_id") != STATE_V1_SCHEMA_ID:
            raise ValueError("State V1 normalizer schema identity mismatch")
        normalizer = cls(
            means=np.asarray(data["means"], dtype=np.float64),
            stds=np.asarray(data["stds"], dtype=np.float64),
            count=int(data["count"]),
            identity=str(data["identity"]),
        )
        if normalizer.compute_identity() != normalizer.identity:
            raise ValueError("State V1 normalizer identity mismatch")
        return normalizer


@dataclass(frozen=True)
class EncodedState:
    vector: np.ndarray
    legal_action_mask: np.ndarray
    logical_state_hash: str
    tensor_state_hash: str


def _causal_payload(payload: StatePayload | None) -> Mapping[str, Any] | None:
    if payload is None:
        return None

    operational_keys = {
        "cache_hit",
        "cached",
        "cost",
        "embedding_cost",
        "embedding_input_tokens",
        "embedding_latency_ms",
        "input_tokens",
        "latency_ms",
        "output_tokens",
        "total_cost",
        "total_latency_ms",
    }

    def semantic_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): semantic_value(item)
                for key, item in value.items()
                if str(key) not in operational_keys
            }
        if isinstance(value, (list, tuple)):
            return [semantic_value(item) for item in value]
        return value

    visible_metadata = {}
    for name in ("question_type", "retrieval1", "retrieval2", "rerank"):
        if name in payload.metadata:
            visible_metadata[name] = semantic_value(payload.metadata[name])
    return {
        "question_id": payload.question_id,
        "question": payload.question,
        "query_1": payload.query_1,
        "query_2": payload.query_2,
        "context": list(payload.context),
        "metadata": visible_metadata,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class StateV1Encoder:
    def __init__(
        self,
        *,
        semantic_encoder: Any,
        cost_priors: DecisionCostPriors,
        normalizer: StateV1Normalizer,
        prompt_identity: str,
        c_ref: float = 0.00333879,
    ) -> None:
        if not prompt_identity:
            raise ValueError("prompt_identity is required")
        self.semantic_encoder = semantic_encoder
        self.cost_priors = cost_priors
        self.normalizer = normalizer
        self.prompt_identity = str(prompt_identity)
        self.c_ref = float(c_ref)

    def raw_blocks(self, observation: DecisionObservation) -> StateV1Blocks:
        if observation.decision_idx not in range(4):
            raise ValueError("State V1 decision index must be in 0..3")
        payload = observation.payload
        question = payload.question or ""
        if not question:
            raise ValueError("State V1 requires a nonempty question")
        passages = extract_passages(payload)
        ordered_texts: list[str] = []
        for value in [question, payload.query_1, payload.query_2]:
            if value and value not in ordered_texts:
                ordered_texts.append(value)
        for passage in passages:
            if passage.text and passage.text not in ordered_texts:
                ordered_texts.append(passage.text)
        vectors = self.semantic_encoder.encode_texts(ordered_texts)
        if vectors.shape != (len(ordered_texts), 1024):
            raise ValueError("semantic encoder must return [N,1024] float vectors")
        embeddings = {
            value: np.asarray(vector, dtype=np.float32)
            for value, vector in zip(ordered_texts, vectors)
        }

        blocks = StateV1Blocks.zeros()
        blocks.question_semantic = embeddings[question].copy()
        blocks.question_structured = question_structured_features(question)
        blocks.question_type = encode_question_type(
            payload.metadata.get("question_type")
        )
        for index, query in enumerate((payload.query_1, payload.query_2)):
            if query:
                blocks.query_semantics[index] = embeddings[query]
        blocks.query_features = query_feature_rows(observation, embeddings)
        passage_features, passage_mask, feature_passages = passage_feature_rows(
            observation, embeddings
        )
        for index, passage in enumerate(feature_passages[:6]):
            if passage.text:
                blocks.passage_semantics[index] = embeddings[passage.text]
        blocks.passage_features = passage_features
        blocks.passage_mask = passage_mask
        blocks.retrieval_summary = retrieval_summary_features(
            observation, embeddings
        )
        blocks.transition_delta = transition_delta_features(observation, embeddings)
        blocks.route_control = route_control_features(observation)
        blocks.cost_history = build_cost_history(
            observation, self.cost_priors, c_ref=self.c_ref
        )
        blocks.availability = availability_features(observation)
        blocks.difficulty_context = difficulty_context_features(
            observation, embeddings
        )
        STATE_V1_SCHEMA.flatten(blocks)
        return blocks

    def _logical_hash(
        self, observation: DecisionObservation, blocks: StateV1Blocks
    ) -> str:
        payload = observation.payload
        passages = extract_passages(payload)
        semantic_text_hashes = {
            "question": text_hash(payload.question or ""),
            "query_1": text_hash(payload.query_1) if payload.query_1 else None,
            "query_2": text_hash(payload.query_2) if payload.query_2 else None,
            "passages": [text_hash(item.text) for item in passages],
        }
        canonical = {
            "schema_id": STATE_V1_SCHEMA_ID,
            "question_id": observation.question_id,
            "decision_idx": observation.decision_idx,
            "payload": _causal_payload(observation.payload),
            "previous_decision_payload": _causal_payload(
                observation.previous_decision_payload
            ),
            "action_history": list(observation.action_history),
            "bundled_cost_history": list(observation.bundled_cost_history),
            "legal_action_mask": legal_action_mask(
                observation.decision_idx
            ).astype(int).tolist(),
            "availability": blocks.availability.tolist(),
            "cost_priors_identity": self.cost_priors.identity,
            "semantic_text_hashes": semantic_text_hashes,
        }
        serialized = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def encode(self, observation: DecisionObservation) -> EncodedState:
        raw = self.raw_blocks(observation)
        logical_hash = self._logical_hash(observation, raw)
        normalized = self.normalizer.transform(raw)
        vector = STATE_V1_SCHEMA.flatten(normalized)
        semantic_identity = str(
            getattr(self.semantic_encoder, "model_identity", "")
        )
        digest = hashlib.sha256()
        digest.update(np.asarray(vector, dtype="<f4").tobytes(order="C"))
        for identity in (
            STATE_V1_SCHEMA_ID,
            self.normalizer.identity,
            semantic_identity,
            self.prompt_identity,
        ):
            encoded = identity.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        return EncodedState(
            vector=vector,
            legal_action_mask=legal_action_mask(observation.decision_idx),
            logical_state_hash=logical_hash,
            tensor_state_hash=digest.hexdigest(),
        )


__all__ = [
    "EncodedState",
    "NORMALIZED_POSITIONS",
    "StateV1Encoder",
    "StateV1Normalizer",
]
