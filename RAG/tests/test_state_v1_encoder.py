from __future__ import annotations

import hashlib

import numpy as np

from RAG.env._shared.state_payload import StatePayload
from RAG.rl.decision_session import DecisionObservation
from RAG.rl.state_v1_encoder import StateV1Encoder, StateV1Normalizer
from RAG.rl.state_v1_features import DecisionCostPriors
from RAG.rl.state_v1_schema import STATE_V1_DIM, StateV1Blocks


class FakeSemanticEncoder:
    model_identity = "fake-gte-v1"

    def encode_texts(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            raw = np.frombuffer(digest * 128, dtype=np.uint8)[:1024].astype(np.float32)
            raw = raw - raw.mean()
            rows.append(raw / np.linalg.norm(raw))
        return np.stack(rows).astype(np.float32)


PRIORS = DecisionCostPriors(
    means=(
        (0.0001, 0.0002, 0.00015, 0.00012),
        (0.0, 0.0002, 0.00015, 0.00012),
        (0.0, 0.001, 0.0008, 0.0006),
        (0.0, 0.0005, 0.0004, 0.0003),
    ),
    identity="train-priors-v1",
)


def _payload(*, reranked: bool = False) -> StatePayload:
    payload = StatePayload(
        question_id=7,
        question="Which city won in 2020?",
        query_1="city winner 2020",
        ground_truth="Alpha",
        context=["Alpha won in 2020", "Beta hosted the event"],
        metadata={
            "question_type": "temporal_query",
            "retrieval1": {
                "results": [
                    {
                        "chunk_id": "c1",
                        "text": "Alpha won in 2020",
                        "rank": 1,
                        "score": 0.9,
                        "metadata": {"source": "news", "title": "Result"},
                    },
                    {
                        "chunk_id": "c2",
                        "text": "Beta hosted the event",
                        "rank": 2,
                        "score": 0.8,
                        "metadata": {"source": "archive", "title": "Host"},
                    },
                ]
            },
        },
    )
    if reranked:
        payload.context = ["Beta hosted the event", "Alpha won in 2020"]
        payload.metadata["rerank"] = {
            "results": [
                {
                    "chunk_id": "c2",
                    "text": "Beta hosted the event",
                    "rerank_rank": 1,
                    "rerank_score": 90,
                },
                {
                    "chunk_id": "c1",
                    "text": "Alpha won in 2020",
                    "rerank_rank": 2,
                    "rerank_score": 80,
                },
            ]
        }
    return payload


def _observation(payload: StatePayload, previous: StatePayload | None = None):
    return DecisionObservation(
        question_id=7,
        decision_idx=3,
        payload=payload,
        previous_decision_payload=previous or payload.snapshot(),
        action_history=(0, 3, 0),
        bundled_cost_history=(0.0001, 0.0, 0.001),
        physical_stage_costs=(0.0,) * 7,
        physical_stage_latencies_ms=(0.0,) * 7,
    )


def _encoder(normalizer: StateV1Normalizer) -> StateV1Encoder:
    return StateV1Encoder(
        semantic_encoder=FakeSemanticEncoder(),
        cost_priors=PRIORS,
        normalizer=normalizer,
        prompt_identity="stage-prompts-v1",
    )


def test_tensor_hash_changes_with_normalizer_but_logical_hash_does_not():
    identity = StateV1Normalizer.identity_transform()
    shifted = StateV1Normalizer.identity_transform()
    shifted.means[1024] = 0.25
    shifted.identity = shifted.compute_identity()

    a = _encoder(identity).encode(_observation(_payload()))
    b = _encoder(shifted).encode(_observation(_payload()))

    assert a.logical_state_hash == b.logical_state_hash
    assert a.tensor_state_hash != b.tensor_state_hash


def test_future_answer_and_f1_cannot_change_pre_answer_state():
    left_payload = _payload()
    left_payload.answer = "Alpha"
    left_payload.evaluation = {"f1_score": 1.0}
    right_payload = _payload()
    right_payload.answer = "Beta"
    right_payload.evaluation = {"f1_score": 0.0}
    encoder = _encoder(StateV1Normalizer.identity_transform())

    left = encoder.encode(_observation(left_payload))
    right = encoder.encode(_observation(right_payload))

    np.testing.assert_array_equal(left.vector, right.vector)
    assert left.logical_state_hash == right.logical_state_hash


def test_operational_retrieval_usage_cannot_change_logical_state_hash():
    left_payload = _payload()
    left_payload.metadata["retrieval1"].update(
        {"embedding_latency_ms": 12.0, "embedding_input_tokens": 111}
    )
    right_payload = _payload()
    right_payload.metadata["retrieval1"].update(
        {"embedding_latency_ms": 987.0, "embedding_input_tokens": 999}
    )
    encoder = _encoder(StateV1Normalizer.identity_transform())

    left = encoder.encode(_observation(left_payload))
    right = encoder.encode(_observation(right_payload))

    np.testing.assert_array_equal(left.vector, right.vector)
    assert left.logical_state_hash == right.logical_state_hash


def test_rerank_order_change_changes_passage_state():
    before_payload = _payload()
    after_payload = _payload(reranked=True)
    encoder = _encoder(StateV1Normalizer.identity_transform())

    before = encoder.encode(_observation(before_payload, before_payload))
    after = encoder.encode(_observation(after_payload, before_payload))

    assert before.vector.shape == (STATE_V1_DIM,)
    assert before.vector.dtype == np.float32
    assert before.logical_state_hash != after.logical_state_hash
    assert not np.array_equal(before.vector, after.vector)


def test_streaming_normalizer_is_finite_and_round_trips(tmp_path):
    first = StateV1Blocks.zeros()
    second = StateV1Blocks.zeros()
    first.question_structured[0] = 1.0
    second.question_structured[0] = 3.0
    normalizer = StateV1Normalizer.fit([first, second])
    transformed = normalizer.transform(second)
    assert np.isfinite(transformed.question_structured).all()

    path = tmp_path / "normalizer.json"
    normalizer.save(path)
    loaded = StateV1Normalizer.load(path)
    assert loaded.identity == normalizer.identity
    np.testing.assert_array_equal(
        loaded.transform(second).question_structured,
        transformed.question_structured,
    )
