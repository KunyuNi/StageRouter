from __future__ import annotations

import numpy as np
import pytest

from RAG.env._shared.state_payload import StatePayload
from RAG.rl.decision_session import DecisionObservation
from RAG.rl.state_v1_features import (
    DecisionCostPriors,
    availability_features,
    build_cost_history,
    encode_question_type,
    transition_delta_features,
    difficulty_context_features,
    retrieval_summary_features,
)
from RAG.rl.state_v1_schema import (
    STATE_V1_DIM,
    STATE_V1_SCHEMA,
    StateV1Blocks,
    legal_action_mask,
)


def test_state_v1_schema_is_exactly_9516_values():
    assert STATE_V1_DIM == 9516
    assert STATE_V1_SCHEMA.flat_sizes == {
        "question_semantic": 1024,
        "question_structured": 24,
        "question_type": 5,
        "query_semantics": 2048,
        "query_features": 16,
        "passage_semantics": 6144,
        "passage_features": 120,
        "passage_mask": 6,
        "retrieval_summary": 32,
        "transition_delta": 24,
        "route_control": 32,
        "cost_history": 17,
        "availability": 8,
        "difficulty_context": 16,
    }
    assert sum(STATE_V1_SCHEMA.flat_sizes.values()) == STATE_V1_DIM

    blocks = StateV1Blocks.zeros()
    vector = STATE_V1_SCHEMA.flatten(blocks)
    assert vector.shape == (STATE_V1_DIM,)
    assert vector.dtype == np.float32
    rebuilt = STATE_V1_SCHEMA.split(vector)
    np.testing.assert_array_equal(
        rebuilt.passage_semantics, blocks.passage_semantics
    )


def test_schema_rejects_wrong_shape_instead_of_padding():
    blocks = StateV1Blocks.zeros()
    blocks.question_semantic = np.zeros(1023, dtype=np.float32)
    with pytest.raises(ValueError, match="question_semantic"):
        STATE_V1_SCHEMA.flatten(blocks)


def test_legal_masks_do_not_encode_offline_support():
    np.testing.assert_array_equal(legal_action_mask(2), [1, 1, 1, 1])
    np.testing.assert_array_equal(legal_action_mask(3), [1, 1, 1, 0])


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("inference_query", [1, 0, 0, 0, 0]),
        ("comparison_query", [0, 1, 0, 0, 0]),
        ("temporal_query", [0, 0, 1, 0, 0]),
        ("null_query", [0, 0, 0, 1, 0]),
        ("bridge", [0, 0, 0, 0, 1]),
    ],
)
def test_question_type_normalizes_dataset_values_and_unknown(raw, want):
    np.testing.assert_array_equal(encode_question_type(raw), want)


def _observation(
    payload: StatePayload,
    *,
    decision_idx: int,
    previous: StatePayload | None,
    actions: tuple[int, ...],
    costs: tuple[float, ...],
) -> DecisionObservation:
    return DecisionObservation(
        question_id=7,
        decision_idx=decision_idx,
        payload=payload,
        previous_decision_payload=previous,
        action_history=actions,
        bundled_cost_history=costs,
        physical_stage_costs=(0.0,) * 7,
        physical_stage_latencies_ms=(0.0,) * 7,
    )


def test_skip_and_non_skip_artifacts_control_availability_and_delta():
    initial = StatePayload(
        question_id=7,
        question="Which city won in 2020?",
        metadata={"question_type": "temporal_query"},
    )
    after_retrieval1 = initial.snapshot()
    after_retrieval1.context = ["Alpha won in 2020"]
    after_retrieval1.metadata["retrieval1"] = {
        "results": [
            {
                "chunk_id": "c1",
                "text": "Alpha won in 2020",
                "rank": 1,
                "score": 0.9,
                "metadata": {"source": "news"},
            }
        ]
    }

    skipped = _observation(
        after_retrieval1,
        decision_idx=1,
        previous=initial,
        actions=(3,),
        costs=(0.0001,),
    )
    np.testing.assert_array_equal(
        availability_features(skipped), [0, 1, 0, 0, 1, 0, 0, 1]
    )
    skipped_delta = transition_delta_features(skipped)
    assert skipped_delta.shape == (24,)
    assert skipped_delta[-2:].tolist() == [1.0, 1.0]

    reranked = after_retrieval1.snapshot()
    reranked.metadata["rerank"] = {
        "results": [
            {
                "chunk_id": "c1",
                "text": "Alpha won in 2020",
                "rerank_rank": 1,
                "rerank_score": 95,
            }
        ]
    }
    at_answer = _observation(
        reranked,
        decision_idx=3,
        previous=after_retrieval1,
        actions=(3, 3, 0),
        costs=(0.0001, 0.0, 0.0002),
    )
    np.testing.assert_array_equal(
        availability_features(at_answer), [0, 1, 0, 0, 1, 1, 1, 1]
    )
    assert transition_delta_features(at_answer)[-2:].tolist() == [0.0, 1.0]


def test_cost_history_exposes_only_completed_costs_and_train_priors():
    payload = StatePayload(question_id=7, question="Who won?")
    payload.cost_tracker.total_input_tokens = 12
    payload.cost_tracker.total_output_tokens = 3
    observation = _observation(
        payload,
        decision_idx=2,
        previous=payload,
        actions=(0, 3),
        costs=(0.001, 0.0),
    )
    priors = DecisionCostPriors(
        means=(
            (0.1, 0.2, 0.3, 0.4),
            (0.5, 0.6, 0.7, 0.8),
            (0.0, 0.0002, 0.0001, 0.0003),
            (0.4, 0.5, 0.6, 0.0),
        ),
        identity="train-priors",
    )
    values = build_cost_history(observation, priors, c_ref=0.01)
    assert values.shape == (17,)
    np.testing.assert_allclose(values[:4], [0.1, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(values[8:12], [0.0, 0.02, 0.01, 0.03])
    np.testing.assert_allclose(values[12:16], [0.0, 0.02, 0.01, 0.03])
    assert values[-1] == pytest.approx(0.1)


def test_retrieval_duplicate_and_score_gap_use_raw_retrieval_artifacts():
    payload = StatePayload(
        question_id=7,
        question="Who won?",
        context=["Alpha won", "Beta lost"],
        metadata={
            "retrieval1": {
                "results": [
                    {"chunk_id": "same", "text": "Alpha won", "rank": 1, "score": 0.9},
                    {"chunk_id": "r1", "text": "Beta lost", "rank": 2, "score": 0.8},
                ]
            },
            "retrieval2": {
                "results": [
                    {"chunk_id": "same", "text": "Alpha won", "rank": 1, "score": 0.7},
                    {"chunk_id": "r2", "text": "Other", "rank": 2, "score": 0.2},
                ]
            },
        },
    )
    observation = _observation(
        payload, decision_idx=2, previous=payload, actions=(0, 0), costs=(0.1, 0.2)
    )
    summary = retrieval_summary_features(observation)
    assert summary[23] > 0.0
    difficulty = difficulty_context_features(observation)
    assert difficulty[6] == pytest.approx(0.9, abs=1e-6)
