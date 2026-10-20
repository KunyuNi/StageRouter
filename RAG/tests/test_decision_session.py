from __future__ import annotations

from pathlib import Path

import pytest

from RAG.rl.decision_session import RAGDecisionSession
from RAG.rl.env_wrapper import RAGEnvironmentWrapper
from RAG.rl.trajectory_buffer import TrajectoryBuffer, TrajectoryRecord
from RAG.rl.transition_cache import TransitionCache


def _operators(calls: list[str]):
    def retrieval1(payload, **_kwargs):
        calls.append("retrieval1")
        output = payload.snapshot()
        output.context = ["Alice won"]
        output.metadata["retrieval1"] = {
            "query": payload.query_1 or payload.question,
            "index_identity": "index-a",
            "results": [
                {
                    "chunk_id": "c1",
                    "text": "Alice won",
                    "score": 1.0,
                    "rank": 1,
                    "metadata": {},
                }
            ],
        }
        output.cost_tracker.record_embedding_call(
            "retrieval1", "text-embedding-v3", 3, 1.0
        )
        return output

    def answer(payload, **_kwargs):
        calls.append("context_question_to_answer")
        output = payload.snapshot()
        output.answer = "Alice"
        output.cost_tracker.record_llm_call(
            "context_question_to_answer", "qwen3.5-flash", 2, 1, 1.0
        )
        return output

    def evaluate(payload, **_kwargs):
        calls.append("qa_evaluation")
        output = payload.snapshot()
        output.evaluation = {"f1_score": 1.0, "answer": output.answer}
        return output

    return {
        "retrieval1": retrieval1,
        ("context_question_to_answer", 2): answer,
        "qa_evaluation": evaluate,
    }


def _item() -> dict[str, object]:
    return {
        "question_id": 7,
        "query": "Who won?",
        "answer": "Alice",
        "question_type": "inference_query",
    }


def test_skip_decisions_keep_tier_zero_transitions_and_mandatory_retrieval(
    tmp_path: Path,
):
    calls: list[str] = []
    wrapper = RAGEnvironmentWrapper(operators=_operators(calls))
    cache = TransitionCache(tmp_path / "dev.db", "workflow-test")
    session = RAGDecisionSession.start(item=_item(), env=wrapper, cache=cache)

    assert session.observation().decision_idx == 0
    first = session.apply_action(3)
    assert first.physical_stages == (0, 1)
    assert first.bundled_cost > 0.0
    assert calls == ["retrieval1"]

    second = session.apply_action(3)
    assert second.physical_stages == (2,)
    assert second.bundled_cost == 0.0
    assert calls == ["retrieval1"]

    third = session.apply_action(3)
    assert third.physical_stages == (4,)
    assert third.bundled_cost == 0.0
    assert calls == ["retrieval1"]

    fourth = session.apply_action(1)
    assert fourth.physical_stages == (5,)
    assert fourth.bundled_cost > 0.0
    assert calls == ["retrieval1", "context_question_to_answer"]

    result = session.finalize()
    assert calls == ["retrieval1", "context_question_to_answer", "qa_evaluation"]
    assert result["actions"] == [3, 3, 3, 1]
    assert result["route_key"] == "3-3-3-1"
    assert result["f1_score"] == 1.0
    assert len(result["stage_costs"]) == 7
    assert len(result["bundled_decision_costs"]) == 4
    assert result["bundled_decision_costs"][0] == pytest.approx(
        result["stage_costs"][0] + result["stage_costs"][1]
    )
    assert result["bundled_decision_costs"][1] == pytest.approx(
        result["stage_costs"][2] + result["stage_costs"][3]
    )
    cache.close()


def test_replay_restores_every_skip_and_operator_transition_from_cache(tmp_path: Path):
    calls: list[str] = []
    wrapper = RAGEnvironmentWrapper(operators=_operators(calls))
    cache = TransitionCache(tmp_path / "dev.db", "workflow-test")

    first = RAGDecisionSession.start(item=_item(), env=wrapper, cache=cache)
    for action in (3, 3, 3, 1):
        first.apply_action(action)
    first.finalize()
    initial_calls = tuple(calls)

    replay = RAGDecisionSession.start(item=_item(), env=wrapper, cache=cache)
    step_hits = [replay.apply_action(action).cache_hits for action in (3, 3, 3, 1)]
    result = replay.finalize()

    assert step_hits == [(True, True), (True,), (True,), (True,)]
    assert result["cache_hits"] == [True, True, True, False, True, True, True]
    assert tuple(calls) == initial_calls
    cache.close()


def test_answer_skip_is_rejected_before_stage_execution(tmp_path: Path):
    calls: list[str] = []
    wrapper = RAGEnvironmentWrapper(operators=_operators(calls))
    cache = TransitionCache(tmp_path / "dev.db", "workflow-test")
    session = RAGDecisionSession.start(item=_item(), env=wrapper, cache=cache)
    for action in (3, 3, 3):
        session.apply_action(action)

    with pytest.raises(ValueError, match="does not allow SKIP"):
        session.apply_action(3)

    assert calls == ["retrieval1"]
    cache.close()


def _trajectory(split: str, answer: str) -> TrajectoryRecord:
    return TrajectoryRecord(
        question_id=7,
        split=split,
        route_name="route",
        route_key="3-3-3-1",
        actions=[3, 3, 3, 1],
        f1_score=1.0,
        reward=1.0,
        answer=answer,
        ground_truth="Alice",
        operator_version=1,
        workflow_identity="workflow-a",
    )


def test_trajectory_lookup_includes_split_in_complete_route_identity(tmp_path: Path):
    buffer = TrajectoryBuffer(
        tmp_path / "routes.json",
        workflow_identity="workflow-a",
        operator_version=1,
    )
    buffer.upsert_by_question_route(_trajectory("train", "train answer"))
    buffer.upsert_by_question_route(_trajectory("test", "test answer"))

    train = buffer.get_exact("train", 7, "3-3-3-1", 1, "workflow-a")
    test = buffer.get_exact("test", 7, "3-3-3-1", 1, "workflow-a")

    assert len(buffer) == 2
    assert train is not None and train.answer == "train answer"
    assert test is not None and test.answer == "test answer"
    assert buffer.get_exact("dev", 7, "3-3-3-1", 1, "workflow-a") is None

