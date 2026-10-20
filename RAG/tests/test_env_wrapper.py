from __future__ import annotations

from pathlib import Path

import pytest

from RAG.env._shared.state_payload import StatePayload
from RAG.rl.env_wrapper import RAGEnvironmentWrapper
from RAG.rl.transition_cache import TransitionCache


def _fake_operators(calls: list[str]):
    def rewrite(payload, **_kwargs):
        calls.append("question_to_query")
        output = payload.snapshot()
        output.query_1 = "first hop"
        output.cost_tracker.record_llm_call("question_to_query", "fake", 2, 1, 1)
        return output

    def retrieval1(payload, **_kwargs):
        calls.append("retrieval1")
        output = payload.snapshot()
        output.context = ["Alice won"]
        output.metadata["retrieval1"] = {
            "query": payload.query_1,
            "index_identity": "index-a",
            "results": [{"chunk_id": "c1", "text": "Alice won", "score": 1.0, "rank": 1, "metadata": {}}],
        }
        output.cost_tracker.record_embedding_call("retrieval1", "text-embedding-v3", 3, 1)
        return output

    def second_rewrite(payload, **_kwargs):
        calls.append("context_question_to_query")
        output = payload.snapshot()
        output.query_2 = "second hop"
        output.cost_tracker.record_llm_call("context_question_to_query", "fake", 2, 1, 1)
        return output

    def retrieval2(payload, **_kwargs):
        calls.append("retrieval2")
        output = payload.snapshot()
        output.context = [*payload.context, "The winner was Alice"]
        output.metadata["retrieval2"] = {
            "query": payload.query_2,
            "index_identity": "index-a",
            "results": [{"chunk_id": "c2", "text": "The winner was Alice", "score": 0.9, "rank": 1, "metadata": {}}],
        }
        output.cost_tracker.record_embedding_call("retrieval2", "text-embedding-v3", 3, 1)
        return output

    def rerank(payload, **_kwargs):
        calls.append("rerank")
        output = payload.snapshot()
        output.context = ["Alice won"]
        output.metadata["rerank"] = {"results": [{"chunk_id": "c1", "text": "Alice won", "rerank_score": 1.0, "rerank_rank": 1}]}
        output.cost_tracker.record_llm_call("rerank", "fake", 2, 1, 1)
        return output

    def answer(payload, **_kwargs):
        calls.append("context_question_to_answer")
        output = payload.snapshot()
        output.answer = "Alice"
        output.cost_tracker.record_llm_call("context_question_to_answer", "fake", 2, 1, 1)
        return output

    def evaluate(payload, **_kwargs):
        calls.append("qa_evaluation")
        output = payload.snapshot()
        output.evaluation = {"f1_score": 1.0, "answer": output.answer}
        return output

    return {
        ("question_to_query", 2): rewrite,
        "retrieval1": retrieval1,
        ("context_question_to_query", 2): second_rewrite,
        "retrieval2": retrieval2,
        ("rerank", 2): rerank,
        ("context_question_to_answer", 2): answer,
        "qa_evaluation": evaluate,
    }


def test_rollout_runs_seven_physical_stages_and_accounts_costs(tmp_path: Path):
    calls: list[str] = []
    wrapper = RAGEnvironmentWrapper(operators=_fake_operators(calls))
    cache = TransitionCache(tmp_path / "new1.db", "workflow-a")

    result = wrapper.rollout_with_cache(
        item={"question_id": 7, "query": "Who won?", "answer": "Alice"},
        actions=[1, 1, 1, 1],
        cache=cache,
    )

    assert result is not None
    assert calls == [
        "question_to_query", "retrieval1", "context_question_to_query",
        "retrieval2", "rerank", "context_question_to_answer", "qa_evaluation",
    ]
    assert result["actions"] == [1, 1, 1, 1]
    assert len(result["stage_costs"]) == 7
    assert len(result["decision_stage_costs"]) == 4
    assert len(result["fixed_stage_costs"]) == 7
    assert sum(result["stage_costs"]) == pytest.approx(result["total_cost"])
    assert sum(result["decision_stage_costs"]) + sum(result["fixed_stage_costs"]) == pytest.approx(result["total_cost"])
    assert result["f1_score"] == 1.0
    assert result["index_identity"] == "index-a"

    replay = wrapper.rollout_with_cache(
        item={"question_id": 7, "query": "Who won?", "answer": "Alice"},
        actions=[1, 1, 1, 1],
        cache=cache,
    )
    assert replay is not None
    assert all(replay["cache_hits"])
    assert calls == [
        "question_to_query", "retrieval1", "context_question_to_query",
        "retrieval2", "rerank", "context_question_to_answer", "qa_evaluation",
    ]
    cache.close()


def test_second_hop_skip_bypasses_rewrite_and_retrieval2(tmp_path: Path):
    calls: list[str] = []
    wrapper = RAGEnvironmentWrapper(operators=_fake_operators(calls))
    cache = TransitionCache(tmp_path / "new1.db", "workflow-a")

    result = wrapper.rollout_with_cache(
        item={"question_id": 8, "query": "Who won?", "answer": "Alice"},
        actions=[1, 3, 1, 1],
        cache=cache,
    )

    assert result is not None
    assert calls == [
        "question_to_query", "retrieval1", "rerank",
        "context_question_to_answer", "qa_evaluation",
    ]
    assert result["stage_costs"][2:4] == [0.0, 0.0]
    cache.close()


def test_t3_action_invokes_t3_operators_not_skip(tmp_path: Path):
    calls: list[str] = []
    operators = _fake_operators(calls)
    operators.update(
        {
            (stage, 3): operator
            for key, operator in list(operators.items())
            if isinstance(key, tuple) and len(key) == 2
            for stage, tier in [key]
            if tier == 2
        }
    )
    wrapper = RAGEnvironmentWrapper(operators=operators)
    cache = TransitionCache(tmp_path / "new1.db", "workflow-a")

    result = wrapper.rollout_with_cache(
        item={"question_id": 10, "query": "Who won?", "answer": "Alice"},
        actions=[2, 2, 2, 2],
        cache=cache,
    )

    assert result is not None
    assert calls == [
        "question_to_query", "retrieval1", "context_question_to_query",
        "retrieval2", "rerank", "context_question_to_answer", "qa_evaluation",
    ]
    assert result["stage_costs"][0] == 0.0
    cache.close()


def test_answer_generation_cannot_skip(tmp_path: Path):
    wrapper = RAGEnvironmentWrapper(operators=_fake_operators([]))
    cache = TransitionCache(tmp_path / "new1.db", "workflow-a")
    with pytest.raises(ValueError, match="does not allow SKIP"):
        wrapper.rollout_with_cache(
            item={"question_id": 9, "query": "Who won?", "answer": "Alice"},
            actions=[1, 1, 1, 3],
            cache=cache,
        )
    cache.close()


def test_operator_failure_carries_physical_and_decision_stage_context(tmp_path: Path):
    operators = _fake_operators([])

    def fail_rewrite(_payload, **_kwargs):
        raise ValueError("invalid operator JSON")

    operators[("question_to_query", 2)] = fail_rewrite
    wrapper = RAGEnvironmentWrapper(operators=operators)
    cache = TransitionCache(tmp_path / "new1.db", "workflow-a")

    with pytest.raises(ValueError, match="invalid operator JSON") as caught:
        wrapper.rollout_with_cache(
            item={"question_id": 11, "query": "Who won?", "answer": "Alice"},
            actions=[1, 1, 1, 1],
            cache=cache,
        )

    assert caught.value.stage_index == 0
    assert caught.value.stage_name == "question_to_query"
    assert caught.value.decision_stage_index == 0
    assert caught.value.tier == 2
    assert caught.value.tier_name == "T2"
    cache.close()
