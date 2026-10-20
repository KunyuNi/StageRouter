from __future__ import annotations

import json
from pathlib import Path

from RAG.env._shared.state_payload import StatePayload
from RAG.rl.dev_evaluator import (
    TrajectoryBufferWriter,
    build_safety_union,
    evaluate_query,
)
from RAG.rl.env_wrapper import RAGEnvironmentWrapper
from RAG.rl.failure_queue import EvaluationFailureQueue
from RAG.rl.trajectory_buffer import TrajectoryBuffer, TrajectoryRecord
from RAG.rl.transition_cache import TransitionCache


class FixedModel:
    def select_action(self, observation):
        return 0


def _operators(calls: list[str], *, fail_answer: bool = False):
    def rewrite1(payload, **_kwargs):
        calls.append("rewrite1")
        output = payload.snapshot()
        output.query_1 = "winner"
        return output

    def retrieval1(payload, **_kwargs):
        calls.append("retrieval1")
        output = payload.snapshot()
        output.context = ["Alice won"]
        output.metadata["retrieval1"] = {
            "index_identity": "index-a",
            "results": [{"chunk_id": "c1", "text": "Alice won", "rank": 1}],
        }
        return output

    def rewrite2(payload, **_kwargs):
        calls.append("rewrite2")
        output = payload.snapshot()
        output.query_2 = "Alice winner"
        return output

    def retrieval2(payload, **_kwargs):
        calls.append("retrieval2")
        output = payload.snapshot()
        output.metadata["retrieval2"] = {"results": []}
        return output

    def rerank(payload, **_kwargs):
        calls.append("rerank")
        output = payload.snapshot()
        output.metadata["rerank"] = {
            "results": [
                {
                    "chunk_id": "c1",
                    "text": "Alice won",
                    "rerank_rank": 1,
                    "rerank_score": 100,
                }
            ]
        }
        return output

    def answer(payload, **_kwargs):
        calls.append("answer")
        if fail_answer:
            raise TimeoutError("temporary answer timeout")
        output = payload.snapshot()
        output.answer = "Alice"
        return output

    def evaluate(payload, **_kwargs):
        calls.append("evaluate")
        output = payload.snapshot()
        output.evaluation = {"f1_score": 1.0}
        return output

    return {
        ("question_to_query", 1): rewrite1,
        "retrieval1": retrieval1,
        ("context_question_to_query", 1): rewrite2,
        "retrieval2": retrieval2,
        ("rerank", 1): rerank,
        ("context_question_to_answer", 1): answer,
        "qa_evaluation": evaluate,
    }


def _item():
    return {"question_id": 7, "query": "Who won?", "answer": "Alice"}


def _record(result) -> TrajectoryRecord:
    return TrajectoryRecord(
        question_id=result["question_id"],
        split="test",
        route_name="policy",
        route_key=result["route_key"],
        actions=result["actions"],
        f1_score=result["f1_score"],
        reward=result["reward"],
        answer=result["answer"],
        ground_truth=result["ground_truth"],
        total_cost=result["total_cost"],
        stage_costs=result["stage_costs"],
        decision_stage_costs=result["decision_stage_costs"],
        fixed_stage_costs=result["fixed_stage_costs"],
        total_latency_ms=result["total_latency_ms"],
        stage_latencies_ms=result["stage_latencies_ms"],
        decision_stage_latencies_ms=result["decision_stage_latencies_ms"],
        fixed_stage_latencies_ms=result["fixed_stage_latencies_ms"],
        operator_version=1,
        workflow_identity="workflow-test",
        index_identity="index-a",
    )


def test_transition_hit_then_route_hit_calls_no_operator_or_evaluator(tmp_path: Path):
    calls: list[str] = []
    env = RAGEnvironmentWrapper(operators=_operators(calls))
    cache = TransitionCache(tmp_path / "dev.db", "workflow-test")
    seeded = env.rollout_with_cache(_item(), [0, 0, 0, 0], cache)
    buffer = TrajectoryBuffer(tmp_path / "dev.json", "workflow-test")
    buffer.upsert_by_question_route(_record(seeded))
    writer = TrajectoryBufferWriter(buffer, autosave_every=10)
    queue = EvaluationFailureQueue(
        tmp_path / "failed.json",
        split="test",
        policy_id="S0/scalar",
        workflow_identity="workflow-test",
        operator_version=1,
    )
    calls.clear()

    result = evaluate_query(
        model=FixedModel(),
        policy=None,
        item=_item(),
        env=env,
        transition_cache=cache,
        trajectory_buffer=buffer,
        writer=writer,
        failure_queue=queue,
        operator_version=1,
        workflow_identity="workflow-test",
    )

    assert result is not None
    assert result.transition_hits == 6
    assert result.route_hit is True
    assert result.operator_calls == 0
    assert result.qa_evaluation_calls == 0
    assert calls == []
    cache.close()


def test_route_miss_writes_only_complete_trajectory(tmp_path: Path):
    calls: list[str] = []
    env = RAGEnvironmentWrapper(operators=_operators(calls))
    cache = TransitionCache(tmp_path / "dev.db", "workflow-test")
    buffer = TrajectoryBuffer(tmp_path / "dev.json", "workflow-test")
    writer = TrajectoryBufferWriter(buffer, autosave_every=10)
    queue = EvaluationFailureQueue(
        tmp_path / "failed.json",
        split="test",
        policy_id="S0/scalar",
        workflow_identity="workflow-test",
        operator_version=1,
    )
    result = evaluate_query(
        model=FixedModel(),
        policy=None,
        item=_item(),
        env=env,
        transition_cache=cache,
        trajectory_buffer=buffer,
        writer=writer,
        failure_queue=queue,
        operator_version=1,
        workflow_identity="workflow-test",
    )
    assert result is not None and result.route_hit is False
    assert buffer.get_exact("test", 7, "0-0-0-0", 1, "workflow-test") is not None
    assert calls[-1] == "evaluate"
    cache.close()


def test_failure_keeps_prefix_cache_but_never_writes_partial_route(tmp_path: Path):
    calls: list[str] = []
    env = RAGEnvironmentWrapper(operators=_operators(calls, fail_answer=True))
    cache = TransitionCache(tmp_path / "dev.db", "workflow-test")
    buffer = TrajectoryBuffer(tmp_path / "dev.json", "workflow-test")
    writer = TrajectoryBufferWriter(buffer, autosave_every=10)
    queue = EvaluationFailureQueue(
        tmp_path / "failed.json",
        split="test",
        policy_id="S0/scalar",
        workflow_identity="workflow-test",
        operator_version=1,
    )
    result = evaluate_query(
        model=FixedModel(),
        policy=None,
        item=_item(),
        env=env,
        transition_cache=cache,
        trajectory_buffer=buffer,
        writer=writer,
        failure_queue=queue,
        operator_version=1,
        workflow_identity="workflow-test",
    )
    assert result is None
    assert len(cache) == 5
    assert buffer.get_exact("test", 7, "0-0-0-0", 1, "workflow-test") is None
    assert queue.unresolved_count == 1
    cache.close()


def _legacy_row(qid: int):
    return {
        "split": "test",
        "question_id": qid,
        "route_name": "all_t1",
        "route_key": "0-0-0-0",
        "actions": [0, 0, 0, 0],
        "operator_version": 1,
        "workflow_identity": "workflow-test",
    }


def test_completed_v1_dev_queue_migrates_without_losing_history(tmp_path: Path):
    path = tmp_path / "failed.json"
    path.write_text(
        json.dumps(
            {
                "version": "rag_failed_pair_queue_v1",
                "workflow_identity": "workflow-test",
                "operator_version": 1,
                "split": "train",
                "unresolved": [],
                "resolved": [_legacy_row(1), _legacy_row(2)],
            }
        ),
        encoding="utf-8",
    )
    queue = EvaluationFailureQueue(path, split="test", policy_id="S0/scalar")
    assert queue.counts() == (0, 2)
    assert queue.payload["version"] == "rag_policy_failure_queue_v2"
    assert queue.payload["split"] == "test"
    assert {row["policy_id"] for row in queue.payload["resolved"]} == {"baseline"}


def test_v1_dev_queue_with_unresolved_rows_refuses_migration(tmp_path: Path):
    path = tmp_path / "failed.json"
    path.write_text(
        json.dumps(
            {
                "version": "rag_failed_pair_queue_v1",
                "workflow_identity": "workflow-test",
                "operator_version": 1,
                "split": "train",
                "unresolved": [_legacy_row(1)],
                "resolved": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        EvaluationFailureQueue(path, split="test", policy_id="S0/scalar")
    except ValueError as exc:
        assert "resolve legacy unresolved" in str(exc)
    else:
        raise AssertionError("migration should have refused unresolved v1 rows")


def test_safety_union_excludes_successful_rows_from_every_policy():
    union = build_safety_union({"p1": {7}, "p2": {9}})
    assert union == {7, 9}
    policy_qids = {"p1": {7, 8, 9}, "p2": {7, 8, 9}}
    assert {name: qids - union for name, qids in policy_qids.items()} == {
        "p1": {8},
        "p2": {8},
    }
