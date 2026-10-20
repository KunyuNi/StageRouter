from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

import RAG.rl.fixed_route_collector as fixed_route_collector_module
import RAG.rl.transition_cache as transition_cache_module
from RAG.pipeline.value_pruned_routes import routes_from_cost_shares
from RAG.rl.fixed_route_collector import FixedRouteCollector
from RAG.rl.transition_cache import TransitionKey

BASE_PROFILE, _ = routes_from_cost_shares((0.4, 0.05, 0.4, 0.15))


class CountingWrapper:
    def __init__(self) -> None:
        self.calls = 0

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        self.calls += 1
        return {
            "question_id": int(item["question_id"]),
            "actions": list(actions),
            "f1_score": 0.75,
            "reward": 0.75,
            "answer": "a",
            "ground_truth": item["answer"],
            "stage_costs": [0.1] + [0.0] * 6,
            "decision_stage_costs": [0.1, 0.0, 0.0, 0.0],
            "fixed_stage_costs": [0.0] * 7,
            "stage_latencies_ms": [1.0] + [0.0] * 6,
            "decision_stage_latencies_ms": [1.0, 0.0, 0.0, 0.0],
            "fixed_stage_latencies_ms": [0.0] * 7,
            "total_cost": 0.1,
            "total_latency_ms": 1.0,
            "metadata": {},
        }


class ParallelWrapper(CountingWrapper):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        try:
            return super().rollout_with_cache(item, actions, cache, **kwargs)
        finally:
            with self._lock:
                self.active -= 1


class ContentSafetyWrapper:
    def __init__(self) -> None:
        self.calls = 0

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        self.calls += 1
        raise RuntimeError(
            "HTTP 403 data_inspection_failed: Input/Output data may contain "
            "inappropriate content"
        )


class SafetyAfterFirstRouteWrapper(CountingWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.calls_by_qid: dict[int, int] = {}

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        qid = int(item["question_id"])
        self.calls_by_qid[qid] = self.calls_by_qid.get(qid, 0) + 1
        if qid == 19 and list(actions) == [1, 1, 1, 1]:
            self.calls += 1
            raise RuntimeError(
                "HTTP 400 data_inspection_failed: Input/Output data may contain "
                "inappropriate content"
            )
        return super().rollout_with_cache(item, actions, cache, **kwargs)


class SharedTransitionWrapper(CountingWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.physical_calls: dict[int, int] = {}

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        qid = int(item["question_id"])
        key = TransitionKey(
            qid,
            stage_idx=1,
            tier=0,
            input_hash="shared",
            operator_version=1,
        )
        if cache.lookup(key) is None:
            self.physical_calls[qid] = self.physical_calls.get(qid, 0) + 1
            cache.store(key, {"context": [f"evidence-{qid}"]}, source="test")
        return super().rollout_with_cache(item, actions, cache, **kwargs)


class ParallelSafetyWrapper(CountingWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.calls_by_qid: dict[int, int] = {}

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        qid = int(item["question_id"])
        self.calls_by_qid[qid] = self.calls_by_qid.get(qid, 0) + 1
        if qid == 2 and list(actions) == [2, 2, 2, 2]:
            self.calls += 1
            raise RuntimeError(
                "HTTP 400 data_inspection_failed: Input/Output data may contain "
                "inappropriate content"
            )
        return super().rollout_with_cache(item, actions, cache, **kwargs)


class OrdinaryErrorWrapper(CountingWrapper):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._failed = False

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        with self._lock:
            should_fail = not self._failed
            if should_fail:
                self._failed = True
                self.calls += 1
        if should_fail:
            raise RuntimeError("Connection error.")
        time.sleep(0.05)
        return super().rollout_with_cache(item, actions, cache, **kwargs)


class PersistentRateLimitWrapper(CountingWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.calls_by_qid: dict[int, int] = {}

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        qid = int(item["question_id"])
        self.calls_by_qid[qid] = self.calls_by_qid.get(qid, 0) + 1
        if qid == 0:
            self.calls += 1
            raise RuntimeError("HTTP 429 limit_requests from text-embedding-v3")
        return super().rollout_with_cache(item, actions, cache, **kwargs)


class QuotaErrorWrapper(CountingWrapper):
    def rollout_with_cache(self, item, actions, cache, **kwargs):
        self.calls += 1
        raise RuntimeError("insufficient_quota: account balance is exhausted")


class LockedDatabaseWrapper(CountingWrapper):
    def rollout_with_cache(self, item, actions, cache, **kwargs):
        self.calls += 1
        raise sqlite3.OperationalError("database is locked")


class RetrievalUnavailableWrapper(CountingWrapper):
    def rollout_with_cache(self, item, actions, cache, **kwargs):
        self.calls += 1
        error = ConnectionRefusedError("Connection refused by retrieval service")
        error.stage_index = 1
        error.stage_name = "retrieval1"
        error.tier = 0
        error.tier_name = "fixed"
        raise error


class PersistentProgramErrorWrapper(CountingWrapper):
    def rollout_with_cache(self, item, actions, cache, **kwargs):
        self.calls += 1
        raise ValueError("invalid operator JSON schema")


class TransitionLookupWrapper(CountingWrapper):
    def __init__(self, key: TransitionKey) -> None:
        super().__init__()
        self.key = key

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        cache.lookup(self.key)
        return super().rollout_with_cache(item, actions, cache, **kwargs)


class InterruptDrainWrapper:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.four_started = threading.Event()
        self.release = threading.Event()

    def rollout_with_cache(self, item, actions, cache, **kwargs):
        with self._lock:
            self.calls += 1
            if self.calls == 4:
                self.four_started.set()
        assert self.release.wait(timeout=5.0)
        return {
            "question_id": int(item["question_id"]),
            "actions": list(actions),
            "f1_score": 0.75,
            "reward": 0.75,
            "answer": "a",
            "ground_truth": item["answer"],
            "stage_costs": [0.1] + [0.0] * 6,
            "decision_stage_costs": [0.1, 0.0, 0.0, 0.0],
            "fixed_stage_costs": [0.0] * 7,
            "stage_latencies_ms": [1.0] + [0.0] * 6,
            "decision_stage_latencies_ms": [1.0, 0.0, 0.0, 0.0],
            "fixed_stage_latencies_ms": [0.0] * 7,
            "total_cost": 0.1,
            "total_latency_ms": 1.0,
            "metadata": {},
        }


def counterfactual_collector(tmp_path, wrapper):
    return FixedRouteCollector(
        tmp_path,
        wrapper,
        identities="workflow-a",
        autosave_every=10,
        routes=BASE_PROFILE.route_map,
        route_order=BASE_PROFILE.route_order,
        route_profile_name=BASE_PROFILE.name,
        route_profile_sha256=BASE_PROFILE.sha256,
    )


def test_completed_route_is_reused_without_calling_wrapper_twice(tmp_path):
    wrapper = CountingWrapper()
    collector = FixedRouteCollector(
        tmp_path,
        wrapper,
        identities="workflow-a",
        autosave_every=10,
    )
    item = {
        "question_id": 9,
        "query": "q",
        "answer": "a",
        "question_type": "inference_query",
    }
    first = collector.collect_one(item, "all_t3")
    second = collector.collect_one(item, "all_t3")
    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert second["route_result_hit"] is True
    assert wrapper.calls == 1
    assert len(collector.trajectory_buffer.records) == 1


def test_collect_parallelizes_qids_but_commits_each_pair_once(tmp_path):
    wrapper = ParallelWrapper()
    collector = FixedRouteCollector(
        tmp_path,
        wrapper,
        identities="workflow-a",
        expected_routes=4,
        n_workers=4,
        max_inflight=8,
    )
    items = [
        {"question_id": qid, "query": f"q{qid}", "answer": "a"}
        for qid in range(4)
    ]

    manifest = collector.collect(items, route_names=("all_t3",))

    assert wrapper.max_active == 4
    assert wrapper.calls == 4
    assert len(collector.route_caches["all_t3"]) == 4
    assert len(collector.trajectory_buffer.records) == 4
    assert manifest["n_workers"] == 4
    assert manifest["max_inflight"] == 8
    assert manifest["scheduler"] == "route_major_qid_parallel_v1"


def test_routes_are_sequential_and_share_transition_cache(tmp_path):
    wrapper = SharedTransitionWrapper()
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=4, max_inflight=8
    )
    items = [
        {"question_id": qid, "query": f"q{qid}", "answer": "a"}
        for qid in range(4)
    ]

    collector.collect(items, route_names=("all_t3", "all_t2"))

    assert wrapper.physical_calls == {0: 1, 1: 1, 2: 1, 3: 1}
    assert wrapper.calls == 8


def test_parallel_safety_error_excludes_only_its_qid(tmp_path):
    wrapper = ParallelSafetyWrapper()
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=4, max_inflight=8
    )
    items = [
        {"question_id": qid, "query": f"q{qid}", "answer": "a"}
        for qid in range(6)
    ]

    collector.collect(items, route_names=("all_t3", "all_t2"))

    assert 2 in collector.excluded_query_ids
    assert wrapper.calls_by_qid[2] == 1
    assert all(wrapper.calls_by_qid[qid] == 2 for qid in (0, 1, 3, 4, 5))


def test_ordinary_pair_error_is_retried_and_collection_continues(tmp_path):
    wrapper = OrdinaryErrorWrapper()
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=4, max_inflight=8
    )
    items = [
        {"question_id": qid, "query": f"q{qid}", "answer": "a"}
        for qid in range(12)
    ]

    manifest = collector.collect(items, route_names=("all_t3",))

    assert manifest["status"] == "completed"
    assert manifest["stop_reason"] == ""
    assert collector.excluded_query_ids == set()
    assert len(collector.trajectory_buffer.records) == 12
    queue = json.loads((tmp_path / "new1_failed_pairs.json").read_text())
    assert queue["unresolved"] == []
    assert len(queue["resolved"]) == 1
    assert queue["resolved"][0]["attempts"] == 1
    assert queue["resolved"][0]["error_category"] == "infrastructure"
    assert queue["resolved"][0]["resolution"] == "success"
    persisted = json.loads((tmp_path / "new1_manifest.json").read_text())
    assert persisted["excluded_query_ids"] == []


def test_embedding_rate_limit_enters_unresolved_queue_without_stopping_other_pairs(
    tmp_path,
):
    wrapper = PersistentRateLimitWrapper()
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=2, max_inflight=4
    )
    items = [
        {"question_id": qid, "query": f"q{qid}", "answer": "a"}
        for qid in range(3)
    ]

    manifest = collector.collect(items, route_names=("all_t3",))

    assert manifest["status"] == "completed_with_failures"
    assert manifest["stop_reason"] == ""
    assert wrapper.calls_by_qid == {0: 2, 1: 1, 2: 1}
    assert {record.question_id for record in collector.trajectory_buffer.records} == {
        1,
        2,
    }
    queue = json.loads((tmp_path / "new1_failed_pairs.json").read_text())
    assert queue["resolved"] == []
    assert len(queue["unresolved"]) == 1
    failure = queue["unresolved"][0]
    assert failure["question_id"] == 0
    assert failure["route_name"] == "all_t3"
    assert failure["error_category"] == "rate_limit"
    assert failure["attempts"] == 2
    assert manifest["failed_pairs_unresolved"] == 1


def test_transient_sqlite_lock_enters_failure_queue(tmp_path):
    wrapper = LockedDatabaseWrapper()
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=1, max_inflight=1
    )

    manifest = collector.collect(
        [{"question_id": 41, "query": "q41", "answer": "a"}],
        route_names=("all_t3",),
    )

    assert manifest["status"] == "completed_with_failures"
    queue = json.loads((tmp_path / "new1_failed_pairs.json").read_text())
    assert queue["unresolved"][0]["error_category"] == "transient_storage"
    assert queue["unresolved"][0]["attempts"] == 2


def test_quota_error_is_fatal_and_never_enters_failure_queue(tmp_path):
    wrapper = QuotaErrorWrapper()
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=1, max_inflight=1
    )

    with pytest.raises(RuntimeError, match="insufficient_quota"):
        collector.collect(
            [{"question_id": 42, "query": "q42", "answer": "a"}],
            route_names=("all_t3",),
        )

    manifest = json.loads((tmp_path / "new1_manifest.json").read_text())
    assert manifest["status"] == "stopped"
    assert not (tmp_path / "new1_failed_pairs.json").exists()


def test_shared_retrieval_service_outage_is_fatal(tmp_path):
    wrapper = RetrievalUnavailableWrapper()
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=1, max_inflight=1
    )

    with pytest.raises(ConnectionRefusedError, match="retrieval service"):
        collector.collect(
            [{"question_id": 43, "query": "q43", "answer": "a"}],
            route_names=("all_t3",),
        )

    manifest = json.loads((tmp_path / "new1_manifest.json").read_text())
    assert manifest["status"] == "stopped"
    assert not (tmp_path / "new1_failed_pairs.json").exists()


def test_same_program_error_across_three_qids_stops_collection(tmp_path):
    wrapper = PersistentProgramErrorWrapper()
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=1, max_inflight=1
    )
    items = [
        {"question_id": qid, "query": f"q{qid}", "answer": "a"}
        for qid in range(3)
    ]

    with pytest.raises(RuntimeError, match="systemic pair failure"):
        collector.collect(items, route_names=("all_t3",))

    queue = json.loads((tmp_path / "new1_failed_pairs.json").read_text())
    assert len(queue["unresolved"]) == 3
    manifest = json.loads((tmp_path / "new1_manifest.json").read_text())
    assert manifest["status"] == "stopped"


def test_corrupt_transition_payload_is_fatal_without_route_error_record(tmp_path):
    key = TransitionKey(31, stage_idx=1, tier=0, input_hash="corrupt", operator_version=1)
    wrapper = TransitionLookupWrapper(key)
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=1, max_inflight=1
    )
    collector.transition_cache.store(key, {"context": ["valid"]}, source="test")
    with collector.transition_cache._lock:
        collector.transition_cache._conn.execute(
            "UPDATE transitions SET output_payload=? WHERE question_id=?",
            ('{"context": [', key.question_id),
        )
        collector.transition_cache._conn.commit()

    with pytest.raises(ValueError) as caught:
        collector.collect(
            [{"question_id": 31, "query": "q31", "answer": "a"}],
            route_names=("all_t3",),
        )

    assert caught.type is transition_cache_module.TransitionCacheIntegrityError
    persisted = json.loads((tmp_path / "new1_manifest.json").read_text())
    assert persisted["status"] == "stopped"
    assert "corrupt transition cache payload" in persisted["stop_reason"]
    assert not (tmp_path / "baselines" / "all_t3" / "results.jsonl").exists()


def test_sqlite_failure_is_fatal_without_route_error_record(tmp_path, monkeypatch):
    key = TransitionKey(32, stage_idx=1, tier=0, input_hash="sqlite", operator_version=1)
    wrapper = TransitionLookupWrapper(key)
    collector = FixedRouteCollector(
        tmp_path, wrapper, identities="workflow-a", n_workers=1, max_inflight=1
    )

    def fail_lookup(_key):
        raise sqlite3.DatabaseError("injected sqlite failure")

    monkeypatch.setattr(collector.transition_cache, "lookup", fail_lookup)

    with pytest.raises(sqlite3.DatabaseError, match="injected sqlite failure"):
        collector.collect(
            [{"question_id": 32, "query": "q32", "answer": "a"}],
            route_names=("all_t3",),
        )

    persisted = json.loads((tmp_path / "new1_manifest.json").read_text())
    assert persisted["status"] == "stopped"
    assert persisted["stop_reason"] == "injected sqlite failure"
    assert not (tmp_path / "baselines" / "all_t3" / "results.jsonl").exists()


def test_keyboard_interrupt_cancels_queued_and_commits_running_results(
    tmp_path, monkeypatch
):
    wrapper = InterruptDrainWrapper()
    collector = FixedRouteCollector(
        tmp_path,
        wrapper,
        identities="workflow-a",
        n_workers=4,
        max_inflight=8,
    )
    items = [
        {"question_id": qid, "query": f"q{qid}", "answer": "a"}
        for qid in range(8)
    ]
    real_wait = fixed_route_collector_module.wait
    drain_started = threading.Event()

    def interrupt_then_drain(futures, *args, **kwargs):
        if not drain_started.is_set():
            assert wrapper.four_started.wait(timeout=5.0)

            def release_if_collector_does_not_drain():
                drain_started.wait(timeout=1.0)
                wrapper.release.set()

            threading.Thread(
                target=release_if_collector_does_not_drain, daemon=True
            ).start()
            raise KeyboardInterrupt
        wrapper.release.set()
        return real_wait(futures, *args, **kwargs)

    wait_calls = 0

    def patched_wait(futures, *args, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 2:
            drain_started.set()
        return interrupt_then_drain(futures, *args, **kwargs)

    monkeypatch.setattr(fixed_route_collector_module, "wait", patched_wait)

    with pytest.raises(KeyboardInterrupt):
        collector.collect(items, route_names=("all_t3",))

    persisted = json.loads((tmp_path / "new1_manifest.json").read_text())
    route_rows = [
        json.loads(line)
        for line in (
            tmp_path / "baselines" / "all_t3" / "results.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert wrapper.calls == 4
    assert len(route_rows) == 4
    assert len(collector.trajectory_buffer.records) == 4
    assert persisted["completed_routes"] == 4
    assert persisted["status"] == "stopped"
    assert persisted["stop_reason"] == "KeyboardInterrupt"


def test_collect_deduplicates_repeated_exact_pair_before_scheduling(tmp_path):
    wrapper = CountingWrapper()
    collector = FixedRouteCollector(
        tmp_path,
        wrapper,
        identities="workflow-a",
        expected_routes=1,
        n_workers=4,
        max_inflight=8,
    )
    item = {"question_id": 23, "query": "q23", "answer": "a"}

    collector.collect([item, dict(item)], route_names=("all_t3",))

    rows = [
        json.loads(line)
        for line in (
            tmp_path / "baselines" / "all_t3" / "results.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert wrapper.calls == 1
    assert len(collector.route_caches["all_t3"]) == 1
    assert [row["qid"] for row in rows] == [23]
    assert len(collector.trajectory_buffer.records) == 1


def test_injected_counterfactual_route_is_reused_with_profile_actions(tmp_path):
    wrapper = CountingWrapper()
    collector = counterfactual_collector(tmp_path, wrapper)
    item = {
        "question_id": 17,
        "query": "q",
        "answer": "a",
        "question_type": "inference_query",
    }

    first = collector.collect_one(item, "ablate_t3_s3")
    second = collector.collect_one(item, "ablate_t3_s3")

    assert first["status"] == "ok"
    assert first["actions"] == [2, 2, 3, 2]
    assert second["route_result_hit"] is True
    assert wrapper.calls == 1


def test_content_safety_failure_excludes_later_routes_for_the_query(tmp_path):
    wrapper = ContentSafetyWrapper()
    collector = counterfactual_collector(tmp_path, wrapper)
    item = {
        "question_id": 19,
        "query": "q",
        "answer": "a",
        "question_type": "inference_query",
    }

    first = collector.collect_one(item, "ablate_t3_s1")
    second = collector.collect_one(item, "ablate_t3_s3")

    assert first["status"] == "error"
    assert second["status"] == "excluded"
    assert wrapper.calls == 1
    assert 19 in collector.excluded_query_ids
    persisted = json.loads((tmp_path / "new1_manifest.json").read_text())
    assert persisted["status"] == "running"
    assert persisted["excluded_query_ids"] == [19]


def test_collect_deducts_only_new_exclusions_from_filtered_target(tmp_path):
    wrapper = SafetyAfterFirstRouteWrapper()
    collector = FixedRouteCollector(
        tmp_path,
        wrapper,
        identities="workflow-a",
        autosave_every=10,
        expected_routes=6,
        excluded_query_ids=(17,),
    )
    items = [
        {
            "question_id": 17,
            "query": "excluded before collection",
            "answer": "a",
            "question_type": "inference_query",
        },
        {
            "question_id": 18,
            "query": "safe",
            "answer": "a",
            "question_type": "inference_query",
        },
        {
            "question_id": 19,
            "query": "excluded after one success",
            "answer": "a",
            "question_type": "inference_query",
        },
    ]

    partial = collector.collect(items, route_names=("all_t3", "all_t2"))

    assert partial["status"] == "running"
    assert partial["completed_routes"] == 2
    assert partial["baselines"]["all_t1"]["completed"] is False

    stale_trajectory = collector.trajectory_buffer.records[0].to_dict()
    stale_trajectory.update(question_id=99, workflow_identity="stale-workflow")
    collector.trajectory_buffer.upsert_by_question_route(stale_trajectory)
    collector.route_caches["all_t1"].append(
        {
            "split": "train",
            "qid": 99,
            "question_id": 99,
            "route_name": "all_t1",
            "route_key": "0-0-0-0",
            "actions": [0, 0, 0, 0],
            "operator_version": 1,
            "workflow_identity": "stale-workflow",
            "status": "ok",
            "total_cost": 99.0,
            "total_latency_ms": 99.0,
        }
    )

    manifest = collector.collect(items, route_names=("all_t1",))

    assert wrapper.calls_by_qid == {18: 3, 19: 2}
    assert len(collector.trajectory_buffer.records) == 4
    assert all(
        record.question_id != 19 for record in collector.trajectory_buffer.records
    )
    assert manifest["status"] == "completed"
    assert manifest["completed_routes"] == 3
    assert manifest["excluded_query_ids"] == [17, 19]
    for summary in manifest["baselines"].values():
        assert summary["n_ok"] == 1
        assert summary["n_error"] == 0
        assert summary["completed"] is True
        assert summary["total_cost"] == 0.1


def test_test_split_collection_uses_dev_artifacts_and_reports_f1(tmp_path):
    collector = FixedRouteCollector(
        tmp_path,
        CountingWrapper(),
        identities="workflow-test",
        split="test",
        artifact_name="dev",
        routes={"all_t1": (0, 0, 0, 0)},
        route_order=("all_t1",),
        expected_routes=1,
    )

    manifest = collector.collect(
        [{"question_id": 501, "query": "q", "answer": "a"}]
    )
    collector.transition_cache.close()

    assert (tmp_path / "dev.json").exists()
    assert (tmp_path / "dev.db").exists()
    assert (tmp_path / "dev_manifest.json").exists()
    assert not (tmp_path / "new1.json").exists()
    record = json.loads((tmp_path / "dev.json").read_text(encoding="utf-8"))[0]
    assert record["split"] == "test"
    summary = manifest["baselines"]["all_t1"]
    assert summary["split"] == "test"
    assert summary["avg_f1"] == pytest.approx(0.75)
    assert summary["avg_cost"] == pytest.approx(0.1)
    assert summary["avg_latency_ms"] == pytest.approx(1.0)
