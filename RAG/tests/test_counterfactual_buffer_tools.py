import json
import sqlite3
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from RAG.rl.counterfactual_route_ladder import (
    RouteProfile,
    RouteTemplate,
    build_random_route_profile,
)
from RAG.rl.transition_cache import TransitionCache, TransitionKey
from RAG.rl.route_result_cache import RouteResultCache, route_key
from RAG.scripts import collect_fixed_routes
from RAG.scripts.build_route_library import build as build_route_library
from RAG.scripts.finalize_query_buffer import finalize
from RAG.scripts.verify_new1 import verify_new1


def _observed_source_plan(cache_root, identity):
    anchor_buffer = cache_root / "anchors.json"
    anchor_manifest = cache_root / "anchors_manifest.json"
    anchor_manifest.write_text(json.dumps({
        "status": "completed", "workflow_identity": identity, "operator_version": 1,
    }), encoding="utf-8")
    anchor_buffer.write_text(json.dumps([
        {
            "question_id": 1, "route_name": name, "actions": [tier] * 4,
            "workflow_identity": identity, "operator_version": 1,
            "f1_score": 0.8 - 0.1 * tier, "total_cost": 0.004,
            "decision_stage_costs": [0.001, 0.001, 0.002, 0.0],
        }
        for name, tier in (("all_t1", 0), ("all_t2", 1), ("all_t3", 2))
    ]), encoding="utf-8")
    plan_path = cache_root / "route_plan.json"
    plan_path.write_text(json.dumps(build_route_library(anchor_buffer, anchor_manifest)), encoding="utf-8")
    return plan_path, collect_fixed_routes._load_observed_route_plan(plan_path)


def test_single_writer_rejects_live_owner_and_recovers_dead_local_pid(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / ".new1_collection.lock"
    with collect_fixed_routes._SingleWriter(lock_path):
        with pytest.raises(RuntimeError, match="already running"):
            with collect_fixed_routes._SingleWriter(lock_path):
                pass
    assert not lock_path.exists()

    lock_path.write_text(
        json.dumps({"pid": 987654321, "host": socket.gethostname()}),
        encoding="utf-8",
    )

    def dead_pid(pid, signal):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(collect_fixed_routes.os, "kill", dead_pid)
    with collect_fixed_routes._SingleWriter(lock_path, recover_stale=True):
        assert lock_path.exists()
    assert not lock_path.exists()


@pytest.mark.skipif(
    collect_fixed_routes.os.name != "nt", reason="Windows PID probing regression"
)
def test_single_writer_recovery_does_not_terminate_live_windows_owner(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / ".new1_collection.lock"
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    lock_path.write_text(
        json.dumps({"pid": child.pid, "host": socket.gethostname()}),
        encoding="utf-8",
    )

    def destructive_zero_signal(pid, signal):
        assert pid == child.pid
        assert signal == 0
        child.terminate()
        child.wait(timeout=5)

    monkeypatch.setattr(collect_fixed_routes.os, "kill", destructive_zero_signal)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            with collect_fixed_routes._SingleWriter(
                lock_path, recover_stale=True
            ):
                pass
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=0.2)
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        lock_path.unlink(missing_ok=True)


def test_single_writer_recovery_guard_rejects_concurrent_recoverer(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / ".new1_collection.lock"
    lock_path.write_text(
        json.dumps({"pid": 987654321, "host": socket.gethostname()}),
        encoding="utf-8",
    )
    probe_started = threading.Event()
    release_probe = threading.Event()
    second_done = threading.Event()
    outcomes = []

    def wait_then_dead():
        probe_started.set()
        if not release_probe.wait(timeout=5):
            raise TimeoutError("test did not release PID probe")

    def dead_pid(_pid, _signal):
        wait_then_dead()
        raise ProcessLookupError(_pid)

    def pid_is_dead(_pid):
        wait_then_dead()
        return False

    monkeypatch.setattr(collect_fixed_routes.os, "kill", dead_pid)
    monkeypatch.setattr(
        collect_fixed_routes, "_pid_is_alive", pid_is_dead, raising=False
    )

    def attempt(*, finished=None):
        try:
            with collect_fixed_routes._SingleWriter(
                lock_path, recover_stale=True
            ):
                outcomes.append("entered")
        except RuntimeError:
            outcomes.append("rejected")
        finally:
            if finished is not None:
                finished.set()

    first = threading.Thread(target=attempt)
    second = threading.Thread(target=attempt, kwargs={"finished": second_done})
    first.start()
    assert probe_started.wait(timeout=5)
    second.start()
    second_failed_closed = second_done.wait(timeout=0.5)
    release_probe.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_failed_closed, "second recoverer reached the stale-owner probe"
    assert not first.is_alive()
    assert not second.is_alive()
    assert outcomes.count("entered") == 1
    assert outcomes.count("rejected") == 1
    assert [path.name for path in tmp_path.iterdir()] == [
        ".new1_collection.lock.recovery"
    ]


def test_single_writer_recovery_guard_is_reusable_after_failed_recovery(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / ".new1_collection.lock"
    guard_path = tmp_path / ".new1_collection.lock.recovery"
    guard_path.write_bytes(b"\0")
    lock_path.write_text(
        json.dumps({"pid": 987654321, "host": socket.gethostname()}),
        encoding="utf-8",
    )

    def probe_error(_pid):
        raise OSError("injected liveness failure")

    monkeypatch.setattr(collect_fixed_routes, "_pid_is_alive", probe_error)
    with pytest.raises(RuntimeError, match="already running"):
        with collect_fixed_routes._SingleWriter(lock_path, recover_stale=True):
            pass
    assert guard_path.exists()

    monkeypatch.setattr(collect_fixed_routes, "_pid_is_alive", lambda _pid: False)
    with collect_fixed_routes._SingleWriter(lock_path, recover_stale=True):
        assert lock_path.exists()
    assert not lock_path.exists()
    assert guard_path.exists()


def test_single_writer_completes_partial_owner_metadata_writes(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / ".new1_collection.lock"
    real_write = collect_fixed_routes.os.write

    def partial_write(fd, payload):
        return real_write(fd, payload[:3])

    monkeypatch.setattr(collect_fixed_routes.os, "write", partial_write)
    with collect_fixed_routes._SingleWriter(lock_path):
        assert json.loads(lock_path.read_text(encoding="utf-8")) == {
            "host": socket.gethostname(),
            "pid": collect_fixed_routes.os.getpid(),
        }
    assert not lock_path.exists()


def test_single_writer_cleans_lock_when_owner_metadata_fsync_fails(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / ".new1_collection.lock"
    real_fsync = collect_fixed_routes.os.fsync

    def fail_fsync(_fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(collect_fixed_routes.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        with collect_fixed_routes._SingleWriter(lock_path):
            pass

    monkeypatch.setattr(collect_fixed_routes.os, "fsync", real_fsync)
    assert not lock_path.exists()
    with collect_fixed_routes._SingleWriter(lock_path):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_single_writer_cleans_unpublished_lock_when_fstat_fails(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / ".new1_collection.lock"
    real_fstat = collect_fixed_routes.os.fstat

    def fail_fstat(_fd):
        raise OSError("injected fstat failure")

    monkeypatch.setattr(collect_fixed_routes.os, "fstat", fail_fstat)
    with pytest.raises(OSError, match="injected fstat failure"):
        with collect_fixed_routes._SingleWriter(lock_path):
            pass

    monkeypatch.setattr(collect_fixed_routes.os, "fstat", real_fstat)
    assert not lock_path.exists()
    with collect_fixed_routes._SingleWriter(lock_path):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_single_writer_cleans_lock_when_owner_payload_construction_fails(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / ".new1_collection.lock"
    real_gethostname = collect_fixed_routes.socket.gethostname

    def fail_gethostname():
        raise RuntimeError("injected hostname failure")

    monkeypatch.setattr(
        collect_fixed_routes.socket, "gethostname", fail_gethostname
    )
    with pytest.raises(RuntimeError, match="injected hostname failure"):
        with collect_fixed_routes._SingleWriter(lock_path):
            pass

    monkeypatch.setattr(
        collect_fixed_routes.socket, "gethostname", real_gethostname
    )
    assert not lock_path.exists()
    with collect_fixed_routes._SingleWriter(lock_path):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_finalize_excludes_content_safety_and_keeps_only_complete_qids(tmp_path):
    cache_root = tmp_path / "cache"
    buffer_path = cache_root / "new1.json"
    train_path = tmp_path / "train.json"
    routes = {
        "route_a": (0, 0, 0, 0),
        "route_b": (1, 1, 1, 1),
    }
    identity = "workflow-test"
    version = 7

    train_path.write_text(
        json.dumps([{"question_id": 1}, {"question_id": 2}]),
        encoding="utf-8",
    )
    cache_root.mkdir()
    buffer_path.write_text(
        json.dumps(
            [
                {
                    "split": "train",
                    "qid": 1,
                    "question_id": 1,
                    "route_name": "route_a",
                    "route_key": route_key(routes["route_a"]),
                    "actions": list(routes["route_a"]),
                    "operator_version": version,
                    "workflow_identity": identity,
                    "f1_score": 0.1,
                },
                {
                    "split": "train",
                    "qid": 1,
                    "question_id": 1,
                    "route_name": "route_b",
                    "route_key": route_key(routes["route_b"]),
                    "actions": list(routes["route_b"]),
                    "operator_version": version,
                    "workflow_identity": identity,
                    "f1_score": 0.2,
                },
                {
                    "split": "train",
                    "qid": 2,
                    "question_id": 2,
                    "route_name": "route_a",
                    "route_key": route_key(routes["route_a"]),
                    "actions": list(routes["route_a"]),
                    "operator_version": version,
                    "workflow_identity": identity,
                    "f1_score": 0.3,
                },
            ]
        ),
        encoding="utf-8",
    )
    (cache_root / "new1_manifest.json").write_text(
        json.dumps(
            {
                "status": "stopped",
                "workflow_identity": identity,
                "operator_version": version,
            }
        ),
        encoding="utf-8",
    )

    for route_name, actions in routes.items():
        cache = RouteResultCache(
            cache_root / "baselines" / route_name / "results.jsonl"
        )
        qids = (1, 2) if route_name == "route_a" else (1,)
        for qid in qids:
            cache.append(
                {
                    "split": "train",
                    "qid": qid,
                    "question_id": qid,
                    "route_name": route_name,
                    "route_key": route_key(actions),
                    "actions": list(actions),
                    "operator_version": version,
                    "workflow_identity": identity,
                    "status": "ok",
                    "error": "",
                    "f1_score": qid / 10,
                    "total_cost": qid / 100,
                    "total_latency_ms": qid,
                }
            )

    RouteResultCache(
        cache_root / "baselines" / "route_b" / "results.jsonl"
    ).append(
        {
            "split": "train",
            "qid": 2,
            "question_id": 2,
            "route_name": "route_b",
            "route_key": route_key(routes["route_b"]),
            "actions": list(routes["route_b"]),
            "operator_version": version,
            "workflow_identity": identity,
            "status": "error",
            "error": (
                "HTTP 400 data_inspection_failed: Input/Output data may contain "
                "inappropriate content"
            ),
        }
    )

    report = finalize(
        cache_root,
        train_path,
        routes,
        profile_name="test_two_routes_v1",
        profile_sha256="test-profile-sha256",
    )

    assert report["excluded_query_ids"] == [2]
    assert report["complete_query_ids"] == [1]
    assert report["buffer_records_after"] == 2
    assert {
        row["question_id"] for row in json.loads(buffer_path.read_text())
    } == {1}


def test_dry_run_reports_exact_reuse_without_external_dependencies(
    tmp_path, monkeypatch, capsys
):
    cache_root = tmp_path / "cache"
    train_path = tmp_path / "train.json"
    cache_root.mkdir()
    train_path.write_text(
        json.dumps([{"question_id": 1}, {"question_id": 2}]),
        encoding="utf-8",
    )
    identity = collect_fixed_routes._identity(train_path, cache_root).sha256
    routes = {
        "route_a": (0, 0, 0, 0),
        "route_b": (1, 1, 1, 1),
    }
    profile = SimpleNamespace(
        name="test_two_routes_v1",
        sha256="test-profile-sha256",
        routes=(object(), object()),
        route_map=routes,
        route_order=tuple(routes),
    )
    (cache_root / "new1_manifest.json").write_text(
        json.dumps({"excluded_query_ids": []}), encoding="utf-8"
    )
    (cache_root / "new1.json").write_text(
        json.dumps(
            [
                {
                    "question_id": 1,
                    "route_key": route_key(routes["route_a"]),
                    "actions": list(routes["route_a"]),
                    "status": "ok",
                    "operator_version": 7,
                    "workflow_identity": identity,
                }
            ]
        ),
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run constructed an external dependency")

    monkeypatch.setattr(collect_fixed_routes, "get_route_profile", lambda _: profile)
    monkeypatch.setattr(collect_fixed_routes, "get_adapter", forbidden)
    monkeypatch.setattr(collect_fixed_routes, "get_retriever", forbidden)
    monkeypatch.setattr(collect_fixed_routes, "TransitionCache", forbidden)
    monkeypatch.setattr(collect_fixed_routes, "_SingleWriter", forbidden)

    assert (
        collect_fixed_routes.main(
            [
                "--train",
                str(train_path),
                "--cache-root",
                str(cache_root),
                "--profile",
                "anchors",
                "--operator-version",
                "7",
                "--dry-run",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "candidate_qids": 2,
        "known_excluded_query_ids": [],
        "missing_pairs": 3,
        "reused_pairs": 1,
        "route_profile_name": "test_two_routes_v1",
        "route_profile_sha256": "test-profile-sha256",
        "routes_per_query": 2,
        "target_pairs": 4,
    }


def test_live_cli_holds_lock_around_resources_and_passes_worker_config(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "cache"
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([{"question_id": 1}]), encoding="utf-8")
    routes = {"route_a": (0, 0, 0, 0)}
    profile = SimpleNamespace(
        name="test_one_route_v1",
        sha256="test-profile-sha256",
        route_map=routes,
        route_order=tuple(routes),
    )
    events = []

    class RecordingLock:
        def __init__(self, path, recover_stale=False):
            assert path == cache_root / ".new1_collection.lock"
            assert recover_stale is True

        def __enter__(self):
            events.append("lock_enter")
            return self

        def __exit__(self, *_):
            events.append("lock_exit")

    class RecordingTransitionCache:
        def __init__(self, path, *, workflow_identity):
            assert path == cache_root / "new1.db"
            assert workflow_identity
            events.append("cache_open")

        def close(self):
            events.append("cache_close")

    class RecordingCollector:
        def __init__(self, *args, **kwargs):
            assert kwargs["n_workers"] == 4
            assert kwargs["max_inflight"] == 8
            events.append("collector_open")

        def collect(self, rows, route_names, *, cache_only=False):
            assert rows == [{"question_id": 1}]
            assert route_names == ["route_a"]
            assert cache_only is False
            events.append("collect")
            return {"status": "completed"}

    def make_wrapper(*, adapter, retriever, index_identity):
        assert adapter == "adapter"
        assert retriever == "retriever"
        events.append("wrapper_open")
        return object()

    monkeypatch.setattr(collect_fixed_routes, "get_route_profile", lambda _: profile)
    monkeypatch.setattr(collect_fixed_routes, "_SingleWriter", RecordingLock)
    monkeypatch.setattr(
        collect_fixed_routes, "TransitionCache", RecordingTransitionCache
    )
    monkeypatch.setattr(
        collect_fixed_routes,
        "get_adapter",
        lambda: events.append("adapter_open") or "adapter",
    )
    monkeypatch.setattr(
        collect_fixed_routes,
        "get_retriever",
        lambda: events.append("retriever_open") or "retriever",
    )
    monkeypatch.setattr(collect_fixed_routes, "RAGEnvironmentWrapper", make_wrapper)
    monkeypatch.setattr(collect_fixed_routes, "FixedRouteCollector", RecordingCollector)

    assert (
        collect_fixed_routes.main(
            [
                "--train",
                str(train_path),
                "--cache-root",
                str(cache_root),
                "--n-workers",
                "4",
                "--recover-stale-lock",
            ]
        )
        == 0
    )
    assert events == [
        "lock_enter",
        "cache_open",
        "adapter_open",
        "retriever_open",
        "wrapper_open",
        "collector_open",
        "collect",
        "cache_close",
        "lock_exit",
    ]


def test_cli_rejects_nonpositive_worker_count():
    with pytest.raises(SystemExit, match="--n-workers must be positive"):
        collect_fixed_routes.main(["--n-workers", "0"])


def test_live_cli_wires_test_split_to_named_dev_artifacts(tmp_path, monkeypatch):
    exp_root = tmp_path / "exp"
    index_root = tmp_path / "cache"
    (index_root / "vector_index").mkdir(parents=True)
    (index_root / "vector_index" / "manifest.json").write_text(
        json.dumps({"index_identity": "index-dev"}), encoding="utf-8"
    )
    test_path = tmp_path / "test.json"
    test_path.write_text(json.dumps([{"question_id": 501}]), encoding="utf-8")
    captured = {}

    class RecordingLock:
        def __init__(self, path, recover_stale=False):
            captured["lock"] = path

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    class RecordingTransitionCache:
        def __init__(self, path, *, workflow_identity):
            captured["db"] = path

        def close(self):
            return None

    class RecordingCollector:
        def __init__(self, *args, **kwargs):
            captured["collector"] = kwargs

        def collect(self, rows, route_names, *, cache_only=False):
            captured["rows"] = rows
            captured["routes"] = route_names
            return {"status": "completed"}

    def make_wrapper(*, adapter, retriever, index_identity):
        captured["index_identity"] = index_identity
        return object()

    monkeypatch.setattr(collect_fixed_routes, "_SingleWriter", RecordingLock)
    monkeypatch.setattr(
        collect_fixed_routes, "TransitionCache", RecordingTransitionCache
    )
    monkeypatch.setattr(collect_fixed_routes, "get_adapter", lambda: object())
    monkeypatch.setattr(collect_fixed_routes, "get_retriever", lambda: object())
    monkeypatch.setattr(collect_fixed_routes, "RAGEnvironmentWrapper", make_wrapper)
    monkeypatch.setattr(collect_fixed_routes, "FixedRouteCollector", RecordingCollector)

    assert collect_fixed_routes.main(
        [
            "--dataset",
            str(test_path),
            "--cache-root",
            str(exp_root),
            "--index-cache-root",
            str(index_root),
            "--profile",
            "anchors",
            "--routes",
            "all_t1,all_t2,all_t3",
            "--split",
            "test",
            "--artifact-name",
            "dev",
            "--n-workers",
            "64",
        ]
    ) == 0
    assert captured["lock"] == exp_root / ".dev_collection.lock"
    assert captured["db"] == exp_root / "dev.db"
    assert captured["index_identity"] == "index-dev"
    assert captured["collector"]["split"] == "test"
    assert captured["collector"]["artifact_name"] == "dev"
    assert captured["collector"]["n_workers"] == 64
    assert captured["routes"] == ["all_t1", "all_t2", "all_t3"]


def _complete_profile_cache(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    train_path.write_text(json.dumps([{"question_id": 1}]), encoding="utf-8")
    test_path.write_text(json.dumps([{"question_id": 99}]), encoding="utf-8")
    routes = {
        "route_a": (0, 0, 0, 0),
        "route_b": (1, 1, 1, 1),
    }
    identity = "workflow-test"
    profile_name = "test_two_routes_v1"
    profile_sha256 = "test-profile-sha256"
    records = []
    for route_name, actions in routes.items():
        record = {
            "split": "train",
            "qid": 1,
            "question_id": 1,
            "route_name": route_name,
            "route_key": route_key(actions),
            "actions": list(actions),
            "status": "ok",
            "operator_version": 7,
            "workflow_identity": identity,
            "index_identity": "",
            "f1_score": 0.5,
            "reward": 0.5,
            "total_cost": 0.1,
            "stage_costs": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "decision_stage_costs": [0.1, 0.0, 0.0, 0.0],
            "fixed_stage_costs": [0.0] * 7,
            "total_latency_ms": 1.0,
            "stage_latencies_ms": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "decision_stage_latencies_ms": [1.0, 0.0, 0.0, 0.0],
            "fixed_stage_latencies_ms": [0.0] * 7,
        }
        records.append(record)
        RouteResultCache(
            cache_root / "baselines" / route_name / "results.jsonl"
        ).append(record)
        summary = {
            "split": "train",
            "route_name": route_name,
            "route_key": route_key(actions),
            "actions": list(actions),
            "operator_version": 7,
            "workflow_identity": identity,
            "n_ok": 1,
            "n_error": 0,
            "completed": True,
            "total_cost": 0.1,
        }
        summary_path = cache_root / "baselines" / route_name / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (cache_root / "new1.json").write_text(json.dumps(records), encoding="utf-8")
    (cache_root / "new1_manifest.json").write_text(
        json.dumps(
            {
                "workflow_identity": identity,
                "operator_version": 7,
                "route_profile_name": profile_name,
                "route_profile_sha256": profile_sha256,
                "route_count_per_query": 2,
                "expected_routes": 2,
                "completed_routes": 2,
                "excluded_query_ids": [],
            }
        ),
        encoding="utf-8",
    )
    transition_cache = TransitionCache(
        cache_root / "new1.db", workflow_identity=identity
    )
    transition_cache.close()
    return (
        cache_root,
        train_path,
        test_path,
        routes,
        profile_name,
        profile_sha256,
    )


def test_profile_aware_verifier_accepts_complete_and_rejects_missing_route(tmp_path):
    (
        cache_root,
        train_path,
        test_path,
        routes,
        profile_name,
        profile_sha256,
    ) = _complete_profile_cache(tmp_path)

    report = verify_new1(
        cache_root,
        train_path,
        test_path,
        routes=routes,
        profile_name=profile_name,
        profile_sha256=profile_sha256,
    )
    assert report["ok"] is True, report["errors"]

    buffer_path = cache_root / "new1.json"
    records = json.loads(buffer_path.read_text(encoding="utf-8"))
    buffer_path.write_text(json.dumps(records[:1]), encoding="utf-8")
    report = verify_new1(
        cache_root,
        train_path,
        test_path,
        routes=routes,
        profile_name=profile_name,
        profile_sha256=profile_sha256,
    )
    assert report["ok"] is False
    assert any("missing approved routes" in error for error in report["errors"])


def test_profile_aware_verifier_ignores_old_identity_route_history(tmp_path):
    (
        cache_root,
        train_path,
        test_path,
        routes,
        profile_name,
        profile_sha256,
    ) = _complete_profile_cache(tmp_path)
    actions = routes["route_a"]
    RouteResultCache(
        cache_root / "baselines" / "route_a" / "results.jsonl"
    ).append(
        {
            "split": "train",
            "qid": 1,
            "question_id": 1,
            "route_name": "route_a",
            "route_key": route_key(actions),
            "actions": list(actions),
            "status": "ok",
            "operator_version": 7,
            "workflow_identity": "old-workflow",
            "total_cost": 99.0,
        }
    )

    report = verify_new1(
        cache_root,
        train_path,
        test_path,
        routes=routes,
        profile_name=profile_name,
        profile_sha256=profile_sha256,
    )
    assert report["ok"] is True, report["errors"]


def test_profile_verifier_uses_effective_count_after_query_exclusion(tmp_path):
    (
        cache_root,
        train_path,
        test_path,
        routes,
        profile_name,
        profile_sha256,
    ) = _complete_profile_cache(tmp_path)
    train_path.write_text(
        json.dumps([{"question_id": 1}, {"question_id": 2}]), encoding="utf-8"
    )
    manifest_path = cache_root / "new1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "excluded_query_ids": [2],
            "expected_routes": 4,
            "completed_routes": 2,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_new1(
        cache_root,
        train_path,
        test_path,
        routes=routes,
        profile_name=profile_name,
        profile_sha256=profile_sha256,
    )
    assert report["ok"] is True, report["errors"]
    assert report["declared_expected_routes"] == 4
    assert report["expected_routes"] == 2


def test_complete_source_qids_requires_every_route_and_matching_identity(tmp_path):
    source = tmp_path / "new1.json"
    approved = {"0-0-0-0", "1-1-1-1"}
    source.write_text(
        json.dumps(
            [
                {
                    "qid": 1,
                    "route_key": route,
                    "split": "train",
                    "status": "ok",
                    "workflow_identity": "workflow-a",
                    "operator_version": 1,
                }
                for route in sorted(approved)
            ]
            + [
                {
                    "qid": 2,
                    "route_key": "0-0-0-0",
                    "split": "train",
                    "status": "ok",
                    "workflow_identity": "workflow-a",
                    "operator_version": 1,
                }
            ]
            + [
                {
                    "qid": 3,
                    "route_key": route,
                    "split": "train",
                    "status": "ok",
                    "workflow_identity": "old-workflow",
                    "operator_version": 1,
                }
                for route in sorted(approved)
            ],
        ),
        encoding="utf-8",
    )

    assert collect_fixed_routes._complete_source_qids(
        source,
        approved_route_keys=approved,
        workflow_identity="workflow-a",
        operator_version=1,
        split="train",
    ) == (1,)


def test_transition_subset_clone_is_atomic_pruned_and_resume_safe(tmp_path):
    source_path = tmp_path / "new1.db"
    destination_path = tmp_path / "random_route_36.db"
    source = TransitionCache(source_path, workflow_identity="workflow-a")
    try:
        for qid in (1, 2, 3):
            source.store(
                TransitionKey(qid, stage_idx=0, tier=0, input_hash=f"input-{qid}"),
                {"qid": qid},
                source="test",
            )
    finally:
        source.close()

    source_rows_before = sqlite3.connect(source_path).execute(
        "SELECT question_id FROM transitions ORDER BY question_id"
    ).fetchall()
    report = collect_fixed_routes._clone_transition_subset(
        source_path,
        destination_path,
        qids=(1, 3),
        workflow_identity="workflow-a",
        route_profile_sha256="profile-a",
    )
    assert report["reused_existing_clone"] is False
    with sqlite3.connect(destination_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute(
            "SELECT question_id FROM transitions ORDER BY question_id"
        ).fetchall() == [(1,), (3,)]
        metadata = dict(connection.execute("SELECT key, value FROM cache_meta"))
    assert metadata["workflow_identity"] == "workflow-a"
    assert metadata["route_profile_sha256"] == "profile-a"

    resumed = collect_fixed_routes._clone_transition_subset(
        source_path,
        destination_path,
        qids=(1, 3),
        workflow_identity="workflow-a",
        route_profile_sha256="profile-a",
    )
    assert resumed["reused_existing_clone"] is True
    with pytest.raises(ValueError, match="clone identity mismatch"):
        collect_fixed_routes._clone_transition_subset(
            source_path,
            destination_path,
            qids=(1, 3),
            workflow_identity="workflow-a",
            route_profile_sha256="profile-b",
        )
    assert sqlite3.connect(source_path).execute(
        "SELECT question_id FROM transitions ORDER BY question_id"
    ).fetchall() == source_rows_before


def test_random_route_dry_run_freezes_plan_and_uses_only_complete_source_qids(
    tmp_path, capsys
):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    train_path = tmp_path / "train.json"
    train_path.write_text(
        json.dumps([{"question_id": qid} for qid in (1, 2, 3)]),
        encoding="utf-8",
    )
    identity = collect_fixed_routes._identity(train_path, cache_root).sha256
    source_plan_path, source_profile = _observed_source_plan(cache_root, identity)
    records = []
    for qid in (1, 2):
        for name, actions in source_profile.route_map.items():
            records.append(
                {
                    "qid": qid,
                    "route_name": name,
                    "route_key": route_key(actions),
                    "actions": list(actions),
                    "split": "train",
                    "status": "ok",
                    "workflow_identity": identity,
                    "operator_version": 1,
                    "decision_stage_costs": [0.0] * 4,
                    "fixed_stage_costs": [0.0] * 7,
                }
            )
    first_name, first_actions = next(iter(source_profile.route_map.items()))
    records.append(
        {
            "qid": 3,
            "route_name": first_name,
            "route_key": route_key(first_actions),
            "actions": list(first_actions),
            "split": "train",
            "status": "ok",
            "workflow_identity": identity,
            "operator_version": 1,
            "decision_stage_costs": [0.0] * 4,
            "fixed_stage_costs": [0.0] * 7,
        }
    )
    (cache_root / "new1.json").write_text(json.dumps(records), encoding="utf-8")
    (cache_root / "new1_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "workflow_identity": identity,
                "operator_version": 1,
                "route_profile_sha256": source_profile.sha256,
                "excluded_query_ids": [],
            }
        ),
        encoding="utf-8",
    )
    source_cache = TransitionCache(cache_root / "new1.db", workflow_identity=identity)
    source_cache.close()
    plan_path = cache_root / "random_route_36_routes.json"

    assert collect_fixed_routes.main(
        [
            "--dataset",
            str(train_path),
            "--cache-root",
            str(cache_root),
            "--artifact-name",
            "random_route_36",
            "--random-route-count",
            "36",
            "--sampling-seed",
            "1145",
            "--route-plan-file",
            str(plan_path),
            "--source-artifact",
            "new1",
            "--source-route-plan-file",
            str(source_plan_path),
            "--clone-source-transitions",
            "--dry-run",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["candidate_qids"] == 2
    assert report["target_pairs"] == 72
    overlap = len(set(build_random_route_profile(count=36, seed=1145).route_map.values()) & set(source_profile.route_map.values()))
    assert report["canonical_overlap_routes"] == overlap
    assert report["estimated_complete_route_hits"] == 2 * overlap
    assert report["source_complete_qids"] == 2
    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    assert frozen["seed"] == 1145
    assert len(frozen["routes"]) == 36
    assert len({tuple(row["actions"]) for row in frozen["routes"]}) == 36


def test_random_route_live_cli_reuses_collector_and_worker_pool(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([{"question_id": 1}]), encoding="utf-8")
    identity = collect_fixed_routes._identity(train_path, cache_root).sha256
    source_plan_path, source_profile = _observed_source_plan(cache_root, identity)
    records = [
        {
            "qid": 1,
            "route_name": name,
            "route_key": route_key(actions),
            "actions": list(actions),
            "split": "train",
            "status": "ok",
            "workflow_identity": identity,
            "operator_version": 1,
        }
        for name, actions in source_profile.route_map.items()
    ]
    (cache_root / "new1.json").write_text(json.dumps(records), encoding="utf-8")
    (cache_root / "new1_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "workflow_identity": identity,
                "operator_version": 1,
                "route_profile_sha256": source_profile.sha256,
                "excluded_query_ids": [],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class RecordingLock:
        def __init__(self, path, recover_stale=False):
            captured["lock"] = path

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    class RecordingTransitionCache:
        def __init__(self, path, *, workflow_identity):
            captured["db"] = path

        def close(self):
            return None

    class RecordingCollector:
        def __init__(self, *args, **kwargs):
            captured["collector"] = kwargs
            self._manifest_extra = {}

        def collect(self, rows, route_names, *, cache_only=False):
            captured["rows"] = rows
            captured["routes"] = route_names
            captured["manifest_extra"] = dict(self._manifest_extra)
            return {"status": "completed"}

    def clone(source, destination, **kwargs):
        captured["clone"] = (source, destination, kwargs)
        return {"reused_existing_clone": False, "transition_count": 0}

    monkeypatch.setattr(collect_fixed_routes, "_SingleWriter", RecordingLock)
    monkeypatch.setattr(collect_fixed_routes, "_clone_transition_subset", clone)
    monkeypatch.setattr(
        collect_fixed_routes, "TransitionCache", RecordingTransitionCache
    )
    monkeypatch.setattr(collect_fixed_routes, "get_adapter", lambda: object())
    monkeypatch.setattr(collect_fixed_routes, "get_retriever", lambda: object())
    monkeypatch.setattr(
        collect_fixed_routes, "RAGEnvironmentWrapper", lambda **_: object()
    )
    monkeypatch.setattr(
        collect_fixed_routes, "FixedRouteCollector", RecordingCollector
    )

    assert collect_fixed_routes.main(
        [
            "--dataset",
            str(train_path),
            "--cache-root",
            str(cache_root),
            "--artifact-name",
            "random_route_36",
            "--random-route-count",
            "36",
            "--route-plan-file",
            str(cache_root / "random_route_36_routes.json"),
            "--source-artifact",
            "new1",
            "--source-route-plan-file",
            str(source_plan_path),
            "--clone-source-transitions",
            "--n-workers",
            "64",
        ]
    ) == 0
    assert captured["lock"] == cache_root / ".random_route_36_collection.lock"
    assert captured["db"] == cache_root / "random_route_36.db"
    assert captured["clone"][0] == cache_root / "new1.db"
    assert captured["clone"][2]["qids"] == (1,)
    assert captured["collector"]["n_workers"] == 64
    assert captured["collector"]["max_inflight"] == 128
    assert captured["collector"]["expected_routes"] == 36
    assert captured["rows"] == [{"question_id": 1}]
    assert len(captured["routes"]) == 36
    assert captured["manifest_extra"]["source_artifact"] == "new1"


def test_dynamic_profile_verifier_uses_named_artifact_and_frozen_plan(
    tmp_path, capsys
):
    (
        cache_root,
        train_path,
        test_path,
        routes,
        profile_name,
        profile_sha256,
    ) = _complete_profile_cache(tmp_path)
    artifact_name = "random_route_test"
    profile_sha256 = RouteProfile(
        profile_name,
        tuple(
            RouteTemplate(rank, name, tuple(actions))
            for rank, (name, actions) in enumerate(routes.items())
        ),
    ).sha256
    plan_path = cache_root / f"{artifact_name}_routes.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "rag_random_route_plan_v1",
                "count": len(routes),
                "seed": 1145,
                "legal_route_sha256": collect_fixed_routes._random_route_plan_payload(
                    RouteProfile(
                        profile_name,
                        tuple(
                            RouteTemplate(rank, name, tuple(actions))
                            for rank, (name, actions) in enumerate(routes.items())
                        ),
                    ),
                    count=len(routes),
                    seed=1145,
                )["legal_route_sha256"],
                "route_profile_name": profile_name,
                "route_profile_sha256": profile_sha256,
                "routes": [
                    {"rank": rank, "name": name, "actions": list(actions)}
                    for rank, (name, actions) in enumerate(routes.items())
                ],
            }
        ),
        encoding="utf-8",
    )
    (cache_root / "new1.json").replace(cache_root / f"{artifact_name}.json")
    (cache_root / "new1.db").replace(cache_root / f"{artifact_name}.db")
    manifest_path = cache_root / "new1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "artifact_name": artifact_name,
            "route_profile_sha256": profile_sha256,
            "route_plan_file_sha256": collect_fixed_routes._sha256(plan_path),
        }
    )
    manifest_path.unlink()
    (cache_root / f"{artifact_name}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    # These malformed defaults prove the named verifier does not inspect new1.*.
    (cache_root / "new1.json").write_text("not-json", encoding="utf-8")
    (cache_root / "new1_manifest.json").write_text("not-json", encoding="utf-8")
    (cache_root / "new1.db").write_text("not-sqlite", encoding="utf-8")

    assert verify_new1.__module__ == "RAG.scripts.verify_new1"
    from RAG.scripts import verify_new1 as verify_module

    assert verify_module.main(
        [
            "--cache-root",
            str(cache_root),
            "--train",
            str(train_path),
            "--test",
            str(test_path),
            "--artifact-name",
            artifact_name,
            "--route-plan-file",
            str(plan_path),
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True, report["errors"]
    assert report["artifact_name"] == artifact_name
    assert report["route_profile_name"] == profile_name
