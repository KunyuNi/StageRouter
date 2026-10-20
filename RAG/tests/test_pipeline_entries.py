"""Offline checks for the staged collection entry points."""

from __future__ import annotations

import json

from RAG.rl.counterfactual_route_ladder import ANCHOR_PROFILE
from RAG.rl.route_result_cache import route_key
from RAG.scripts import build_route_library, collect_fixed_routes
from RAG.scripts import collect_dev_cache, collect_train_buffer


def test_train_entry_requires_explicit_execution(monkeypatch, capsys):
    monkeypatch.setattr(collect_train_buffer.collect_fixed_routes, "main", lambda args: (_ for _ in ()).throw(AssertionError("paid collection")))
    assert collect_train_buffer.main(["--phase", "all"]) == 0
    assert '"paid_calls_enabled": false' in capsys.readouterr().out


def test_train_entry_runs_anchors_plan_remaining(monkeypatch):
    steps = []
    monkeypatch.setattr(collect_train_buffer.collect_fixed_routes, "main", lambda args: steps.append(tuple(args)))
    monkeypatch.setattr(collect_train_buffer.build_route_library, "main", lambda args: steps.append(tuple(args)))
    assert collect_train_buffer.main(["--phase", "all", "--execute"]) == 0
    assert len(steps) == 3
    assert ("--profile", "anchors") == steps[0][steps[0].index("--profile"):steps[0].index("--profile") + 2]
    assert "--cost-share-threshold" in steps[1]
    assert ("--profile", "observed") == steps[2][steps[2].index("--profile"):steps[2].index("--profile") + 2]
    assert "--clone-source-transitions" in steps[2]


def test_dev_entry_collects_only_homogeneous_routes(monkeypatch):
    calls = []
    monkeypatch.setattr(collect_dev_cache, "collect", lambda args: calls.append(args) or 0)
    assert collect_dev_cache.main(["--execute"]) == 0
    args = calls[0]
    assert args[args.index("--routes") + 1] == "all_t1,all_t2,all_t3"
    assert args[args.index("--artifact-name") + 1] == "dev"


def test_observed_collection_clones_complete_anchors(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    dataset = tmp_path / "train.json"
    dataset.write_text(json.dumps([{"question_id": 1}]), encoding="utf-8")
    identity = collect_fixed_routes._identity(dataset, cache_root).sha256
    anchor_rows = [
        {
            "question_id": 1, "route_name": route.name,
            "route_key": route_key(route.actions), "actions": list(route.actions),
            "split": "train", "status": "ok", "workflow_identity": identity,
            "operator_version": 1, "total_cost": 0.004,
            "decision_stage_costs": [0.001, 0.0004, 0.002, 0.0006],
        }
        for route in ANCHOR_PROFILE.routes
    ]
    (cache_root / "anchors.json").write_text(json.dumps(anchor_rows), encoding="utf-8")
    (cache_root / "anchors_manifest.json").write_text(json.dumps({
        "status": "completed", "workflow_identity": identity,
        "operator_version": 1, "route_profile_sha256": ANCHOR_PROFILE.sha256,
        "excluded_query_ids": [],
    }), encoding="utf-8")
    plan_path = cache_root / "route_plan.json"
    plan_path.write_text(json.dumps(build_route_library.build(
        cache_root / "anchors.json", cache_root / "anchors_manifest.json"
    )), encoding="utf-8")
    captured = {}

    class Lock:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Cache:
        def __init__(self, path, *, workflow_identity):
            captured["cache_path"] = path

        def close(self):
            pass

    class Collector:
        def __init__(self, *_args, **kwargs):
            captured["routes"] = kwargs["routes"]
            self._manifest_extra = {}

        def collect(self, rows, routes, *, cache_only=False):
            captured["rows"] = rows
            captured["manifest_extra"] = dict(self._manifest_extra)
            return {"status": "completed"}

    def clone(source, destination, **kwargs):
        captured["clone"] = (source, destination, kwargs)
        return {"transition_count": 3}

    monkeypatch.setattr(collect_fixed_routes, "_SingleWriter", Lock)
    monkeypatch.setattr(collect_fixed_routes, "_clone_transition_subset", clone)
    monkeypatch.setattr(collect_fixed_routes, "TransitionCache", Cache)
    monkeypatch.setattr(collect_fixed_routes, "FixedRouteCollector", Collector)
    monkeypatch.setattr(collect_fixed_routes, "get_adapter", lambda: object())
    monkeypatch.setattr(collect_fixed_routes, "get_retriever", lambda: object())
    monkeypatch.setattr(collect_fixed_routes, "RAGEnvironmentWrapper", lambda **_: object())
    assert collect_fixed_routes.main([
        "--dataset", str(dataset), "--cache-root", str(cache_root),
        "--artifact-name", "new1", "--profile", "observed",
        "--route-plan-file", str(plan_path), "--source-artifact", "anchors",
        "--clone-source-transitions",
    ]) == 0
    assert captured["clone"][0] == cache_root / "anchors.db"
    assert captured["clone"][1] == cache_root / "new1.db"
    assert captured["clone"][2]["qids"] == (1,)
    assert captured["manifest_extra"]["transition_clone"] == {"transition_count": 3}
    assert len(captured["routes"]) == len(json.loads(plan_path.read_text())["routes"])
