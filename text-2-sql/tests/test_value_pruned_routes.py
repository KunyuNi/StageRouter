import pytest

from pipeline.value_pruned_routes import build_value_pruned_routes


def _anchors():
    return [
        {
            "question_id": qid,
            "actions": [model] * 4,
            "decision_stage_costs": [1.0, 5.0, 1.0, 3.0],
            "total_cost": 10.0,
        }
        for qid in (7, 8)
        for model in range(3)
    ]


def test_route_pruning_profiles_anchors_and_covers_three_variation_families():
    result = build_value_pruned_routes(
        _anchors(), model_count=3, stage_count=4, skippable_stages=(0, 2)
    )
    assert result.anchor_requests == 2
    assert result.mean_cost_shares == pytest.approx((0.1, 0.5, 0.1, 0.3))
    assert result.high_cost_stages == (1, 3)
    actions = {tuple(route.actions) for route in result.routes}
    assert len(actions) == len(result.routes)
    assert (0, 0, 0, 0) in actions
    assert (0, 2, 0, 0) in actions  # single-stage isolation
    assert (3, 0, 0, 0) in actions  # optional-stage ablation
    assert (0, 2, 0, 1) in actions  # coupled-stage variation
    assert all(route[stage] != 3 for route in actions for stage in (1, 3))


def test_route_pruning_rejects_missing_or_duplicate_anchor():
    with pytest.raises(ValueError, match="all homogeneous anchors"):
        build_value_pruned_routes(
            _anchors()[:-1], model_count=3, stage_count=4, skippable_stages=(0,)
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_value_pruned_routes(
            _anchors() + [_anchors()[0]], model_count=3, stage_count=4, skippable_stages=(0,)
        )


def test_generated_library_is_accepted_by_collection_loader(tmp_path):
    import json
    from scripts._collect_routes import load_route_templates, parse_args

    traces = []
    for qid in (1, 2):
        for model in range(3):
            traces.append({"question_id": qid, "actions": [model] * 6,
                           "decision_stage_costs": [0.05, 0.6, 0.05, 0.05, 0.2, 0.05],
                           "total_cost": 1.0})
    result = build_value_pruned_routes(
        traces, model_count=3, stage_count=6, skippable_stages=(0, 2, 3, 5)
    )
    path = tmp_path / "routes.json"
    path.write_text(json.dumps([{"name": route.name, "actions": route.actions}
                                for route in result.routes]), encoding="utf-8")
    loaded = load_route_templates(str(path))
    args = parse_args(["--route_library", str(path)])
    assert args.reward_version == "ex_first_cost_clipped_v1"
    assert len(loaded) == len(result.routes)
    assert result.high_cost_stages == (1, 4)
    assert all(route.actions[1] != 3 and route.actions[4] != 3 for route in loaded)


def test_collection_profiles_anchors_then_resumes_without_new_calls(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from scripts import collect_train_buffer

    data = tmp_path / "selected_train.json"
    data.write_text(json.dumps([{"question_id": qid} for qid in (1, 2, 3)]), encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    buffer = tmp_path / "buffer.json"
    library = tmp_path / "routes.json"
    calls = []

    def fake_child(command, **_kwargs):
        def option(name):
            return command[command.index(name) + 1]

        candidates = json.loads(Path(option("--route_library")).read_text(encoding="utf-8"))
        actions = next(row["actions"] for row in candidates if row["name"] == option("--route_names"))
        current = json.loads(buffer.read_text(encoding="utf-8")) if buffer.exists() else []
        existing = {(row["question_id"], tuple(row["actions"])) for row in current}
        for qid in (1, 2, 3):
            if (qid, tuple(actions)) not in existing:
                current.append({"question_id": qid, "actions": actions,
                                "decision_stage_costs": [0.05, 0.6, 0.05, 0.05, 0.2, 0.05],
                                "total_cost": 1.0})
        buffer.write_text(json.dumps(current), encoding="utf-8")
        calls.append(tuple(actions))

    monkeypatch.setattr(collect_train_buffer.subprocess, "run", fake_child)
    options = ["--data", str(data), "--contract", str(contract),
               "--buffer", str(buffer), "--route-library", str(library)]
    with pytest.raises(FileNotFoundError, match="route-library manifest"):
        collect_train_buffer.main([*options, "--phase", "remaining", "--execute"])
    collect_train_buffer.main([*options, "--phase", "anchors", "--execute"])
    assert len(calls) == 3
    assert not library.exists()
    collect_train_buffer.main([*options, "--phase", "route-library"])
    assert len(calls) == 3
    routes = json.loads(library.read_text(encoding="utf-8"))
    collect_train_buffer.main([*options, "--phase", "remaining", "--execute"])
    assert len(json.loads(buffer.read_text(encoding="utf-8"))) == 3 * len(routes)
    assert json.loads(library.with_suffix(".manifest.json").read_text())["high_cost_stages"] == [1, 4]
    first_call_count = len(calls)
    collect_train_buffer.main([*options, "--execute"])
    assert len(calls) == first_call_count
    contract.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="different collection design"):
        collect_train_buffer.main([*options, "--phase", "route-library"])
