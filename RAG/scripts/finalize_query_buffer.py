"""Finalize ``new1`` using profile-aware query-level route exclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from RAG.rl.counterfactual_route_ladder import get_route_profile
from RAG.rl.query_filter import (
    QUERY_FILTER_POLICY,
    complete_query_ids,
    content_safety_query_ids,
    latest_route_statuses,
)
from RAG.rl.route_result_cache import RouteResultCache, route_key


_DEFAULT_PROFILE = get_route_profile("anchors")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.finalize.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _materialize_complete_buffer(
    path: Path,
    complete: set[int],
    routes: Mapping[str, Sequence[int]],
    workflow_identity: str,
    operator_version: int,
) -> tuple[int, int]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError("trajectory buffer must contain a JSON list")
    canonical_routes = {
        str(name): (route_key(actions), tuple(int(action) for action in actions))
        for name, actions in routes.items()
    }
    kept: list[dict[str, Any]] = []
    for value in data:
        if not isinstance(value, Mapping):
            raise ValueError("trajectory buffer records must be objects")
        raw_qid = value.get("question_id", value.get("qid"))
        if raw_qid is None or int(raw_qid) not in complete:
            continue
        if str(value.get("split", "")) != "train":
            continue
        route_name = value.get("route_name")
        canonical = canonical_routes.get(route_name)
        if canonical is None or str(value.get("route_key", "")) != canonical[0]:
            continue
        try:
            version = int(value.get("operator_version"))
        except (TypeError, ValueError):
            continue
        if value.get("actions") != list(canonical[1]):
            continue
        if str(value.get("workflow_identity", "")) != workflow_identity:
            continue
        if version != operator_version:
            continue
        kept.append(dict(value))
    _dump_json(path, kept)
    return len(data), len(kept)


def finalize(
    cache_root: str | Path,
    train_path: str | Path,
    routes: Mapping[str, Sequence[int]] | None = None,
    *,
    profile_name: str | None = None,
    profile_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(cache_root)
    route_map = dict(routes) if routes is not None else _DEFAULT_PROFILE.route_map
    resolved_profile_name = str(profile_name or _DEFAULT_PROFILE.name)
    resolved_profile_sha256 = str(profile_sha256 or _DEFAULT_PROFILE.sha256)
    manifest_path = root / "new1_manifest.json"
    manifest = dict(_load_json(manifest_path))
    identity = str(manifest.get("workflow_identity", ""))
    version = int(manifest.get("operator_version", 1))
    train_rows = _load_json(Path(train_path))
    train_qids = {
        int(row.get("question_id", row.get("qid")))
        for row in train_rows
        if isinstance(row, dict) and row.get("question_id", row.get("qid")) is not None
    }
    statuses = latest_route_statuses(
        root,
        route_names=route_map,
        split="train",
        workflow_identity=identity,
        operator_version=version,
    )
    known_excluded = {
        int(qid) for qid in manifest.get("excluded_query_ids", [])
    }
    safety_excluded = content_safety_query_ids(statuses)
    excluded = (known_excluded | safety_excluded) & train_qids
    eligible = train_qids - excluded
    complete = complete_query_ids(statuses, route_map) & eligible
    incomplete = eligible - complete
    expected_routes = len(eligible) * len(route_map)
    before, after = _materialize_complete_buffer(
        root / "new1.json",
        complete,
        route_map,
        identity,
        version,
    )

    summaries: dict[str, Any] = {}
    for name, actions in route_map.items():
        path = root / "baselines" / name / "results.jsonl"
        cache = RouteResultCache(path)
        selected = [
            row
            for (row_split, qid, row_route_key, row_version, row_identity), row
            in getattr(cache, "_index", {}).items()
            if row_split == "train"
            and qid in complete
            and row_route_key == route_key(actions)
            and row_version == version
            and row_identity == identity
        ]
        ok_rows = [row for row in selected if row.get("status", "ok") == "ok"]
        total_cost = sum(float(row.get("total_cost", 0.0) or 0.0) for row in ok_rows)
        total_latency = sum(float(row.get("total_latency_ms", 0.0) or 0.0) for row in ok_rows)
        summaries[name] = {
            "split": "train",
            "route_name": name,
            "route_key": route_key(actions),
            "actions": list(actions),
            "operator_version": version,
            "workflow_identity": identity,
            "n_ok": len(ok_rows),
            "n_error": len(selected) - len(ok_rows),
            "completed": len(ok_rows) == len(complete) and len(selected) == len(complete),
            "total_cost": total_cost,
            "avg_cost": total_cost / len(ok_rows) if ok_rows else 0.0,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(ok_rows) if ok_rows else 0.0,
            "excluded_query_count": len(excluded),
        }
        _dump_json(root / "baselines" / name / "summary.json", summaries[name])

    manifest.update(
        {
            "status": "completed" if not incomplete else "stopped",
            "completed_routes": after,
            "expected_routes": expected_routes,
            "baselines": summaries,
            "query_filter": {
                "policy": QUERY_FILTER_POLICY,
                "excluded_query_ids": sorted(excluded),
                "excluded_query_count": len(excluded),
                "valid_query_count": len(eligible),
                "valid_route_count": expected_routes,
                "raw_buffer_records_before_filter": before,
            },
            "excluded_query_ids": sorted(excluded),
            "incomplete_query_ids": sorted(incomplete),
            "complete_query_ids": sorted(complete),
            "route_profile_name": resolved_profile_name,
            "route_profile_sha256": resolved_profile_sha256,
            "valid_query_count": len(eligible),
        }
    )
    manifest["total_cost"] = sum(item["total_cost"] for item in summaries.values())
    manifest["total_latency_ms"] = sum(item["total_latency_ms"] for item in summaries.values())
    if not incomplete:
        manifest["stop_reason"] = ""
    _dump_json(manifest_path, manifest)
    return {
        "excluded_query_ids": sorted(excluded),
        "excluded_query_count": len(excluded),
        "incomplete_query_ids": sorted(incomplete),
        "complete_query_ids": sorted(complete),
        "route_profile_name": resolved_profile_name,
        "route_profile_sha256": resolved_profile_sha256,
        "valid_query_count": len(eligible),
        "buffer_records_before": before,
        "buffer_records_after": after,
        "expected_routes": expected_routes,
        "statuses": statuses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="RAG/cache")
    parser.add_argument("--train", default="RAG/data/train.json")
    parser.add_argument("--profile", choices=("anchors", "observed"), default="anchors")
    parser.add_argument("--route-plan-file", type=Path)
    args = parser.parse_args(argv)
    if args.profile == "observed":
        if args.route_plan_file is None:
            raise SystemExit("--profile observed requires --route-plan-file")
        from RAG.scripts.collect_fixed_routes import _load_observed_route_plan
        profile = _load_observed_route_plan(args.route_plan_file)
    else:
        if args.route_plan_file is not None:
            raise SystemExit("--route-plan-file requires --profile observed")
        profile = get_route_profile("anchors")
    print(
        json.dumps(
            finalize(
                args.cache_root,
                args.train,
                profile.route_map,
                profile_name=profile.name,
                profile_sha256=profile.sha256,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
