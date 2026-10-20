"""Collect train-buffer anchors, then cost-pruned diagnostic routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from env._shared.model_config import TIER_LLM_MODELS
from pipeline.value_pruned_routes import build_value_pruned_routes
from rl.route_ladder import RouteTemplate, route_key
from rl.stage_mapping import NON_SKIPPABLE_DECISION_STAGE_IDXS, NUM_RL_STAGES


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _buffer_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("trajectory buffer must be a JSON list")
    return records


def _library_payload(routes: tuple[RouteTemplate, ...]) -> list[dict]:
    return [{"name": route.name, "actions": route.actions} for route in routes]


def _completed(records: list[dict], qids: set[int], allowed: set[str]) -> dict[str, set[int]]:
    completed: dict[str, set[int]] = {}
    for row in records:
        qid = int(row["question_id"])
        key = route_key(row["actions"])
        if qid not in qids or key not in allowed:
            raise ValueError("buffer contains a request or route outside this collection design")
        if qid in completed.setdefault(key, set()):
            raise ValueError("buffer contains a duplicate request-route trajectory")
        completed[key].add(qid)
    return completed


def _collect_routes(args, routes: tuple[RouteTemplate, ...], qids: set[int], data_sha: str,
                    library_path: Path, allowed: set[str]) -> None:
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_text(json.dumps(_library_payload(routes), indent=2) + "\n", encoding="utf-8")
    capacity = len(qids) * len(allowed) + 1000
    for index, route in enumerate(routes, 1):
        completed = _completed(_buffer_records(args.buffer), qids, allowed)
        if completed.get(route_key(route.actions), set()) == qids:
            print(f"[{index}/{len(routes)}] {route.name}: complete", flush=True)
            continue
        command = [
            sys.executable, "-m", "scripts._collect_routes",
            "--data_path", str(args.data),
            "--output_path", str(args.buffer),
            "--transition_db", str(args.transition_db),
            "--operator_pkg", "env.chess",
            "--db_root", str(ROOT / "data/train"),
            "--route_library", str(library_path),
            "--route_names", route.name,
            "--collection_policy", "adaptive",
            "--min_routes_per_query", "1",
            "--max_routes_per_query", "1",
            "--trajectories_per_query", "1",
            "--max_queries", str(len(qids)),
            "--no-stop_on_first_success",
            "--append_buffer",
            "--buffer_capacity", str(capacity),
            "--autosave_every", "50",
            "--n_workers", str(args.workers),
            "--exact_only",
            "--disable_semantic_cache",
            "--cache_identity_manifest", str(args.contract),
            "--data_sha256", data_sha,
        ]
        print(f"[{index}/{len(routes)}] {route.name}: collecting", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        updated = _completed(_buffer_records(args.buffer), qids, allowed)
        if updated.get(route_key(route.actions), set()) != qids:
            raise RuntimeError(f"route {route.name} finished with missing trajectories")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("anchors", "route-library", "remaining", "all"), default="all",
        help="Collect anchors, derive the route library, collect remaining routes, or run all phases",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/train/selected/paper/selected_train.json")
    parser.add_argument("--contract", type=Path, default=ROOT / "cache/new2_exact_cache_contract.json")
    parser.add_argument("--buffer", type=Path, default=ROOT / "cache/new2.json")
    parser.add_argument("--transition-db", type=Path, default=ROOT / "cache/new2.db")
    parser.add_argument("--route-library", type=Path, default=ROOT / "cache/new2_route_library.json")
    parser.add_argument("--cost-share-threshold", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Allow paid LLM calls")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if not math.isfinite(args.cost_share_threshold) or not 0 <= args.cost_share_threshold < 1:
        parser.error("cost-share-threshold must be in [0, 1)")
    model_count = len(TIER_LLM_MODELS)
    stage_count = NUM_RL_STAGES
    optional = tuple(i for i in range(stage_count) if i not in NON_SKIPPABLE_DECISION_STAGE_IDXS)
    anchors = tuple(RouteTemplate(f"anchor_t{model + 1}", [model] * stage_count, model)
                    for model in range(model_count))
    if not args.execute and args.phase != "route-library":
        print(json.dumps({"phase": args.phase,
                          "anchors": _library_payload(anchors),
                          "threshold": args.cost_share_threshold}))
        return
    if not args.data.is_file() or not args.contract.is_file():
        raise FileNotFoundError("selected data and exact-cache contract are required")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    qids = {int(row["question_id"]) for row in data}
    if not qids or len(qids) != len(data):
        raise ValueError("selected data is empty or has duplicate request IDs")
    data_sha = sha256(args.data)
    contract_sha = sha256(args.contract)
    anchor_keys = {route_key(route.actions) for route in anchors}
    manifest_path = args.route_library.with_suffix(".manifest.json")
    prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if prior_manifest and (
        prior_manifest["data_sha256"] != data_sha
        or prior_manifest.get("contract_sha256") != contract_sha
        or prior_manifest["cost_share_threshold"] != args.cost_share_threshold
        or prior_manifest["model_count"] != model_count
        or prior_manifest["stage_count"] != stage_count
    ):
        raise ValueError("saved route library belongs to a different collection design")
    prior_keys = {route_key(row["actions"]) for row in json.loads(args.route_library.read_text(encoding="utf-8"))} \
        if prior_manifest and args.route_library.exists() else set()
    if prior_manifest and not prior_keys:
        raise ValueError("route-library manifest exists without its routes")
    if args.phase == "remaining" and not prior_manifest:
        raise FileNotFoundError("route-library manifest is required before collecting remaining routes")
    existing = _buffer_records(args.buffer)
    _completed(existing, qids, prior_keys if prior_manifest else anchor_keys)

    anchor_path = args.route_library.with_name(args.route_library.stem + "_anchors.json")
    if args.phase in ("anchors", "all"):
        _collect_routes(args, anchors, qids, data_sha, anchor_path,
                        prior_keys if prior_manifest else anchor_keys)
    if args.phase == "anchors":
        return
    records = _buffer_records(args.buffer)
    anchor_traces = [row for row in records if route_key(row["actions"]) in anchor_keys]
    completed_anchors = _completed(anchor_traces, qids, anchor_keys)
    if any(completed_anchors.get(key, set()) != qids for key in anchor_keys):
        raise ValueError("every selected request needs all homogeneous anchors before route pruning")
    result = build_value_pruned_routes(
        anchor_traces, model_count=model_count, stage_count=stage_count,
        skippable_stages=optional, cost_share_threshold=args.cost_share_threshold,
    )
    profile = [{key: row[key] for key in ("question_id", "actions", "decision_stage_costs", "total_cost")}
               for row in sorted(anchor_traces, key=lambda row: (row["question_id"], row["actions"][0]))]
    profile_sha = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    full_keys = {route_key(route.actions) for route in result.routes}
    if prior_manifest and (prior_manifest["anchor_profile_sha256"] != profile_sha or prior_keys != full_keys):
        raise ValueError("anchor cost profile differs from the saved route library")
    _completed(records, qids, full_keys)
    if args.phase in ("route-library", "all"):
        args.route_library.parent.mkdir(parents=True, exist_ok=True)
        args.route_library.write_text(json.dumps(_library_payload(result.routes), indent=2) + "\n", encoding="utf-8")
        manifest = {
            "method": "homogeneous_anchor_cost_share_v1",
            "data_sha256": data_sha,
            "contract_sha256": contract_sha,
            "anchor_profile_sha256": profile_sha,
            "model_count": model_count,
            "stage_count": stage_count,
            "cost_share_threshold": args.cost_share_threshold,
            "mean_cost_shares": result.mean_cost_shares,
            "high_cost_stages": result.high_cost_stages,
            "anchor_requests": result.anchor_requests,
            "route_count": len(result.routes),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest), flush=True)
    if args.phase in ("remaining", "all"):
        _collect_routes(args, result.routes, qids, data_sha, args.route_library, full_keys)


if __name__ == "__main__":
    main()
