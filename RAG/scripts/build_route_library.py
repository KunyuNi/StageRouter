"""Build a cost-pruned RAG route library from homogeneous-anchor traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from RAG.pipeline.value_pruned_routes import PLAN_SCHEMA, STAGE_COUNT, SKIPPABLE_STAGES, routes_from_cost_shares
from RAG.rl.counterfactual_route_ladder import ANCHOR_PROFILE


def build(anchor_buffer: Path, anchor_manifest: Path, *, cost_share_threshold: float = 0.10) -> dict:
    manifest = json.loads(anchor_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("anchor collection must be complete")
    rows = json.loads(anchor_buffer.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("anchor buffer must be a nonempty JSON list")
    grouped = defaultdict(dict)
    stage_shares: list[list[float]] = [[] for _ in range(STAGE_COUNT)]
    for row in rows:
        name = str(row["route_name"])
        if name not in ANCHOR_PROFILE.route_map:
            raise ValueError(f"unexpected anchor route: {name}")
        qid = int(row.get("question_id", row.get("qid")))
        if name in grouped[qid]:
            raise ValueError(f"duplicate anchor qid={qid} route={name}")
        if tuple(row["actions"]) != ANCHOR_PROFILE.route_map[name]:
            raise ValueError("anchor actions do not match the legal route")
        if str(row.get("workflow_identity", "")) != str(manifest["workflow_identity"]):
            raise ValueError("anchor workflow identity mismatch")
        if int(row.get("operator_version", 1)) != int(manifest.get("operator_version", 1)):
            raise ValueError("anchor operator version mismatch")
        costs = tuple(float(value) for value in row["decision_stage_costs"])
        total = float(row["total_cost"])
        if (len(costs) != STAGE_COUNT or not math.isfinite(total) or total <= 0
                or any(not math.isfinite(value) or value < 0 for value in costs)
                or sum(costs) > total + 1e-9):
            raise ValueError("anchor costs must be finite, nonnegative, and consistent with total cost")
        for stage, cost in enumerate(costs):
            stage_shares[stage].append(cost / total)
        grouped[qid][name] = row
    if any(set(routes) != set(ANCHOR_PROFILE.route_order) for routes in grouped.values()):
        raise ValueError("each eligible qid must have all three homogeneous anchors")

    shares = tuple(sum(values) / len(values) for values in stage_shares)
    profile, high_cost = routes_from_cost_shares(shares, cost_share_threshold)
    return {
        "schema_version": PLAN_SCHEMA,
        "route_profile_name": profile.name,
        "route_profile_sha256": profile.sha256,
        "anchor_buffer_sha256": hashlib.sha256(anchor_buffer.read_bytes()).hexdigest(),
        "workflow_identity": manifest["workflow_identity"],
        "operator_version": int(manifest.get("operator_version", 1)),
        "anchor_qids": len(grouped),
        "cost_share_threshold": cost_share_threshold,
        "mean_cost_shares": list(shares),
        "high_cost_stages": list(high_cost),
        "skippable_stages": list(SKIPPABLE_STAGES),
        "routes": [
            {"rank": route.rank, "name": route.name, "actions": list(route.actions)}
            for route in profile.routes
        ],
        "api_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-buffer", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-share-threshold", type=float, default=0.10)
    args = parser.parse_args(argv)
    result = build(args.anchor_buffer, args.anchor_manifest, cost_share_threshold=args.cost_share_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text(encoding="utf-8")) != result:
        raise ValueError("existing route plan differs from observed anchors")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"routes": len(result["routes"]), "high_cost_stages": result["high_cost_stages"], "anchor_qids": result["anchor_qids"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
