"""Collect RAG anchors, derive cost-pruned routes, then complete the Train buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RAG.scripts import build_route_library, collect_fixed_routes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("anchors", "route-library", "remaining", "all"), default="all")
    parser.add_argument("--dataset", type=Path, default=Path("RAG/data/train.json"))
    parser.add_argument("--qid-file", type=Path, default=Path("RAG/cache/data_selection/selected_qids.txt"))
    parser.add_argument("--cache-root", type=Path, default=Path("RAG/cache"))
    parser.add_argument("--index-cache-root", type=Path)
    parser.add_argument("--route-plan", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cost-share-threshold", type=float, default=0.10)
    parser.add_argument("--execute", action="store_true", help="Allow paid model calls during collection")
    parser.add_argument("--recover-stale-lock", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    route_plan = args.route_plan or args.cache_root / "route_plan.json"
    if not args.execute and args.phase != "route-library":
        print(json.dumps({"phase": args.phase, "dataset": str(args.dataset),
                          "qid_file": str(args.qid_file), "workers": args.workers,
                          "cost_share_threshold": args.cost_share_threshold,
                          "paid_calls_enabled": False}))
        return 0

    common = [
        "--dataset", str(args.dataset), "--qid-file", str(args.qid_file),
        "--cache-root", str(args.cache_root), "--split", "train",
        "--n-workers", str(args.workers),
    ]
    if args.index_cache_root:
        common.extend(("--index-cache-root", str(args.index_cache_root)))
    if args.recover_stale_lock:
        common.append("--recover-stale-lock")
    if args.phase in ("anchors", "all"):
        collect_fixed_routes.main(common + ["--profile", "anchors", "--artifact-name", "anchors"])
    if args.phase in ("route-library", "all"):
        build_route_library.main([
            "--anchor-buffer", str(args.cache_root / "anchors.json"),
            "--anchor-manifest", str(args.cache_root / "anchors_manifest.json"),
            "--output", str(route_plan),
            "--cost-share-threshold", str(args.cost_share_threshold),
        ])
    if args.phase in ("remaining", "all"):
        collect_fixed_routes.main(common + [
            "--profile", "observed", "--route-plan-file", str(route_plan),
            "--artifact-name", "new1", "--source-artifact", "anchors",
            "--clone-source-transitions",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
