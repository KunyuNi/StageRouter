"""Collect only the three homogeneous RAG Dev baseline routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RAG.scripts.collect_fixed_routes import main as collect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("RAG/data/test.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("RAG/exp"))
    parser.add_argument("--index-cache-root", type=Path, default=Path("RAG/cache"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Allow paid model calls")
    parser.add_argument("--recover-stale-lock", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.execute:
        print(json.dumps({"routes": ["all_t1", "all_t2", "all_t3"], "paid_calls_enabled": False}))
        return 0
    command = [
        "--dataset", str(args.dataset), "--cache-root", str(args.cache_root),
        "--index-cache-root", str(args.index_cache_root), "--split", "test",
        "--artifact-name", "dev", "--profile", "anchors",
        "--routes", "all_t1,all_t2,all_t3", "--n-workers", str(args.workers),
    ]
    if args.recover_stale_lock:
        command.append("--recover-stale-lock")
    return collect(command)


if __name__ == "__main__":
    raise SystemExit(main())
