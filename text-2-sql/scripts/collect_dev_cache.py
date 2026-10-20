"""Collect homogeneous dev routes into isolated evaluation caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_RUN_NAME, get_run_config
from rl.route_ladder import route_key
from rl.stage_mapping import NUM_RL_STAGES


ROOT = Path(__file__).resolve().parents[1]
ROUTES = tuple(
    {"name": f"all_t{tier + 1}", "actions": [tier] * NUM_RL_STAGES}
    for tier in range(3)
)


def _completed(buffer_path: Path, qids: set[int]) -> dict[str, set[int]]:
    completed = {route_key(route["actions"]): set() for route in ROUTES}
    if not buffer_path.exists():
        return completed
    rows = json.loads(buffer_path.read_text(encoding="utf-8"))
    for row in rows:
        qid = int(row["question_id"])
        key = route_key(row["actions"])
        if qid not in qids or key not in completed or qid in completed[key]:
            raise ValueError("dev buffer contains an unexpected or duplicate request-route pair")
        completed[key].add(qid)
    return completed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Allow paid model calls")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    config = get_run_config(DEFAULT_RUN_NAME)
    paths = {key: ROOT / str(config[key]) for key in (
        "dev_data_path", "dev_transition_db", "dev_trajectory_buffer_path",
        "dev_exact_cache_contract", "dev_stage_cost_normalizer_path",
    )}
    if not args.execute:
        print(json.dumps({"routes": ROUTES, "outputs": [str(paths["dev_trajectory_buffer_path"]),
                                                   str(paths["dev_transition_db"])],
                          "paid_calls": False}))
        return
    for key in ("dev_data_path", "dev_exact_cache_contract", "dev_stage_cost_normalizer_path"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    data_path = paths["dev_data_path"]
    records = json.loads(data_path.read_text(encoding="utf-8"))
    qids = {int(row["question_id"]) for row in records}
    if len(records) != 1534 or len(qids) != len(records):
        raise ValueError("dev collection requires 1534 distinct BIRD development requests")
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    output_path = paths["dev_trajectory_buffer_path"]
    library_path = output_path.with_name("dev_route_library.json")
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_payload = json.dumps(ROUTES, indent=2) + "\n"
    if library_path.exists() and library_path.read_text(encoding="utf-8") != library_payload:
        raise ValueError("existing dev route library differs from the expected anchors")
    library_path.write_text(library_payload, encoding="utf-8")
    for route in ROUTES:
        key = route_key(route["actions"])
        if _completed(output_path, qids)[key] == qids:
            print(f"{route['name']}: complete")
            continue
        command = [
            sys.executable, "-m", "scripts._collect_routes",
            "--run_name", DEFAULT_RUN_NAME,
            "--data_path", str(data_path),
            "--db_root", str(ROOT / str(config["dev_db_root"])),
            "--db_mode", "dev",
            "--output_path", str(output_path),
            "--transition_db", str(paths["dev_transition_db"]),
            "--route_library", str(library_path),
            "--route_names", route["name"],
            "--operator_pkg", str(config["operator_pkg"]),
            "--min_routes_per_query", "1",
            "--max_routes_per_query", "1",
            "--trajectories_per_query", "1",
            "--max_queries", str(len(records)),
            "--buffer_capacity", str(3 * len(records) + 100),
            "--autosave_every", "50",
            "--n_workers", str(args.workers),
            "--no-stop_on_first_success", "--append_buffer",
            "--exact_only", "--disable_semantic_cache",
            "--cache_identity_manifest", str(paths["dev_exact_cache_contract"]),
            "--data_sha256", data_sha256,
            "--reward_version", "ex_first_global_stage_cost_v1",
            "--stage_cost_normalizer_path", str(paths["dev_stage_cost_normalizer_path"]),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        if _completed(output_path, qids)[key] != qids:
            raise RuntimeError(f"{route['name']} collection ended with missing requests")


if __name__ == "__main__":
    main()
