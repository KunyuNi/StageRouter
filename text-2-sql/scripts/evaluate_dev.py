#!/usr/bin/env python3
"""Evaluate the locked router on the development workflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, local
from typing import Any, Callable, Iterable, Iterator

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from config import get_run_config
from config.new2_state_v26 import LOCKED_RUN_NAME
from rl.dataset_utils import load_question_records, question_map_from_records
from rl.env_wrapper import EnvironmentWrapper
from rl.state_v26_evaluation import (
    load_state_v26_evaluation_bundle,
)
from rl.transition_cache import TransitionCache
from scripts._dev_runtime import (
    DEFAULT_LLM_MODEL,
    DevTrajectoryCache,
    default_model_config_version,
    dev_environment_kwargs,
    evaluate_query,
)
from rl.exact_cache_identity import load_exact_cache_identity_manifest


def _bind_exact_operator_namespace(project_root: Path) -> None:
    """Resolve exact Dev operators from the source frozen by the cache contract."""
    import env

    env_root = (project_root / "env").resolve()
    if not env_root.is_dir():
        raise ValueError(f"missing exact operator source directory: {env_root}")
    loaded = sys.modules.get("env.chess")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None or not Path(loaded_file).resolve().is_relative_to(env_root):
            raise RuntimeError("env.chess was loaded before exact Dev source binding")
    remaining = [
        path
        for path in env.__path__
        if Path(path).resolve() != env_root
    ]
    env.__path__[:] = [str(env_root), *remaining]
    package = importlib.import_module("env.chess")
    package_file = getattr(package, "__file__", None)
    if package_file is None or not Path(package_file).resolve().is_relative_to(env_root):
        raise RuntimeError("env.chess did not resolve from the exact Dev source")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.preparing")
    temporary.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sqlite_integrity(path: Path) -> str:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        result = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()
    if result != "ok":
        raise ValueError(f"SQLite integrity failed for {path}: {result}")
    return result


def _summary(rows: list[dict[str, Any]], *, run_name: str, checkpoint_sha: str) -> dict[str, Any]:
    qids = [int(row["question_id"]) for row in rows]
    costs = np.asarray([float(row["total_cost"]) for row in rows])
    ex = np.asarray([int(row["final_ex"]) for row in rows])
    actions = [int(row["actions"][1]) for row in rows]
    if len(rows) != 1534 or len(set(qids)) != 1534:
        raise ValueError("State V26 Dev evaluation requires 1534 unique qids")
    result = {
        "run_name": run_name,
        "checkpoint_sha256": checkpoint_sha,
        "qid_count": len(rows),
        "unique_qid_count": len(set(qids)),
        "ex": float(ex.mean()),
        "mean_cost": float(costs.mean()),
        "total_cost": float(costs.sum()),
        "s4_tier_counts": {
            f"T{action + 1}": actions.count(action) for action in range(3)
        },
        "transition_cache_hits": sum(
            int(row["transition_cache"]["hits"]) for row in rows
        ),
        "transition_cache_misses": sum(
            int(row["transition_cache"]["misses"]) for row in rows
        ),
        "route_cache_hits": sum(row["route_result_cache"] == "hit" for row in rows),
        "route_cache_misses": sum(row["route_result_cache"] == "miss" for row in rows),
        "status": "complete",
        "system_error": "",
    }
    if not all(
        math.isfinite(float(result[field]))
        for field in ("ex", "mean_cost", "total_cost")
    ):
        raise ValueError("State V26 Dev evaluation produced nonfinite metrics")
    return result


def _validate_run_mode(run_name: str) -> None:
    if run_name != LOCKED_RUN_NAME:
        raise ValueError("Only the locked State V26 run is supported")


class _StateV26PolicyAdapter:
    """Expose scalar State V26 Q-values through the existing Dev policy API."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.stage_idx = 0
        self.decisions: list[dict[str, Any]] = []

    def select_action(self, state: Any, mask: Any) -> int:
        encoded = np.asarray(state, dtype=np.float32)
        legal = np.asarray(mask, dtype=np.float32)
        if encoded.shape != (5256,) or not np.isfinite(encoded).all():
            raise ValueError("State V26 Dev requires finite float32[5256]")
        if self.stage_idx >= 6:
            raise ValueError("State V26 Dev received too many decisions")
        if self.stage_idx == 1 and not np.array_equal(legal, [1, 1, 1, 0]):
            raise ValueError("State V26 S4 must expose exactly T1/T2/T3")
        q_values = np.asarray(self.policy.q_values(encoded, legal), dtype=np.float64)
        if q_values.shape != (4,) or not np.isfinite(q_values[legal.astype(bool)]).all():
            raise ValueError("State V26 policy produced invalid legal Q values")
        action = int(np.where(legal.astype(bool), q_values, -np.inf).argmax())
        if legal[action] != 1:
            raise ValueError("State V26 policy selected an illegal action")
        self.decisions.append(
            {
                "stage_idx": self.stage_idx,
                "action": action,
                "q_values": q_values.tolist(),
                "action_mask": legal.astype(int).tolist(),
            }
        )
        self.stage_idx += 1
        return int(action)


class _SingleThreadInference:
    """Own the encoder, semantic SQLite connection, and GPU policy on one thread."""

    def __init__(self, loader: Callable[[], Any]) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="state-v26-inference",
        )
        try:
            self._bundle = self._executor.submit(loader).result()
        except BaseException:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise
        self.config = self._bundle.config
        self.checkpoint_sha256 = self._bundle.checkpoint_sha256
        self.checkpoint_path = self._bundle.checkpoint_path
        self.state_version = self._bundle.encoder.state_version
        self.state_dim = self._bundle.encoder.state_dim
        self.encoder = self
        self.policy = self
        self._closed = False

    def encode_question(self, *args: Any, **kwargs: Any) -> Any:
        return self._executor.submit(
            self._bundle.encoder.encode_question,
            *args,
            **kwargs,
        ).result()

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self._executor.submit(
            self._bundle.encoder.encode,
            *args,
            **kwargs,
        ).result()

    def q_values(self, *args: Any, **kwargs: Any) -> Any:
        return self._executor.submit(
            self._bundle.policy.q_values,
            *args,
            **kwargs,
        ).result()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._executor.submit(self._bundle.close).result()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "_SingleThreadInference":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class _LockedDevTrajectoryCache:
    """Keep the established JSON route cache behind one in-process writer."""

    def __init__(self, cache: Any) -> None:
        self._cache = cache
        self._lock = Lock()

    def get(self, **kwargs: Any) -> Any:
        with self._lock:
            return self._cache.get(**kwargs)

    def append(self, record: Any) -> None:
        with self._lock:
            self._cache.append(record)


def _bounded_ordered_map(
    function: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: int,
) -> Iterator[Any]:
    """Reuse New2's bounded worker pattern while yielding deterministic order."""
    if max_workers <= 0:
        raise ValueError("n_workers must be positive")
    if max_workers == 1:
        yield from map(function, items)
        return

    indexed = iter(enumerate(items))
    ready: dict[int, Any] = {}
    next_output = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending: dict[Any, int] = {}

        def submit_one() -> bool:
            try:
                index, item = next(indexed)
            except StopIteration:
                return False
            pending[pool.submit(function, item)] = index
            return True

        for _ in range(max_workers):
            if not submit_one():
                break
        try:
            while pending:
                future = next(as_completed(tuple(pending)))
                index = pending.pop(future)
                ready[index] = future.result()
                submit_one()
                while next_output in ready:
                    yield ready.pop(next_output)
                    next_output += 1
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def _load_resume_prefix(
    path: Path,
    records: list[dict[str, Any]],
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not resume:
        raise FileExistsError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid resume JSONL line {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError("resume JSONL rows must be objects")
            rows.append(row)
    if len(rows) > len(records):
        raise ValueError("resume output contains more rows than the Dev split")
    for index, row in enumerate(rows):
        qid = int(row.get("qid", row.get("question_id", -1)))
        if qid != int(records[index]["question_id"]):
            raise ValueError("resume output is not an exact Dev qid prefix")
        if row.get("status") != "ok" or row.get("system_error"):
            raise ValueError("resume output contains an invalid prediction")
    return rows


def _evaluate_query(
    *,
    item: dict[str, Any],
    run_name: str,
    bundle: Any,
    env: Any,
    transition_cache: Any,
    route_cache: Any,
    data_sha256: str,
    model_config_version: str,
) -> dict[str, Any]:
    policy = _StateV26PolicyAdapter(bundle.policy)
    prediction, _ = evaluate_query(
        item=item,
        run_name=run_name,
        checkpoint_path=str(bundle.checkpoint_path),
        checkpoint_sha256=bundle.checkpoint_sha256,
        data_sha256=data_sha256,
        split="dev",
        route_cache=route_cache,
        env=env,
        transition_cache=transition_cache,
        state_encoder=bundle.encoder,
        policy=policy,
        selection_config=None,
        operator_pkg=str(bundle.config["operator_pkg"]),
        operator_version=1,
        model_config_version=model_config_version,
    )
    if prediction["status"] != "ok" or prediction["system_error"] or policy.stage_idx != 6:
        raise ValueError("State V26 formal Dev requires an error-free full route")
    return {
        **prediction,
        "question_id": int(prediction["qid"]),
        "final_ex": int(prediction["exec_res"]),
        "decisions": policy.decisions,
    }


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.n_workers <= 0:
        raise ValueError("--n_workers must be positive")
    _validate_run_mode(args.run_name)

    root = Path(args.project_root).resolve()
    _bind_exact_operator_namespace(root)
    load_dotenv(root.parent / ".env", override=True)
    data_path = root / "data/dev/dev.json"
    records = load_question_records(data_path)
    if len(records) != 1534:
        raise ValueError("State V26 Dev requires exactly 1534 source rows")
    question_map = question_map_from_records(records)
    os.environ["NL2SQL_DB_ROOT"] = str((root / "data/dev").resolve())
    os.environ["NL2SQL_DB_MODE"] = "dev"
    from env.runner.database_manager import DatabaseManager

    manager_state = local()

    def manager(db_id: str):
        managers = getattr(manager_state, "managers", None)
        if managers is None:
            managers = {}
            manager_state.managers = managers
        if db_id not in managers:
            managers[db_id] = DatabaseManager(db_mode="dev", db_id=db_id)
        return managers[db_id]

    config = get_run_config(args.run_name)
    cache_identity = load_exact_cache_identity_manifest(
        root / str(config["dev_exact_cache_contract"]),
        question_map,
        operator_pkg=str(config["operator_pkg"]),
        project_root=root,
        db_root=root / str(config["dev_db_root"]),
        db_mode="dev",
    )
    env_kwargs = dev_environment_kwargs(
        config,
        db_manager_factory=manager,
        total_budget=10.0,
        cache_identity=cache_identity,
    )
    env_kwargs["model"] = DEFAULT_LLM_MODEL
    environment_state = local()

    def worker_environment() -> EnvironmentWrapper:
        environment = getattr(environment_state, "environment", None)
        if environment is None:
            environment = EnvironmentWrapper(**env_kwargs)
            environment.set_question_map(question_map)
            environment_state.environment = environment
        return environment
    lock_path = root / "tmp/eval_state_v26_dev.writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("another State V26 Dev writer is active") from exc
    os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
    transition_path = root / str(config["dev_transition_db"])
    transition_cache = None
    try:
        _sqlite_integrity(transition_path)
        transition_cache = TransitionCache(transition_path, allow_semantic_lookup=False)
        route_cache = _LockedDevTrajectoryCache(
            DevTrajectoryCache(
                root / str(config["dev_trajectory_buffer_path"]),
                normalizer_path=str(root / str(config["dev_stage_cost_normalizer_path"])),
            )
        )
        data_sha256 = _sha256(data_path)
        model_config_version = default_model_config_version()
        def load_bundle() -> Any:
            return load_state_v26_evaluation_bundle(
                project_root=root,
                run_name=args.run_name,
                device=args.device,
                split="dev",
                manager_factory=manager,
                checkpoint_path=args.checkpoint_path,
            )

        with _SingleThreadInference(load_bundle) as bundle:
            prediction_path = Path(args.prediction_path) if args.prediction_path else (
                root / "model" / args.run_name / "dev_predictions.jsonl"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            rows = _load_resume_prefix(prediction_path, records, resume=args.resume)

            def evaluate_item(item: dict[str, Any]) -> dict[str, Any]:
                row = _evaluate_query(
                    item=item,
                    run_name=args.run_name,
                    bundle=bundle,
                    env=worker_environment(),
                    transition_cache=transition_cache,
                    route_cache=route_cache,
                    data_sha256=data_sha256,
                    model_config_version=model_config_version,
                )
                row.update(
                    {
                        "run_name": args.run_name,
                        "checkpoint_sha256": bundle.checkpoint_sha256,
                        "s4_gate": "off",
                    }
                )
                return row

            remaining = records[len(rows) :]
            mode = "a" if prediction_path.exists() else "x"
            with prediction_path.open(mode, encoding="utf-8") as stream:
                for ordinal, row in enumerate(
                    _bounded_ordered_map(
                        evaluate_item, remaining, max_workers=args.n_workers,
                    ),
                    start=len(rows) + 1,
                ):
                    rows.append(row)
                    stream.write(
                        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    if ordinal % 25 == 0:
                        print(f"dev={ordinal}/1534", flush=True)
            summary = {
                **_summary(rows, run_name=args.run_name, checkpoint_sha=bundle.checkpoint_sha256),
                "evaluation_mode": "s4_gate_off",
                "prediction_file_sha256": _sha256(prediction_path),
            }
            summary_path = Path(args.summary_path) if args.summary_path else prediction_path.with_name(
                prediction_path.stem + ".summary.json"
            )
            _atomic_json(summary_path, summary)
            print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
        _sqlite_integrity(transition_path)
    finally:
        if transition_cache is not None:
            transition_cache.close()
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
