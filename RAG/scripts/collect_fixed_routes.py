"""CLI for resumable fixed-profile RAG collection."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import sqlite3
import socket
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from RAG.env._shared.llm_adapter import get_adapter
from RAG.env._shared.model_config import TIER_LLM_MODELS
from RAG.env._shared.retriever import get_retriever
from RAG.rl.counterfactual_route_ladder import (
    ANCHOR_PROFILE,
    RouteProfile,
    RouteTemplate,
    build_random_route_profile,
    enumerate_legal_routes,
    get_route_profile,
)
from RAG.rl.env_wrapper import RAGEnvironmentWrapper
from RAG.rl.fixed_route_collector import FixedRouteCollector
from RAG.rl.random_cache_sampling import qid_selection_sha256
from RAG.rl.route_result_cache import route_key
from RAG.rl.transition_cache import TransitionCache
from RAG.rl.workflow_identity import WorkflowIdentity


def _pid_is_alive(pid: int) -> bool:
    pid = int(pid)
    if pid <= 0:
        raise ValueError("lock owner PID must be positive")
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = open_process(process_query_limited_information, False, pid)
    if handle:
        try:
            return True
        finally:
            close_handle(handle)

    error = ctypes.get_last_error()
    if error == error_invalid_parameter:
        return False
    if error == error_access_denied:
        return True
    raise ctypes.WinError(error)


def _file_identity(stat_result: os.stat_result) -> tuple[int, int]:
    return int(stat_result.st_dev), int(stat_result.st_ino)


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        current_identity = _file_identity(path.stat(follow_symlinks=False))
    except FileNotFoundError:
        return
    if current_identity != identity:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class _RecoveryGuard:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def acquire(self) -> bool:
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        self.fd = fd
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                if not os.read(fd, 1):
                    os.lseek(fd, 0, os.SEEK_SET)
                    if os.write(fd, b"\0") != 1:
                        raise OSError("failed to initialize recovery guard")
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        self._close()
                        return False
                    raise
            else:
                import fcntl

                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        self._close()
                        return False
                    raise
        except BaseException:
            self._close()
            raise
        return True

    def release(self) -> None:
        fd = self.fd
        self.fd = None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _close(self) -> None:
        fd = self.fd
        self.fd = None
        if fd is not None:
            os.close(fd)


class _SingleWriter:
    def __init__(self, path: Path, recover_stale: bool = False):
        self.path = path
        self.fd: int | None = None
        self._identity: tuple[int, int] | None = None
        self.recover_stale = recover_stale

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._acquire()
        except FileExistsError as exc:
            if not self.recover_stale or not self._recover():
                raise RuntimeError(
                    f"New1 collection is already running: {self.path}"
                ) from exc
            return self
        self._initialize_owner()
        return self

    def _acquire(self) -> None:
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            identity = _file_identity(os.fstat(fd))
        except BaseException:
            try:
                os.close(fd)
            finally:
                self.path.unlink(missing_ok=True)
            raise
        self.fd = fd
        self._identity = identity

    def _initialize_owner(self) -> None:
        if self.fd is None:
            raise RuntimeError("single-writer lock is not acquired")
        try:
            owner = {"pid": os.getpid(), "host": socket.gethostname()}
            payload = json.dumps(owner, sort_keys=True).encode("utf-8")
            offset = 0
            while offset < len(payload):
                written = os.write(self.fd, payload[offset:])
                if written <= 0:
                    raise OSError("failed to write single-writer owner metadata")
                offset += written
            os.fsync(self.fd)
        except BaseException:
            self._release()
            raise

    def __exit__(self, *_):
        self._release()

    def _release(self) -> None:
        fd = self.fd
        identity = self._identity
        self.fd = None
        self._identity = None
        try:
            if fd is not None:
                os.close(fd)
        finally:
            if identity is not None:
                _unlink_if_owned(self.path, identity)

    def _recover(self) -> bool:
        guard_path = self.path.with_name(f"{self.path.name}.recovery")
        guard = _RecoveryGuard(guard_path)
        try:
            if not guard.acquire():
                return False
        except Exception:
            return False
        try:
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
                if owner.get("host") != socket.gethostname():
                    return False
                if _pid_is_alive(int(owner["pid"])):
                    return False
            except Exception:
                return False

            try:
                self.path.unlink()
                self._acquire()
            except (FileNotFoundError, FileExistsError):
                return False
            self._initialize_owner()
            return True
        finally:
            guard.release()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _random_route_plan_payload(
    profile: RouteProfile, *, count: int, seed: int
) -> dict[str, Any]:
    legal_payload = json.dumps(
        enumerate_legal_routes(), separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": "rag_random_route_plan_v1",
        "count": int(count),
        "seed": int(seed),
        "legal_route_sha256": hashlib.sha256(legal_payload).hexdigest(),
        "route_profile_name": profile.name,
        "route_profile_sha256": profile.sha256,
        "routes": [
            {
                "rank": route.rank,
                "name": route.name,
                "actions": list(route.actions),
            }
            for route in profile.routes
        ],
    }


def _load_or_create_random_route_plan(
    path: Path, *, count: int, seed: int
) -> RouteProfile:
    expected_profile = build_random_route_profile(count=int(count), seed=int(seed))
    expected = _random_route_plan_payload(
        expected_profile, count=int(count), seed=int(seed)
    )
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                f"frozen random route plan does not match requested identity: {path}"
            )
    else:
        _atomic_json_dump(expected, path)
    routes = tuple(
        RouteTemplate(
            rank=int(row["rank"]),
            name=str(row["name"]),
            actions=tuple(int(value) for value in row["actions"]),
        )
        for row in expected["routes"]
    )
    return RouteProfile(str(expected["route_profile_name"]), routes)


def _load_observed_route_plan(path: Path) -> RouteProfile:
    from RAG.pipeline.value_pruned_routes import PLAN_SCHEMA, routes_from_cost_shares

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("route plan schema mismatch")
    routes = tuple(
        RouteTemplate(int(row["rank"]), str(row["name"]), tuple(int(x) for x in row["actions"]))
        for row in payload["routes"]
    )
    profile = RouteProfile(str(payload["route_profile_name"]), routes)
    if [route.rank for route in routes] != list(range(len(routes))):
        raise ValueError("route plan ranks must be contiguous")
    shares = tuple(float(value) for value in payload["mean_cost_shares"])
    expected, high_cost = routes_from_cost_shares(shares, float(payload["cost_share_threshold"]))
    if profile != expected or list(high_cost) != payload.get("high_cost_stages"):
        raise ValueError("route plan does not match cost-pruning rule")
    if payload.get("route_profile_sha256") != profile.sha256:
        raise ValueError("route plan hash mismatch")
    return profile


def _record_qid(record: Mapping[str, Any]) -> int:
    value = record.get("qid", record.get("question_id"))
    if value is None:
        raise ValueError("trajectory record requires qid or question_id")
    return int(value)


def _complete_source_qids(
    buffer_path: Path,
    *,
    approved_route_keys: set[str],
    workflow_identity: str,
    operator_version: int,
    split: str = "train",
) -> tuple[int, ...]:
    records = json.loads(buffer_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"source buffer must contain a JSON list: {buffer_path}")
    routes_by_qid: dict[int, list[str]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if str(record.get("status", "ok")) != "ok":
            continue
        if str(record.get("split", "train")) != str(split):
            continue
        if str(record.get("workflow_identity", "")) != str(workflow_identity):
            continue
        if int(record.get("operator_version", 1)) != int(operator_version):
            continue
        key = str(record.get("route_key") or route_key(record.get("actions", [])))
        if key in approved_route_keys:
            routes_by_qid[_record_qid(record)].append(key)
    expected_count = len(approved_route_keys)
    return tuple(
        sorted(
            qid
            for qid, keys in routes_by_qid.items()
            if len(keys) == expected_count and set(keys) == approved_route_keys
        )
    )


def _sqlite_snapshot(source_path: Path, temporary_path: Path) -> str:
    source = sqlite3.connect(source_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        destination = sqlite3.connect(temporary_path)
        try:
            source.backup(destination)
            integrity = str(destination.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                raise ValueError(f"source SQLite integrity check failed: {integrity}")
        finally:
            destination.close()
    finally:
        source.close()
    return _sha256(temporary_path)


def _clone_transition_subset(
    source_path: Path,
    destination_path: Path,
    *,
    qids: Sequence[int],
    workflow_identity: str,
    route_profile_sha256: str,
) -> dict[str, Any]:
    selected_qids = tuple(sorted({int(qid) for qid in qids}))
    if not selected_qids:
        raise ValueError("transition subset clone requires at least one qid")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.snapshot.",
        suffix=".db",
        dir=str(destination_path.parent),
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    selection_sha = qid_selection_sha256(selected_qids)
    expected_metadata = {
        "workflow_identity": str(workflow_identity),
        "source_snapshot_sha256": "",
        "selected_qid_sha256": selection_sha,
        "route_profile_sha256": str(route_profile_sha256),
    }
    try:
        source_snapshot_sha = _sqlite_snapshot(source_path, temporary_path)
        expected_metadata["source_snapshot_sha256"] = source_snapshot_sha
        if destination_path.exists():
            connection = sqlite3.connect(destination_path)
            try:
                integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
                metadata = dict(connection.execute("SELECT key, value FROM cache_meta"))
                transition_count = int(
                    connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
                )
            finally:
                connection.close()
            mismatches = {
                key: {"expected": value, "actual": metadata.get(key)}
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            if integrity != "ok" or mismatches:
                raise ValueError(
                    "transition clone identity mismatch: "
                    + json.dumps(
                        {"integrity": integrity, "mismatches": mismatches},
                        sort_keys=True,
                    )
                )
            return {
                "reused_existing_clone": True,
                "source_snapshot_sha256": source_snapshot_sha,
                "selected_qid_sha256": selection_sha,
                "selected_qids": len(selected_qids),
                "transition_count": transition_count,
            }

        connection = sqlite3.connect(temporary_path)
        try:
            connection.execute("CREATE TEMP TABLE selected_qids(qid INTEGER PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO selected_qids(qid) VALUES(?)",
                ((qid,) for qid in selected_qids),
            )
            connection.execute(
                "DELETE FROM transitions WHERE question_id NOT IN "
                "(SELECT qid FROM selected_qids)"
            )
            for key, value in expected_metadata.items():
                connection.execute(
                    "INSERT OR REPLACE INTO cache_meta(key,value) VALUES(?,?)",
                    (key, value),
                )
            connection.commit()
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                raise ValueError(f"destination SQLite integrity check failed: {integrity}")
            transition_count = int(
                connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
            )
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode=DELETE")
        finally:
            connection.close()
        with temporary_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination_path)
        return {
            "reused_existing_clone": False,
            "source_snapshot_sha256": source_snapshot_sha,
            "selected_qid_sha256": selection_sha,
            "selected_qids": len(selected_qids),
            "transition_count": transition_count,
        }
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_path.with_name(f"{temporary_path.name}-wal").unlink(missing_ok=True)
        temporary_path.with_name(f"{temporary_path.name}-shm").unlink(missing_ok=True)


def _identity(
    train_path: Path,
    cache_root: Path,
    index_cache_root: Path | None = None,
) -> WorkflowIdentity:
    corpus_sha = ""
    split_manifest = train_path.parent / "split_manifest.json"
    if split_manifest.exists():
        try:
            manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
            corpus_sha = str(
                manifest.get("source_files", {})
                .get("corpus.json", {})
                .get("sha256", "")
            )
        except (OSError, ValueError, TypeError):
            corpus_sha = ""
    index_identity = ""
    index_root = cache_root if index_cache_root is None else index_cache_root
    index_manifest = index_root / "vector_index" / "manifest.json"
    if index_manifest.exists():
        try:
            index_identity = str(json.loads(index_manifest.read_text(encoding="utf-8")).get("index_identity", ""))
        except (OSError, ValueError, TypeError):
            index_identity = ""
    return WorkflowIdentity.from_inputs(
        train_dataset_sha256=_sha256(train_path),
        corpus_sha256=corpus_sha,
        index_identity=index_identity,
        tier_models=TIER_LLM_MODELS,
        embedding_model="text-embedding-v3",
    )


def _load_train(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"train dataset must be a JSON list: {path}")
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("question_id") is None:
            raise ValueError("every train row requires question_id")
        result.append(dict(row))
    return result


def _select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_qid = {int(row["question_id"]): row for row in rows}
    if args.qid_file:
        qids = [int(value.strip()) for value in args.qid_file.read_text(encoding="utf-8").splitlines() if value.strip()]
        if not qids or len(qids) != len(set(qids)):
            raise ValueError("qid file must contain unique qids")
        missing = sorted(set(qids) - set(by_qid))
        if missing:
            raise ValueError(f"selected qids are not in the dataset: {missing}")
        return [by_qid[qid] for qid in qids]
    if args.qids:
        qids = []
        for value in args.qids.split(","):
            value = value.strip()
            if value:
                qids.append(int(value))
        missing = sorted(set(qids) - set(by_qid))
        if missing:
            raise ValueError(f"requested qids are not in train split: {missing}")
        return [by_qid[qid] for qid in qids]
    if args.smoke_four_types:
        selected: list[dict[str, Any]] = []
        prefixes = ("comparison", "inference", "temporal", "null")
        for prefix in prefixes:
            matches = [
                row
                for row in rows
                if str(row.get("question_type", "")).lower().startswith(prefix)
            ]
            if not matches:
                raise ValueError(f"train split has no question_type matching {prefix!r}")
            selected.append(min(matches, key=lambda row: int(row["question_id"])))
        return selected
    return rows


def _load_known_exclusions(
    cache_root: Path, artifact_name: str = "new1"
) -> set[int]:
    manifest_path = cache_root / f"{artifact_name}_manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("new1_manifest.json must contain an object")
    try:
        return {int(value) for value in manifest.get("excluded_query_ids", [])}
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest excluded_query_ids is invalid") from exc


def _route_names(
    value: str,
    route_map: Mapping[str, Sequence[int]],
    route_order: Sequence[str],
) -> list[str]:
    if value == "all":
        return list(route_order)
    selected = [name.strip() for name in value.split(",") if name.strip()]
    if not selected:
        raise ValueError("--routes requires 'all' or at least one route name")
    unknown = sorted(set(selected) - set(route_map))
    if unknown:
        raise ValueError(f"Unknown routes for selected profile: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("--routes contains duplicate route names")
    return selected


def _dry_run_report(
    *,
    profile: Any,
    eligible_rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    workflow_identity: str,
    operator_version: int,
    known_excluded: set[int],
    artifact_name: str = "new1",
) -> dict[str, Any]:
    eligible_qids = {int(row["question_id"]) for row in eligible_rows}
    canonical_keys = {
        route_key(actions) for actions in profile.route_map.values()
    }
    reused: set[tuple[int, str]] = set()
    buffer_path = cache_root / f"{artifact_name}.json"
    if buffer_path.exists():
        records = json.loads(buffer_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("new1.json must contain a list")
        for record in records:
            if not isinstance(record, Mapping) or record.get("status", "ok") != "ok":
                continue
            try:
                qid = int(record.get("question_id", record.get("qid")))
                record_key = str(
                    record.get("route_key") or route_key(record.get("actions", []))
                )
                record_version = int(record.get("operator_version", 1))
            except (TypeError, ValueError):
                continue
            if (
                qid in eligible_qids
                and record_key in canonical_keys
                and record_version == int(operator_version)
                and str(record.get("workflow_identity", "")) == workflow_identity
            ):
                reused.add((qid, record_key))
    target_pairs = len(eligible_rows) * len(profile.route_map)
    return {
        "route_profile_name": profile.name,
        "route_profile_sha256": profile.sha256,
        "candidate_qids": len(eligible_rows),
        "routes_per_query": len(profile.route_map),
        "target_pairs": target_pairs,
        "reused_pairs": len(reused),
        "missing_pairs": target_pairs - len(reused),
        "known_excluded_query_ids": sorted(known_excluded),
    }


def _random_route_cost_estimate(
    *,
    source_buffer_path: Path,
    destination_buffer_path: Path,
    profile: RouteProfile,
    qids: Sequence[int],
    workflow_identity: str,
    operator_version: int,
    split: str,
    source_route_keys: set[str],
) -> dict[str, Any]:
    source_records = json.loads(source_buffer_path.read_text(encoding="utf-8"))
    if not isinstance(source_records, list):
        raise ValueError("source buffer must contain a JSON list")
    selected_qids = {int(qid) for qid in qids}
    stage_action_costs: dict[tuple[int, int], list[float]] = defaultdict(list)
    fixed_costs: list[float] = []
    for record in source_records:
        if not isinstance(record, Mapping):
            continue
        if _record_qid(record) not in selected_qids:
            continue
        if str(record.get("status", "ok")) != "ok":
            continue
        if str(record.get("split", "train")) != split:
            continue
        if str(record.get("workflow_identity", "")) != workflow_identity:
            continue
        if int(record.get("operator_version", 1)) != int(operator_version):
            continue
        actions = tuple(int(value) for value in record.get("actions", []))
        decision_costs = tuple(
            float(value) for value in record.get("decision_stage_costs", [])
        )
        if len(actions) == 4 and len(decision_costs) == 4:
            for stage_index, (action, cost) in enumerate(
                zip(actions, decision_costs)
            ):
                stage_action_costs[(stage_index, action)].append(cost)
        fixed = tuple(float(value) for value in record.get("fixed_stage_costs", []))
        if fixed:
            fixed_costs.append(sum(fixed))
    mean_stage_action_cost = {
        key: sum(values) / len(values) for key, values in stage_action_costs.items()
    }
    mean_fixed_cost = sum(fixed_costs) / len(fixed_costs) if fixed_costs else 0.0

    destination_hits: set[tuple[int, str]] = set()
    if destination_buffer_path.exists():
        destination_records = json.loads(
            destination_buffer_path.read_text(encoding="utf-8")
        )
        if not isinstance(destination_records, list):
            raise ValueError("destination buffer must contain a JSON list")
        for record in destination_records:
            if not isinstance(record, Mapping):
                continue
            if str(record.get("status", "ok")) != "ok":
                continue
            if str(record.get("split", "train")) != split:
                continue
            if str(record.get("workflow_identity", "")) != workflow_identity:
                continue
            if int(record.get("operator_version", 1)) != int(operator_version):
                continue
            qid = _record_qid(record)
            if qid not in selected_qids:
                continue
            key = str(
                record.get("route_key") or route_key(record.get("actions", []))
            )
            destination_hits.add((qid, key))

    estimated_cost = 0.0
    estimated_api_pairs = 0
    exact_complete_hits = 0
    for qid in selected_qids:
        for actions in profile.route_map.values():
            key = route_key(actions)
            if (qid, key) in destination_hits:
                exact_complete_hits += 1
                continue
            if key in source_route_keys:
                exact_complete_hits += 1
                continue
            route_cost = mean_fixed_cost
            for stage_index, action in enumerate(actions):
                lookup = (stage_index, int(action))
                if lookup not in mean_stage_action_cost:
                    raise ValueError(
                        "source buffer lacks an empirical cost for "
                        f"decision stage/action {lookup}"
                    )
                route_cost += mean_stage_action_cost[lookup]
            estimated_cost += route_cost
            estimated_api_pairs += 1
    return {
        "estimated_incremental_cost": estimated_cost,
        "estimated_api_route_pairs_before_transition_reuse": estimated_api_pairs,
        "estimated_complete_route_hits": exact_complete_hits,
        "empirical_mean_fixed_retrieval_cost": mean_fixed_cost,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        "--dataset",
        dest="dataset",
        type=Path,
        default=Path("RAG/data/train.json"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("RAG/cache"))
    parser.add_argument("--index-cache-root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--artifact-name", default="new1")
    parser.add_argument(
        "--profile",
        choices=("anchors", "observed"),
        default=None,
    )
    parser.add_argument(
        "--routes",
        "--baseline",
        dest="routes",
        default="all",
        help="all or a comma-separated list of names from the selected profile",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qids", help="comma-separated train qids")
    parser.add_argument("--qid-file", type=Path, help="one selected train qid per line")
    parser.add_argument("--smoke-four-types", action="store_true")
    parser.add_argument("--autosave-every", type=int, default=10)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--operator-version", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--recover-stale-lock", action="store_true")
    parser.add_argument("--random-route-count", type=int)
    parser.add_argument("--sampling-seed", type=int, default=1145)
    parser.add_argument("--route-plan-file", type=Path)
    parser.add_argument("--source-artifact", default="new1")
    parser.add_argument("--source-route-plan-file", type=Path)
    parser.add_argument("--clone-source-transitions", action="store_true")
    args = parser.parse_args(argv)
    if args.n_workers <= 0:
        raise SystemExit("--n-workers must be positive")
    if sum(bool(value) for value in (args.qids, args.qid_file, args.smoke_four_types)) > 1:
        raise SystemExit("--qids, --qid-file and --smoke-four-types are mutually exclusive")
    identity = _identity(
        args.dataset,
        args.cache_root,
        args.index_cache_root,
    )
    all_rows = _load_train(args.dataset)
    random_mode = args.random_route_count is not None
    random_context: dict[str, Any] | None = None
    if random_mode:
        if args.profile is not None:
            raise SystemExit("--profile cannot be combined with --random-route-count")
        if args.routes != "all":
            raise SystemExit("--routes cannot be combined with --random-route-count")
        if args.qids or args.qid_file or args.smoke_four_types:
            raise SystemExit(
                "random-route mode always uses all complete source qids"
            )
        if args.route_plan_file is None:
            raise SystemExit("--route-plan-file is required in random-route mode")
        if args.source_artifact == args.artifact_name:
            raise SystemExit("--source-artifact must differ from --artifact-name")
        if args.source_route_plan_file is None:
            raise SystemExit("--source-route-plan-file is required in random-route mode")
        if not args.clone_source_transitions:
            raise SystemExit(
                "--clone-source-transitions is required in random-route mode"
            )
        profile = _load_or_create_random_route_plan(
            args.route_plan_file,
            count=args.random_route_count,
            seed=args.sampling_seed,
        )
        routes = list(profile.route_order)
        source_manifest_path = (
            args.cache_root / f"{args.source_artifact}_manifest.json"
        )
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(source_manifest, Mapping):
            raise ValueError("source manifest must contain an object")
        if str(source_manifest.get("status", "")) != "completed":
            raise ValueError("source manifest status must be completed")
        if str(source_manifest.get("workflow_identity", "")) != identity.sha256:
            raise ValueError("source manifest workflow identity mismatch")
        if int(source_manifest.get("operator_version", 1)) != args.operator_version:
            raise ValueError("source manifest operator version mismatch")
        source_profile = _load_observed_route_plan(args.source_route_plan_file)
        if str(source_manifest.get("route_profile_sha256", "")) != source_profile.sha256:
            raise ValueError("source manifest and source route plan hash mismatch")
        source_route_keys = {route_key(actions) for actions in source_profile.route_map.values()}
        source_buffer_path = args.cache_root / f"{args.source_artifact}.json"
        source_complete_qids = _complete_source_qids(
            source_buffer_path,
            approved_route_keys=source_route_keys,
            workflow_identity=identity.sha256,
            operator_version=args.operator_version,
            split=args.split,
        )
        source_complete_set = set(source_complete_qids)
        source_excluded = _load_known_exclusions(
            args.cache_root, args.source_artifact
        )
        destination_excluded = _load_known_exclusions(
            args.cache_root, args.artifact_name
        )
        known_excluded = source_excluded | destination_excluded
        eligible_rows = [
            row
            for row in all_rows
            if int(row["question_id"]) in source_complete_set
            and int(row["question_id"]) not in known_excluded
        ]
        if len(eligible_rows) != len(source_complete_set - known_excluded):
            raise ValueError("complete source qids do not exactly match the dataset")
        rows = eligible_rows
        random_context = {
            "source_artifact": str(args.source_artifact),
            "source_buffer": source_buffer_path,
            "source_db": args.cache_root / f"{args.source_artifact}.db",
            "source_complete_qids": source_complete_qids,
            "route_plan_file": args.route_plan_file,
            "route_plan_file_sha256": _sha256(args.route_plan_file),
            "source_route_keys": source_route_keys,
            "source_actions": set(source_profile.route_map.values()),
        }
    else:
        if args.profile == "observed":
            if args.route_plan_file is None:
                raise ValueError("--profile observed requires --route-plan-file")
            if args.clone_source_transitions and args.source_artifact == args.artifact_name:
                raise ValueError("source and destination artifacts must differ")
            plan = json.loads(args.route_plan_file.read_text(encoding="utf-8"))
            if str(plan.get("workflow_identity")) != identity.sha256:
                raise ValueError("observed route plan workflow identity mismatch")
            if int(plan.get("operator_version", 1)) != args.operator_version:
                raise ValueError("observed route plan operator version mismatch")
            profile = _load_observed_route_plan(args.route_plan_file)
            if args.clone_source_transitions:
                source_manifest_path = args.cache_root / f"{args.source_artifact}_manifest.json"
                source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
                if (source_manifest.get("status") != "completed"
                        or source_manifest.get("workflow_identity") != identity.sha256
                        or int(source_manifest.get("operator_version", 1)) != args.operator_version
                        or source_manifest.get("route_profile_sha256") != ANCHOR_PROFILE.sha256):
                    raise ValueError("anchor source manifest identity mismatch")
        else:
            if args.route_plan_file is not None:
                raise ValueError("--route-plan-file requires --profile observed")
            if args.clone_source_transitions:
                raise ValueError("transition cloning requires an observed route plan")
            profile = get_route_profile("anchors")
        routes = _route_names(args.routes, profile.route_map, profile.route_order)
        known_excluded = _load_known_exclusions(
            args.cache_root, args.artifact_name
        )
        if args.profile == "observed" and args.clone_source_transitions:
            known_excluded |= _load_known_exclusions(args.cache_root, args.source_artifact)
        eligible_rows = [
            row
            for row in all_rows
            if int(row["question_id"]) not in known_excluded
        ]
        rows = _select_rows(eligible_rows, args)
    if args.dry_run:
        dry_run_report = _dry_run_report(
            profile=profile,
            eligible_rows=eligible_rows,
            cache_root=args.cache_root,
            workflow_identity=identity.sha256,
            operator_version=args.operator_version,
            known_excluded=known_excluded,
            artifact_name=args.artifact_name,
        )
        if random_context is not None:
            overlap = sum(
                1 for actions in profile.route_map.values()
                if tuple(actions) in random_context["source_actions"]
            )
            cost_estimate = _random_route_cost_estimate(
                source_buffer_path=random_context["source_buffer"],
                destination_buffer_path=(
                    args.cache_root / f"{args.artifact_name}.json"
                ),
                profile=profile,
                qids=[int(row["question_id"]) for row in eligible_rows],
                workflow_identity=identity.sha256,
                operator_version=args.operator_version,
                split=args.split,
                source_route_keys=random_context["source_route_keys"],
            )
            dry_run_report.update(
                {
                    "source_artifact": random_context["source_artifact"],
                    "source_complete_qids": len(
                        random_context["source_complete_qids"]
                    ),
                    "canonical_overlap_routes": overlap,
                    "route_plan_file": str(random_context["route_plan_file"]),
                    "route_plan_file_sha256": random_context[
                        "route_plan_file_sha256"
                    ],
                    **cost_estimate,
                }
            )
        print(
            json.dumps(
                dry_run_report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    lock_path = args.cache_root / f".{args.artifact_name}_collection.lock"
    with _SingleWriter(lock_path, recover_stale=args.recover_stale_lock):
        clone_report: dict[str, Any] | None = None
        if random_context is not None:
            clone_report = _clone_transition_subset(
                random_context["source_db"],
                args.cache_root / f"{args.artifact_name}.db",
                qids=random_context["source_complete_qids"],
                workflow_identity=identity.sha256,
                route_profile_sha256=profile.sha256,
            )
        elif args.profile == "observed" and args.clone_source_transitions:
            source_buffer = args.cache_root / f"{args.source_artifact}.json"
            source_qids = _complete_source_qids(
                source_buffer,
                approved_route_keys={route_key(actions) for actions in ANCHOR_PROFILE.route_map.values()},
                workflow_identity=identity.sha256,
                operator_version=args.operator_version,
                split=args.split,
            )
            expected_qids = {int(row["question_id"]) for row in rows}
            if not expected_qids or set(source_qids) != expected_qids:
                raise ValueError("selected training requests lack complete anchor trajectories")
            clone_report = _clone_transition_subset(
                args.cache_root / f"{args.source_artifact}.db",
                args.cache_root / f"{args.artifact_name}.db",
                qids=source_qids,
                workflow_identity=identity.sha256,
                route_profile_sha256=profile.sha256,
            )
        transition_cache = TransitionCache(
            args.cache_root / f"{args.artifact_name}.db",
            workflow_identity=identity.sha256,
        )
        try:
            wrapper = RAGEnvironmentWrapper(
                adapter=get_adapter(),
                retriever=get_retriever(),
                index_identity=identity.index_identity,
            )
            collector = FixedRouteCollector(
                args.cache_root,
                wrapper,
                transition_cache,
                identities=identity,
                operator_version=args.operator_version,
                autosave_every=args.autosave_every,
                expected_routes=len(eligible_rows) * len(profile.route_map),
                routes=profile.route_map,
                route_order=profile.route_order,
                route_profile_name=profile.name,
                route_profile_sha256=profile.sha256,
                excluded_query_ids=known_excluded,
                n_workers=args.n_workers,
                max_inflight=args.n_workers * 2,
                split=args.split,
                artifact_name=args.artifact_name,
            )
            if random_context is not None:
                collector._manifest_extra.update(
                    {
                        "sampling_seed": int(args.sampling_seed),
                        "source_artifact": random_context["source_artifact"],
                        "source_complete_qids": len(
                            random_context["source_complete_qids"]
                        ),
                        "source_qid_sha256": qid_selection_sha256(
                            random_context["source_complete_qids"]
                        ),
                        "route_plan_file": str(random_context["route_plan_file"]),
                        "route_plan_file_sha256": random_context[
                            "route_plan_file_sha256"
                        ],
                        "transition_clone": clone_report,
                    }
                )
            elif args.profile == "observed":
                collector._manifest_extra.update({
                    "route_plan_file": str(args.route_plan_file),
                    "route_plan_file_sha256": _sha256(args.route_plan_file),
                    "transition_clone": clone_report,
                })
            try:
                report = collector.collect(rows, routes, cache_only=args.cache_only)
            except KeyboardInterrupt:
                collector._stop_reason = "KeyboardInterrupt"
                collector.save(status="stopped")
                raise
        finally:
            transition_cache.close()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
