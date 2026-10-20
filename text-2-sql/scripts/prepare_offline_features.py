#!/usr/bin/env python3
"""Prepare immutable semantics and Candidate-EX labels from the train buffer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.new2_state_v26 import CONTROL_ROOT

DECISION_COUNT = 6
NON_SKIPPABLE_DECISIONS = (1, 4)  # S4 and S7
EVALUATOR_TIMEOUT_SECONDS = 30
REPLAY_CHECKPOINT_VERSION = "state_v25_new2_replay_checkpoint_v1"
REPLAY_CHECKPOINT_SYNC_INTERVAL = 1_000
CACHE_MISS_REPORT_VERSION = "state_v25_new2_cache_miss_report_v1"
CACHE_MISS_PATTERN = re.compile(
    r"^Cache-only replay MISS for qid=(\d+) stage=([A-Za-z0-9_]+) tier=(\d+)$"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _actions(raw: Any) -> tuple[int, ...]:
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or len(raw) != DECISION_COUNT
    ):
        raise ValueError("every New2 route must contain six actions")
    actions = tuple(raw)
    if any(type(action) is not int or action not in range(4) for action in actions):
        raise ValueError("New2 actions must be integer ids in [0, 3]")
    return actions


def _replay_key(item: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    qid = item.get("question_id")
    if isinstance(qid, bool) or type(qid) is not int:
        raise ValueError("replay checkpoint row has invalid question_id")
    return qid, _actions(item.get("actions"))


def _validated_replay_checkpoint_row(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("replay checkpoint row is not a mapping")
    if raw.get("version") != REPLAY_CHECKPOINT_VERSION:
        raise ValueError("replay checkpoint version mismatch")
    qid, actions = _replay_key(raw)
    candidate_sql = raw.get("candidate_sql")
    final_sql = raw.get("final_sql")
    frozen_exec_res = raw.get("frozen_exec_res")
    if not isinstance(candidate_sql, str) or not isinstance(final_sql, str):
        raise ValueError("replay checkpoint SQL fields must be strings")
    if (
        isinstance(frozen_exec_res, bool)
        or type(frozen_exec_res) is not int
        or frozen_exec_res not in (0, 1)
    ):
        raise ValueError("replay checkpoint has invalid frozen_exec_res")
    return {
        "question_id": qid,
        "actions": list(actions),
        "candidate_sql": candidate_sql,
        "final_sql": final_sql,
        "frozen_exec_res": frozen_exec_res,
    }


def load_replay_checkpoint(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[int, tuple[int, ...]], dict[str, Any]], int, bool]:
    """Load completed replays, pruning deleted routes and a partial final line."""
    if not path.is_file():
        return {}, 0, False

    current_records = {_replay_key(record): record for record in records}
    content = path.read_bytes()
    raw_lines = content.splitlines()
    checkpoint: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}
    seen: set[tuple[int, tuple[int, ...]]] = set()
    dropped = 0
    trailing_partial = False
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line:
            raise ValueError(f"empty replay checkpoint line {line_number}")
        try:
            decoded = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if line_number == len(raw_lines) and not content.endswith(b"\n"):
                trailing_partial = True
                break
            raise ValueError(
                f"invalid replay checkpoint line {line_number}"
            ) from exc
        row = _validated_replay_checkpoint_row(decoded)
        key = _replay_key(row)
        if key in seen:
            raise ValueError(f"duplicate replay checkpoint qid-route: {key}")
        seen.add(key)
        current = current_records.get(key)
        if current is None:
            dropped += 1
            continue
        if int(current["exec_res"]) != row["frozen_exec_res"]:
            raise ValueError(f"replay checkpoint label drift for qid-route: {key}")
        checkpoint[key] = row

    if dropped or trailing_partial:
        ordered = [
            checkpoint[_replay_key(record)]
            for record in records
            if _replay_key(record) in checkpoint
        ]
        _atomic_write(
            path,
            b"".join(
                _canonical_bytes({"version": REPLAY_CHECKPOINT_VERSION, **row})
                for row in ordered
            ),
        )
    return checkpoint, dropped, trailing_partial


def cache_miss_report_row(
    error: RuntimeError,
    *,
    record_index: int,
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    match = CACHE_MISS_PATTERN.fullmatch(str(error))
    if match is None:
        return None
    qid = int(record["question_id"])
    if int(match.group(1)) != qid:
        raise ValueError("cache-miss qid does not match the frozen trajectory")
    return {
        "record_index": record_index,
        "question_id": qid,
        "actions": list(_actions(record["actions"])),
        "stage": match.group(2),
        "tier": int(match.group(3)),
        "error": str(error),
    }


def load_cache_miss_report(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, tuple[int, ...]], dict[str, Any]]:
    if not path.is_file():
        return {}
    current_keys = {_replay_key(record) for record in records}
    misses: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = json.loads(raw_line)
        if not isinstance(raw, Mapping) or raw.get("version") != CACHE_MISS_REPORT_VERSION:
            raise ValueError(f"invalid cache-miss report line {line_number}")
        key = _replay_key(raw)
        if key not in current_keys:
            continue
        if key in misses:
            raise ValueError(f"duplicate cache-miss report qid-route: {key}")
        misses[key] = {key: raw[key] for key in (
            "record_index", "question_id", "actions", "stage", "tier", "error"
        )}
    return misses


def write_cache_miss_report(
    path: Path,
    misses: Mapping[tuple[int, tuple[int, ...]], Mapping[str, Any]],
) -> None:
    ordered = sorted(misses.values(), key=lambda row: int(row["record_index"]))
    _atomic_write(
        path,
        b"".join(
            _canonical_bytes({"version": CACHE_MISS_REPORT_VERSION, **row})
            for row in ordered
        ),
    )


def load_frozen_records(
    path: str | Path,
    *,
    expected_trajectories: int | None = None,
    expected_qids: int | None = None,
    expected_routes: int | None = None,
) -> list[dict[str, Any]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse frozen New2 buffer: {path}") from exc
    if not isinstance(raw, list):
        raise ValueError("frozen New2 buffer must be a JSON array")
    if not raw:
        raise ValueError("frozen New2 buffer is empty")
    if expected_trajectories is not None and len(raw) != expected_trajectories:
        raise ValueError(
            f"New2 requires {expected_trajectories} trajectories, got {len(raw)}"
        )

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    qids: set[int] = set()
    routes: set[tuple[int, ...]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"New2 record {index} is not a mapping")
        qid = item.get("question_id")
        if isinstance(qid, bool) or type(qid) is not int:
            raise ValueError(f"New2 record {index} has invalid question_id")
        actions = _actions(item.get("actions"))
        if any(actions[decision] == 3 for decision in NON_SKIPPABLE_DECISIONS):
            raise ValueError(f"New2 record {index} violates the S4/S7 hard mask")
        exec_res = item.get("exec_res")
        if isinstance(exec_res, bool) or type(exec_res) is not int or exec_res not in (0, 1):
            raise ValueError(f"New2 record {index} has invalid exec_res")
        key = (qid, actions)
        if key in seen:
            raise ValueError(f"duplicate New2 qid-route: {key}")
        seen.add(key)
        qids.add(qid)
        routes.add(actions)
        records.append(dict(item))

    if expected_qids is not None and len(qids) != expected_qids:
        raise ValueError(f"New2 requires {expected_qids} qids, got {len(qids)}")
    if expected_routes is not None and len(routes) != expected_routes:
        raise ValueError(f"New2 requires {expected_routes} routes, got {len(routes)}")
    return records


def build_route_support_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_qid: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    stage_actions = [Counter() for _ in range(DECISION_COUNT)]
    adjacent = [Counter() for _ in range(DECISION_COUNT - 1)]
    prefixes = [Counter() for _ in range(1, DECISION_COUNT + 1)]
    for item in records:
        qid = int(item["question_id"])
        actions = _actions(item["actions"])
        by_qid[qid].add(actions)
        for stage, action in enumerate(actions):
            stage_actions[stage][str(action)] += 1
        for stage in range(DECISION_COUNT - 1):
            adjacent[stage][f"{actions[stage]}-{actions[stage + 1]}"] += 1
        for length in range(1, DECISION_COUNT + 1):
            prefixes[length - 1]["-".join(map(str, actions[:length]))] += 1

    route_count_distribution = Counter(len(value) for value in by_qid.values())
    return {
        "version": "new2_route_support_v1",
        "diagnostic_only": True,
        "trajectory_count": len(records),
        "transition_count": len(records) * DECISION_COUNT,
        "qid_count": len(by_qid),
        "global_route_count": len({_actions(row["actions"]) for row in records}),
        "routes_per_qid_distribution": {
            str(key): value for key, value in sorted(route_count_distribution.items())
        },
        "stage_action_counts": [dict(sorted(count.items())) for count in stage_actions],
        "adjacent_action_pair_counts": [
            dict(sorted(count.items())) for count in adjacent
        ],
        "prefix_counts": [dict(sorted(count.items())) for count in prefixes],
        "minimum_observed_stage_action_count": min(
            value for count in stage_actions for value in count.values()
        ),
    }


class ReadOnlyTransitionCache:
    """The lookup-only subset of TransitionCache used by exact replay."""

    allow_semantic_lookup = False

    def __init__(self, path: str | Path) -> None:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.connection = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro", uri=True
        )
        self.connection.row_factory = sqlite3.Row
        integrity = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError("New2 transition DB integrity_check is not ok")

    def lookup(self, key: Any, stored_hash: str | None = None) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT output_payload, stored_hash FROM transitions
            WHERE question_id=? AND stage_idx=? AND tier=?
              AND input_hash=? AND operator_version=?
            """,
            (
                key.question_id,
                key.stage_idx,
                key.tier,
                key.input_payload_hash,
                key.operator_version,
            ),
        ).fetchone()
        if row is None or (stored_hash is not None and row["stored_hash"] != stored_hash):
            return None
        return json.loads(row["output_payload"])

    def close(self) -> None:
        self.connection.close()


class AuditedTextEncoder:
    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.contract_sha256 = encoder.contract_sha256
        self.counts = Counter()

    def token_count(self, text: str) -> int:
        return int(self.encoder.token_count(text))

    def encode(self, role: str, text: str):
        vector, audit = self.encoder.encode(role, text)
        self.counts["encoded_text_count"] += 1
        self.counts["direct_text_count" if audit.chunk_count == 1 else "chunked_text_count"] += 1
        self.counts["silent_truncation_count"] += int(audit.silently_truncated)
        self.counts["max_token_count"] = max(
            self.counts["max_token_count"], audit.total_non_special_tokens
        )
        self.counts["max_chunk_count"] = max(
            self.counts["max_chunk_count"], audit.chunk_count
        )
        return vector, audit


def evaluator_contract_sha256() -> str:
    from env.database_utils import execution

    contract = {
        "version": "bird_execution_equivalence_v1",
        "implementation_sha256": sha256_file(Path(execution.__file__)),
        "timeout_seconds": EVALUATOR_TIMEOUT_SECONDS,
        "multiple_statements": "invalid_ex0",
        "unbound_parameters": "invalid_ex0",
    }
    return hashlib.sha256(_canonical_bytes(contract)).hexdigest()


def _evaluate_sql(db_path: Path, predicted_sql: str, gold_sql: str) -> tuple[int, str]:
    from env.database_utils.execution import compare_sqls

    if not predicted_sql.strip():
        return 0, "empty_sql"
    try:
        result = compare_sqls(
            db_path=str(db_path),
            predicted_sql=predicted_sql,
            ground_truth_sql=gold_sql,
            meta_time_out=EVALUATOR_TIMEOUT_SECONDS,
        )
    except (sqlite3.ProgrammingError, sqlite3.Warning) as exc:
        message = str(exc)
        invalid_sql = (
            message == "You can only execute one statement at a time."
            or message.startswith("Incorrect number of bindings supplied.")
        )
        if not invalid_sql:
            raise
        return 0, message
    exec_res = int(result.get("exec_res", 0))
    if exec_res not in (0, 1):
        raise ValueError("BIRD evaluator returned a non-binary result")
    return exec_res, str(result.get("exec_err") or "")


def _db_path(db_root: Path, db_mode: str, db_id: str) -> Path:
    path = db_root / f"{db_mode}_databases" / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _evaluate_candidate_sqls(
    replayed: Sequence[Mapping[str, Any]],
    question_map: Mapping[int, Mapping[str, Any]],
    *,
    db_root: Path,
    db_mode: str,
    evaluator_sha: str,
    seeded: Mapping[tuple[int, str], tuple[int, str]] | None = None,
) -> tuple[dict[tuple[int, str, str], tuple[int, str]], int, int]:
    results: dict[tuple[int, str, str], tuple[int, str]] = {}
    seeded_count = 0
    evaluated_count = 0
    for item in replayed:
        qid = int(item["question_id"])
        candidate_sql = str(item["candidate_sql"])
        sql_sha = hashlib.sha256(candidate_sql.encode("utf-8")).hexdigest()
        key = (qid, sql_sha, evaluator_sha)
        if key in results:
            continue
        seed_value = None if seeded is None else seeded.get((qid, sql_sha))
        if seed_value is not None:
            results[key] = seed_value
            seeded_count += 1
        else:
            question = question_map[qid]
            results[key] = _evaluate_sql(
                _db_path(db_root, db_mode, str(question["db_id"])),
                candidate_sql,
                str(question.get("SQL") or ""),
            )
            evaluated_count += 1
        completed = seeded_count + evaluated_count
        if completed % 1_000 == 0:
            print(
                f"candidate_ex={completed} seeded={seeded_count} "
                f"evaluated={evaluated_count}",
                flush=True,
            )
    return results, seeded_count, evaluated_count


def load_candidate_ex_seed(
    jsonl_path: Path,
    manifest_path: Path,
    *,
    evaluator_sha: str,
    question_sha: str,
    db_schema_sha: str,
    db_mode: str,
    db_ids: Sequence[str],
) -> dict[tuple[int, str], tuple[int, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "evaluator_contract_sha256": evaluator_sha,
        "question_data_sha256": question_sha,
        "db_schema_fingerprint": db_schema_sha,
        "db_content_sha256": db_schema_sha,
        "db_mode": db_mode,
        "db_ids": list(db_ids),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Candidate-EX seed {field} mismatch")
    content = jsonl_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest.get("candidate_ex_jsonl_sha256"):
        raise ValueError("Candidate-EX seed JSONL hash mismatch")

    seeded: dict[tuple[int, str], tuple[int, str]] = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        raw = json.loads(line)
        qid = raw.get("question_id")
        sql_sha = raw.get("candidate_sql_sha256")
        candidate_ex = raw.get("candidate_exec_res")
        status = raw.get("candidate_status")
        if isinstance(qid, bool) or type(qid) is not int:
            raise ValueError(f"Candidate-EX seed row {line_number} has invalid qid")
        if (
            type(sql_sha) is not str
            or len(sql_sha) != 64
            or any(char not in "0123456789abcdef" for char in sql_sha)
        ):
            raise ValueError(f"Candidate-EX seed row {line_number} has invalid SQL hash")
        if type(candidate_ex) is not int or candidate_ex not in (0, 1):
            raise ValueError(f"Candidate-EX seed row {line_number} has invalid EX")
        if type(status) is not str:
            raise ValueError(f"Candidate-EX seed row {line_number} has invalid status")
        if raw.get("candidate_evaluator_contract_sha256") != evaluator_sha:
            raise ValueError(
                f"Candidate-EX seed row {line_number} evaluator contract mismatch"
            )
        key = (qid, sql_sha)
        value = (candidate_ex, status)
        previous = seeded.setdefault(key, value)
        if previous != value:
            raise ValueError(f"Candidate-EX seed conflict for qid={qid} SQL={sql_sha}")
    return seeded


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", default="cache/new2.json")
    parser.add_argument("--transition_db", default="cache/new2.db")
    parser.add_argument("--data_path", default="data/train/selected/paper/selected_train.json")
    parser.add_argument("--db_root", default="data/train")
    parser.add_argument("--db_mode", default="train")
    parser.add_argument("--operator_pkg", default="env.chess")
    parser.add_argument(
        "--semantic_model",
        default=os.getenv("NL2SQL_SEMANTIC_MODEL", "models/gte-large-en-v1-5"),
    )
    parser.add_argument(
        "--output_root",
        default=f"{CONTROL_ROOT}/base",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume_semantic_work", action="store_true")
    parser.add_argument("--route_library", default="cache/new2_route_library.json")
    parser.add_argument(
        "--cache_miss_report",
        help="record exact cache-only replay misses and finish the full scan",
    )
    parser.add_argument("--candidate_ex_seed")
    parser.add_argument("--candidate_ex_seed_manifest")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.operator_pkg != "env.chess" or args.db_mode != "train":
        raise ValueError("New2 preparation requires env.chess train mode")

    from rl.dataset_utils import load_question_records, question_map_from_records
    from rl.env_wrapper import EnvironmentWrapper
    from rl.state_v25_encoder import StateV25Encoder
    from rl.state_v25_semantic import StateV25SemanticCache, StateV25TextEncoder

    buffer_path = Path(args.buffer)
    transition_db_path = Path(args.transition_db)
    output_root = Path(args.output_root)
    question_path = Path(args.data_path)
    db_root = Path(args.db_root)
    buffer_sha_before = sha256_file(buffer_path)
    question_sha_before = sha256_file(question_path)
    from rl.transition_cache_identity import sqlite_transition_identity

    def logical_sqlite_sha256(path: str | Path) -> str:
        identity = sqlite_transition_identity(path)
        if identity.get("integrity") != "ok":
            raise ValueError("transition DB integrity_check is not ok")
        value = identity.get("logical_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("transition DB logical identity is invalid")
        return value

    db_sha_before = logical_sqlite_sha256(transition_db_path)
    route_library = json.loads(Path(args.route_library).read_text(encoding="utf-8"))
    expected = {tuple(_actions(route["actions"])) for route in route_library}
    if not expected or len(expected) != len(route_library):
        raise ValueError("route library is empty or contains duplicate actions")
    records = load_frozen_records(buffer_path)
    actual = {(int(row["question_id"]), tuple(row["actions"])) for row in records}
    expected_qids = {int(row["question_id"]) for row in json.loads(
        Path(args.data_path).read_text(encoding="utf-8")
    )}
    if actual != {(qid, route) for qid in expected_qids for route in expected}:
        raise ValueError("buffer does not cover every selected request and pruned route exactly once")
    support = build_route_support_report(records)

    question_map = question_map_from_records(load_question_records(args.data_path))
    missing_qids = sorted({int(row["question_id"]) for row in records} - set(question_map))
    if missing_qids:
        raise ValueError(f"question metadata missing for qids: {missing_qids[:5]}")
    db_ids = sorted(
        {
            str(question_map[int(row["question_id"])]["db_id"])
            for row in records
        }
    )
    from rl.dataset_utils import db_schema_fingerprint

    db_schema_fingerprint_before = db_schema_fingerprint(
        db_root, args.db_mode, db_ids
    )

    output_root.mkdir(parents=True, exist_ok=True)
    semantic_final = output_root / "train_semantics.db"
    semantic_work = output_root / ".train_semantics.db.preparing"
    replay_checkpoint = output_root / ".replay_checkpoint.jsonl"
    cache_miss_report = Path(args.cache_miss_report) if args.cache_miss_report else None
    if args.resume_semantic_work:
        if not semantic_work.is_file():
            raise FileNotFoundError(semantic_work)
    else:
        semantic_work.unlink(missing_ok=True)
        replay_checkpoint.unlink(missing_ok=True)
    base_text_encoder = StateV25TextEncoder(
        args.semantic_model, device=args.device, max_length=8192
    )
    audited_encoder = AuditedTextEncoder(base_text_encoder)
    semantic_cache = StateV25SemanticCache(
        semantic_work, audited_encoder, writable=True
    )
    state_encoder = StateV25Encoder(
        db_root=args.db_root,
        db_mode=args.db_mode,
        text_encoder=audited_encoder,
        semantic_cache=semantic_cache,
    )
    cache = ReadOnlyTransitionCache(transition_db_path)
    env = EnvironmentWrapper(
        operator_pkg=args.operator_pkg,
        db_manager_factory=None,
        exact_replay_only=True,
    )
    env.set_question_map(question_map)
    if env.non_skippable_decision_stage_idxs != NON_SKIPPABLE_DECISIONS:
        raise ValueError("environment S4/S7 action-mask contract drifted")

    checkpoint_rows, dropped_checkpoint_rows, trailing_partial = (
        load_replay_checkpoint(replay_checkpoint, records)
    )
    if checkpoint_rows or dropped_checkpoint_rows or trailing_partial:
        print(
            "replay_checkpoint_loaded="
            f"{len(checkpoint_rows)} dropped={dropped_checkpoint_rows} "
            f"trailing_partial={int(trailing_partial)}",
            flush=True,
        )

    replayed: list[dict[str, Any]] = []
    resumed_count = 0
    new_checkpoint_rows = 0
    cache_misses = (
        load_cache_miss_report(cache_miss_report, records)
        if cache_miss_report is not None
        else {}
    )
    if cache_misses:
        print(f"cache_miss_report_loaded={len(cache_misses)}", flush=True)
    checkpoint_stream = replay_checkpoint.open("ab", buffering=0)
    try:
        for index, record in enumerate(records, start=1):
            key = _replay_key(record)
            checkpointed = checkpoint_rows.get(key)
            if checkpointed is not None:
                replayed.append(checkpointed)
                resumed_count += 1
            elif key in cache_misses:
                pass
            else:
                try:
                    trajectory = env.rollout_with_cache(
                        question_id=int(record["question_id"]),
                        action_sequence=list(_actions(record["actions"])),
                        cache=cache,
                        state_encoder=state_encoder,
                        evaluate_terminal=False,
                        cache_only=True,
                        capture_candidate_sql=True,
                    )
                except RuntimeError as exc:
                    miss = cache_miss_report_row(
                        exc, record_index=index - 1, record=record
                    )
                    if cache_miss_report is None or miss is None:
                        raise
                    cache_misses[key] = miss
                    write_cache_miss_report(cache_miss_report, cache_misses)
                    print(
                        "cache_miss_recorded="
                        f"{len(cache_misses)} qid={miss['question_id']} "
                        f"actions={miss['actions']} stage={miss['stage']} "
                        f"tier={miss['tier']}",
                        flush=True,
                    )
                    continue
                if trajectory is None or len(trajectory.get("states", [])) != 7:
                    raise ValueError(f"invalid exact replay for record {index - 1}")
                replay_row = {
                    "question_id": int(record["question_id"]),
                    "actions": list(_actions(record["actions"])),
                    "candidate_sql": str(trajectory.get("candidate_sql") or ""),
                    "final_sql": str(trajectory.get("final_sql") or ""),
                    "frozen_exec_res": int(record["exec_res"]),
                }
                replayed.append(replay_row)
                checkpoint_stream.write(
                    _canonical_bytes(
                        {"version": REPLAY_CHECKPOINT_VERSION, **replay_row}
                    )
                )
                checkpoint_rows[key] = replay_row
                new_checkpoint_rows += 1
                if new_checkpoint_rows % REPLAY_CHECKPOINT_SYNC_INTERVAL == 0:
                    os.fsync(checkpoint_stream.fileno())
            if index % 1000 == 0:
                print(
                    f"replayed={index}/{len(records)} resumed={resumed_count} "
                    f"cache_misses={len(cache_misses)}",
                    flush=True,
                )
    finally:
        os.fsync(checkpoint_stream.fileno())
        checkpoint_stream.close()
        cache.close()
        semantic_cache.close()

    if cache_misses:
        raise RuntimeError(
            f"full replay scan found {len(cache_misses)} cache-only misses; "
            f"repair entries listed in {cache_miss_report}"
        )

    if bool(args.candidate_ex_seed) != bool(args.candidate_ex_seed_manifest):
        raise ValueError(
            "candidate_ex_seed and candidate_ex_seed_manifest must be provided together"
        )
    evaluator_sha = evaluator_contract_sha256()
    seeded_results = None
    seed_manifest_sha = None
    if args.candidate_ex_seed:
        seed_path = Path(args.candidate_ex_seed)
        seed_manifest_path = Path(args.candidate_ex_seed_manifest)
        seeded_results = load_candidate_ex_seed(
            seed_path,
            seed_manifest_path,
            evaluator_sha=evaluator_sha,
            question_sha=question_sha_before,
            db_schema_sha=db_schema_fingerprint_before,
            db_mode=args.db_mode,
            db_ids=db_ids,
        )
        seed_manifest_sha = sha256_file(seed_manifest_path)
        print(f"candidate_ex_seed_loaded={len(seeded_results)}", flush=True)
    candidate_results, seeded_count, evaluated_count = _evaluate_candidate_sqls(
        replayed,
        question_map,
        db_root=db_root,
        db_mode=args.db_mode,
        evaluator_sha=evaluator_sha,
        seeded=seeded_results,
    )

    rows = []
    for item in replayed:
        qid = item["question_id"]
        candidate_sha = hashlib.sha256(item["candidate_sql"].encode("utf-8")).hexdigest()
        final_sha = hashlib.sha256(item["final_sql"].encode("utf-8")).hexdigest()
        candidate_ex, candidate_status = candidate_results[(qid, candidate_sha, evaluator_sha)]
        final_ex = item["frozen_exec_res"]
        rows.append(
            {
                "question_id": qid,
                "actions": item["actions"],
                "candidate_exec_res": candidate_ex,
                "final_exec_res": final_ex,
                "candidate_sql_sha256": candidate_sha,
                "final_sql_sha256": final_sha,
                "candidate_evaluator_contract_sha256": evaluator_sha,
                "final_label_source": "frozen_buffer_exec_res",
                "candidate_status": candidate_status,
                "final_status": "frozen_buffer_exec_res",
            }
        )

    rows.sort(key=lambda row: (row["question_id"], row["actions"]))
    jsonl = b"".join(_canonical_bytes(row) for row in rows)
    candidate_path = output_root / "candidate_ex.jsonl"
    candidate_manifest = {
        "version": "candidate_ex_delta_v1",
        "row_count": len(rows),
        "buffer_sha256": buffer_sha_before,
        "transition_db_logical_sha256": db_sha_before,
        "question_data_sha256": question_sha_before,
        "db_schema_fingerprint": db_schema_fingerprint_before,
        "db_content_sha256": db_schema_fingerprint_before,
        "db_mode": args.db_mode,
        "db_ids": db_ids,
        "evaluator_contract_sha256": evaluator_sha,
        "final_label_source": "frozen_buffer_exec_res",
        "candidate_ex_jsonl_sha256": hashlib.sha256(jsonl).hexdigest(),
        "candidate_unique_sql_count": len(candidate_results),
        "candidate_seeded_unique_sql_count": seeded_count,
        "candidate_evaluated_unique_sql_count": evaluated_count,
        "final_label_count": len(rows),
        "final_label_mismatch_count": 0,
    }
    if seed_manifest_sha is not None:
        candidate_manifest["candidate_ex_seed_manifest_sha256"] = seed_manifest_sha
    semantic_reader = StateV25SemanticCache(
        semantic_work,
        None,
        writable=False,
        encoder_contract_sha256=audited_encoder.contract_sha256,
    )
    semantic_audit = semantic_reader.audit()
    semantic_reader.close()
    semantic_manifest = {
        "version": "state_v25_semantics_v1",
        "encoder_contract_sha256": audited_encoder.contract_sha256,
        "row_count": len(rows),
        **semantic_audit,
        **dict(audited_encoder.counts),
    }
    if semantic_manifest.get("silent_truncation_count", 0) != 0:
        raise ValueError("State V25 semantic preparation silently truncated text")
    if sha256_file(buffer_path) != buffer_sha_before:
        raise ValueError("frozen New2 buffer changed during preparation")
    if logical_sqlite_sha256(transition_db_path) != db_sha_before:
        raise ValueError("frozen New2 transition DB changed during preparation")
    if sha256_file(question_path) != question_sha_before:
        raise ValueError("question data changed during preparation")
    if (
        db_schema_fingerprint(db_root, args.db_mode, db_ids)
        != db_schema_fingerprint_before
    ):
        raise ValueError("database content changed during preparation")

    os.replace(semantic_work, semantic_final)
    semantic_manifest["semantic_db_sha256"] = sha256_file(semantic_final)
    _atomic_write(candidate_path, jsonl)
    _atomic_write(
        output_root / "candidate_ex.manifest.json",
        _canonical_bytes(candidate_manifest),
    )
    _atomic_write(
        output_root / "train_semantics.manifest.json",
        _canonical_bytes(semantic_manifest),
    )
    _atomic_write(output_root / "route_support.json", _canonical_bytes(support))
    replay_checkpoint.unlink(missing_ok=True)
    print(_canonical_bytes({**candidate_manifest, **semantic_manifest}).decode(), end="")


if __name__ == "__main__":
    main()
