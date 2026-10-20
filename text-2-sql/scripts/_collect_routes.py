#!/usr/bin/env python3
"""Execute scheduled Text-to-SQL routes with exact transition reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PROJECT_ROOT
_ENV_ROOT = _PROJECT_ROOT / "env"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_ENV_ROOT))

from _shared.model_config import DEFAULT_LLM_MODEL
from config import (
    DEFAULT_MODEL_ROOT as CONFIG_DEFAULT_MODEL_ROOT,
    DEFAULT_RUN_NAME as CONFIG_DEFAULT_RUN_NAME,
    get_run_config,
)
from rl.reward import (
    DEFAULT_COST_CLIP,
    DEFAULT_COST_WEIGHT,
    DEFAULT_LEGACY_ALPHA,
    DEFAULT_REWARD_BETA,
    DEFAULT_REWARD_VERSION,
    DEFAULT_SUCCESS_BONUS,
    EX_FIRST_COST_CLIPPED_V1,
    EX_FIRST_GLOBAL_STAGE_COST_V1,
    RewardEvaluator,
    reward_config_from_args,
)

load_dotenv(_PROJECT_ROOT.parent / ".env", override=True)


class _SuppressNoisyTelemetryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Failed to send telemetry event" not in record.getMessage()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_SuppressNoisyTelemetryFilter())
logger = logging.getLogger("collect")
for _noisy_logger_name in (
    "httpx",
    "httpcore",
    "openai",
    "dashscope",
):
    logging.getLogger(_noisy_logger_name).setLevel(logging.WARNING)

DEFAULT_DATA_PATH = str(_PROJECT_ROOT / "data" / "train" / "train.json")
DEFAULT_MAX_QUERIES = int(os.getenv("NL2SQL_MAX_QUERIES", "9428")) # 最大query  训练集9428  dev 1534
DEFAULT_OUTPUT_PATH = os.getenv(
    "NL2SQL_BUFFER_PATH",
    str(_PROJECT_ROOT / "cache" / "adaptive_trajectories.json"),
)
DEFAULT_TRANSITION_DB = os.getenv(
    "NL2SQL_TRANSITION_DB",
    str(_PROJECT_ROOT / "cache" / "transitions.db"),
)
DEFAULT_DB_MODE = "train"
DEFAULT_DB_ROOT = str(_PROJECT_ROOT / "data" / "train")
DEFAULT_RUN_NAME = os.getenv("NL2SQL_RUN_NAME", CONFIG_DEFAULT_RUN_NAME)
DEFAULT_MODEL_ROOT = os.getenv(
    "NL2SQL_MODEL_ROOT",
    str(_PROJECT_ROOT / CONFIG_DEFAULT_MODEL_ROOT),
)


def configure_database_environment(db_root: str, db_mode: str) -> None:
    """Keep legacy and DatabaseManager database settings consistent."""
    os.environ["DB_ROOT_PATH"] = db_root
    os.environ["DATA_MODE"] = db_mode
    os.environ["NL2SQL_DB_ROOT"] = db_root
    os.environ["NL2SQL_DB_MODE"] = db_mode


def expected_database_path(db_root: str | os.PathLike[str], db_mode: str, db_id: str) -> Path:
    """Return the sqlite path expected by env.runner.database_manager.DatabaseManager."""
    return Path(db_root) / f"{db_mode}_databases" / db_id / f"{db_id}.sqlite"


def validate_database_paths(question_map, db_mode: str, *, max_missing: int = 5) -> None:
    """Fail fast when DB_ROOT_PATH cannot satisfy the questions selected for collection."""
    db_root = os.getenv("DB_ROOT_PATH")
    if not db_root:
        raise RuntimeError(
            "DB_ROOT_PATH is not set. For BIRD train data it should usually point to "
            ".../text-to-SQL/data/train, so sqlite files resolve as "
            "DB_ROOT_PATH/train_databases/<db_id>/<db_id>.sqlite."
        )

    missing = []
    checked = 0
    seen_db_ids = set()
    for task in question_map.values():
        db_id = task.get("db_id")
        if not db_id or db_id in seen_db_ids:
            continue
        seen_db_ids.add(db_id)
        checked += 1
        db_path = expected_database_path(db_root, db_mode, str(db_id))
        if not db_path.exists():
            missing.append((str(db_id), str(db_path)))
            if len(missing) >= max_missing:
                break

    if missing:
        examples = "\n".join(f"  - db_id={db_id}: {path}" for db_id, path in missing)
        raise FileNotFoundError(
            "Cannot find sqlite database files before collection starts.\n"
            f"DB_ROOT_PATH={db_root}\n"
            f"db_mode={db_mode}\n"
            "Expected layout: DB_ROOT_PATH/<db_mode>_databases/<db_id>/<db_id>.sqlite\n"
            f"Missing examples:\n{examples}\n"
            "If your train DBs live under text-to-SQL/data/train/train_databases, set "
            "DB_ROOT_PATH to text-to-SQL/data/train, not to the train_databases folder itself."
        )

    logger.info("Validated %d database path(s) under DB_ROOT_PATH=%s", checked, db_root)


def save_buffer_if_nonempty(traj_buffer, output_path: str, *, reason: str) -> bool:
    if len(traj_buffer) == 0:
        logger.warning(
            "Skip saving empty TrajectoryBuffer (%s); existing buffer is left untouched: %s",
            reason,
            output_path,
        )
        return False
    traj_buffer.save(output_path)
    return True


def load_existing_buffer_if_requested(traj_buffer, output_path: str, *, append_buffer: bool) -> int:
    if not append_buffer or not os.path.exists(output_path):
        return 0
    traj_buffer.load(output_path)
    return len(traj_buffer)


def write_collection_status(
    path: str | os.PathLike[str],
    *,
    outcome: str,
    elapsed_seconds: float,
    tracker,
    error: BaseException | None = None,
    batch_sha256: str,
    concurrency: int,
    invocation_id: str,
) -> None:
    """Atomically publish credential-free subprocess status for the outer writer."""
    from rl.api_guard import (
        ApiInfrastructureError,
        ApiQuotaExhaustedError,
        ApiRateLimitError,
    )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    usage = tracker.audit_summary()
    error_kind = None
    if isinstance(error, ApiRateLimitError):
        error_kind = "rate_limit"
    elif isinstance(error, ApiQuotaExhaustedError):
        error_kind = "quota"
    elif isinstance(error, ApiInfrastructureError):
        error_kind = "infrastructure"
    elif error is not None:
        error_kind = "failure"
    payload = {
        "version": "new2_collection_status_v1",
        "outcome": outcome,
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 6),
        "error": None if error is None else {
            "kind": str(error_kind), "type": type(error).__name__,
        },
        "batch_sha256": str(batch_sha256),
        "concurrency": int(concurrency),
        "invocation_id": str(invocation_id),
        "usage": usage,
    }
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_route_names_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_route_templates(path: str):
    from rl.route_ladder import RouteTemplate
    from rl.stage_mapping import NUM_RL_STAGES, legal_decision_actions

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("route library must be a nonempty list")
    templates = []
    names = set()
    routes = set()
    for rank, entry in enumerate(loaded):
        name = str(entry["name"])
        actions = [int(action) for action in entry["actions"]]
        if len(actions) != NUM_RL_STAGES or any(
            action not in legal_decision_actions(stage)
            for stage, action in enumerate(actions)
        ):
            raise ValueError(f"illegal route actions: {name}")
        key = tuple(actions)
        if not name or name in names or key in routes:
            raise ValueError("route library has duplicate or empty entries")
        names.add(name)
        routes.add(key)
        templates.append(RouteTemplate(name=name, actions=actions, rank=rank))
    return tuple(templates)


def _validate_exact_flags(args) -> bool:
    enabled = (
        bool(getattr(args, "exact_only", False)),
        bool(getattr(args, "disable_semantic_cache", False)),
        bool(getattr(args, "cache_identity_manifest", None)),
        bool(getattr(args, "data_sha256", None)),
    )
    if any(enabled) and not all(enabled):
        raise ValueError(
            "--exact_only, --disable_semantic_cache, and "
            "--cache_identity_manifest, and --data_sha256 must be enabled together"
        )
    return all(enabled)


def load_authenticated_question_records(
    data_path: str | os.PathLike[str],
    expected_sha256: str,
    max_queries: int | None = None,
    start_query: int = 0,
) -> list[dict[str, Any]]:
    """Load and normalize the exact bytes authenticated by the outer runner."""
    from rl.dataset_utils import _as_record_list, normalize_question_records

    if start_query < 0:
        raise ValueError("start_query must be >= 0")
    raw = Path(data_path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("authenticated question data digest mismatch")
    try:
        records = _as_record_list(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("authenticated question data is invalid JSON") from exc
    selected = records[start_query:]
    if max_queries is not None:
        selected = selected[:max_queries]
    return normalize_question_records(selected, fallback_start=start_query)


def exact_runtime_options(args, cache_identity):
    """Return constructor kwargs without changing legacy construction semantics."""
    if not _validate_exact_flags(args):
        return {}, {}
    if not isinstance(cache_identity, Mapping) or not cache_identity:
        raise ValueError("exact mode requires a complete cache identity mapping")
    return (
        {"allow_semantic_lookup": False},
        {"exact_only": True, "cache_identity": cache_identity},
    )


def load_exact_cache_identity_manifest(
    path: str | os.PathLike[str],
    question_map: Mapping[int, Mapping[str, Any]],
    *,
    operator_pkg: str,
    project_root: str | os.PathLike[str] | None = None,
    db_root: str | os.PathLike[str] | None = None,
    db_mode: str = "train",
    embedding_endpoint: str | None = None,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Expand a credential-free compact new2 contract into Task 2 envelopes."""
    from rl.exact_cache_contract import (
        CONTRACT_KEYS,
        canonical_sha,
        expand_exact_cache_contract,
    )
    from pipeline.new2_manifest_schema import (
        FROZEN_EXACT_CACHE_KEY,
        FROZEN_EXACT_CACHE_VERSION,
    )

    manifest_path = Path(path)
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load exact cache identity manifest: {manifest_path}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("exact cache identity manifest must contain a mapping")
    if set(loaded) == CONTRACT_KEYS:
        contract = loaded
    else:
        frozen = loaded.get(FROZEN_EXACT_CACHE_KEY)
        build_identity = loaded.get("build_identity")
        if (
            not isinstance(frozen, Mapping)
            or set(frozen) != {"version", "contract", "contract_sha256"}
            or frozen.get("version") != FROZEN_EXACT_CACHE_VERSION
            or not isinstance(frozen.get("contract"), Mapping)
            or not isinstance(build_identity, Mapping)
        ):
            raise ValueError("exact cache identity manifest strict schema mismatch")
        contract = frozen["contract"]
        if loaded.get("build_identity_sha256") != canonical_sha(build_identity):
            raise ValueError("exact cache identity manifest build identity mismatch")
        if frozen.get("contract_sha256") != canonical_sha(contract):
            raise ValueError("exact cache identity frozen contract digest mismatch")
        if build_identity.get("exact_cache_contract_sha256") != canonical_sha(contract):
            raise ValueError("exact cache identity manifest digest mismatch")
    return expand_exact_cache_contract(
        contract,
        question_map,
        operator_pkg=operator_pkg,
        project_root=project_root,
        db_root=db_root,
        db_mode=db_mode,
        embedding_endpoint=embedding_endpoint,
    )


def _preparse_run_name(argv=None) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    args, _ = parser.parse_known_args(argv)
    return str(args.run_name)


def parse_args(argv=None):
    run_name = _preparse_run_name(argv)
    run_config = get_run_config(run_name)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", default=run_name)
    parser.add_argument("--model_root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--start_query",
        type=int,
        default=int(os.getenv("NL2SQL_START_QUERY", "0")),
        help=(
            "0-based global dataset row/item offset; for example, "
            "--start_query 200 --max_queries 100 selects dataset rows/items 200-299."
        ),
    )
    parser.add_argument("--max_queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--append_buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load existing trajectory JSON before collection; matching qid+route records are replaced.",
    )
    parser.add_argument(
        "--autosave_every",
        type=int,
        default=int(os.getenv("NL2SQL_AUTOSAVE_EVERY", "1")),
        help="Save trajectory JSON every N completed TrajectoryRecord(s); 0 disables autosave.",
    )

    parser.add_argument("--transition_db", type=str, default=DEFAULT_TRANSITION_DB)
    parser.add_argument("--buffer_capacity", type=int, default=50000)
    parser.add_argument(
        "--collection_policy",
        choices=["adaptive"],
        default="adaptive",
        help="Collect scheduled route-ladder trajectories.",
    )
    parser.add_argument("--n_workers", type=int, default=int(os.getenv("NL2SQL_N_WORKERS", "4")))
    parser.add_argument(
        "--stop_cache_hit_rate",
        "--target_hit_rate",
        dest="stop_cache_hit_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--trajectories_per_query",
        type=int,
        default=int(os.getenv("NL2SQL_TRAJECTORIES_PER_QUERY", "4")),
    )
    parser.add_argument(
        "--min_routes_per_query",
        type=int,
        default=int(os.getenv("NL2SQL_MIN_ROUTES_PER_QUERY", "1")),
    )
    parser.add_argument(
        "--max_routes_per_query",
        type=int,
        default=int(os.getenv("NL2SQL_MAX_ROUTES_PER_QUERY", "5")),
    )
    parser.add_argument("--route_library", required=True, help="JSON route library generated after anchor profiling")
    parser.add_argument(
        "--route_names",
        type=str,
        default=os.getenv("NL2SQL_ROUTE_NAMES", ""),
        help=(
            "Comma-separated route-library names to collect. "
            "When omitted, collection uses the route-library prefix."
        ),
    )
    parser.add_argument("--stop_on_first_success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit_route_rate", type=float, default=float(os.getenv("NL2SQL_AUDIT_ROUTE_RATE", "0.0")))

    parser.add_argument("--operator_pkg", type=str, default="env.chess")
    parser.add_argument("--exact_only", action="store_true")
    parser.add_argument("--disable_semantic_cache", action="store_true")
    parser.add_argument("--cache_identity_manifest")
    parser.add_argument("--data_sha256")
    parser.add_argument("--collection_status_path")
    parser.add_argument("--sdk_max_retries", type=int)
    parser.add_argument("--collection_batch_sha256")
    parser.add_argument("--collection_concurrency", type=int)
    parser.add_argument("--collection_invocation_id")
    parser.add_argument("--failure_queue_path")
    parser.add_argument(
        "--retry_deferred_pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry infrastructure failures serially at the end of this invocation.",
    )
    parser.add_argument("--model_name", "--model", dest="model_name", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--db_root", type=str, default=DEFAULT_DB_ROOT)
    parser.add_argument("--db_mode", type=str, default=DEFAULT_DB_MODE)
    parser.add_argument("--skip_db_validation", action="store_true")
    parser.add_argument(
        "--reward_version",
        choices=[EX_FIRST_COST_CLIPPED_V1, EX_FIRST_GLOBAL_STAGE_COST_V1],
        default=EX_FIRST_COST_CLIPPED_V1,
    )
    parser.add_argument("--reward_beta", type=float, default=run_config.get("reward_beta", DEFAULT_REWARD_BETA))
    parser.add_argument("--reward_alpha", type=float, default=DEFAULT_LEGACY_ALPHA)
    parser.add_argument("--reward_success_bonus", type=float, default=run_config.get("reward_success_bonus", DEFAULT_SUCCESS_BONUS))
    parser.add_argument("--reward_cost_weight", type=float, default=run_config.get("reward_cost_weight", DEFAULT_COST_WEIGHT))
    parser.add_argument("--reward_cost_clip", type=float, default=run_config.get("reward_cost_clip", DEFAULT_COST_CLIP))
    parser.add_argument("--stage_cost_normalizer_path", type=Path)
    args = parser.parse_args(argv)
    args.stage_cost_reward = args.reward_version == EX_FIRST_GLOBAL_STAGE_COST_V1
    args.stage_cost_normalizer_file = run_config.get("stage_cost_normalizer_file")
    _validate_exact_flags(args)
    audit_flags = (
        bool(args.collection_status_path), args.sdk_max_retries is not None,
        bool(args.collection_batch_sha256), args.collection_concurrency is not None,
        bool(args.collection_invocation_id),
    )
    if any(audit_flags) and not all(audit_flags):
        raise ValueError(
            "structured collection status flags must be enabled together"
        )
    if args.sdk_max_retries is not None and args.sdk_max_retries not in {0, 1}:
        raise ValueError("audited --sdk_max_retries must be 0 or 1")
    if args.collection_concurrency is not None and args.collection_concurrency <= 0:
        raise ValueError("collection concurrency must be positive")
    return args


def load_collection_stage_cost_normalizer(args):
    if not args.stage_cost_reward:
        return None
    from rl.stage_cost_normalization import load_global_stage_cost_normalizer

    path = args.stage_cost_normalizer_path or (
        Path(args.model_root) / args.run_name / str(args.stage_cost_normalizer_file)
    )
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Stage cost normalizer not found: {path}")
    return load_global_stage_cost_normalizer(path)


def main():
    args = parse_args()
    configure_database_environment(args.db_root, args.db_mode)
    audit_tracker = None
    if args.collection_status_path:
        os.environ["NL2SQL_SDK_MAX_RETRIES"] = str(args.sdk_max_retries)
        from _shared.llm_adapter import set_audit_cost_tracker
        from _shared.state_payload import CostTracker

        audit_tracker = CostTracker()
        set_audit_cost_tracker(audit_tracker)

    from rl.dataset_utils import load_question_records, question_map_from_records
    from rl.env_wrapper import EnvironmentWrapper
    from rl.experience_collector import ExperienceCollector
    from rl.api_guard import (
        ApiInfrastructureError,
        ApiQuotaExhaustedError,
        ApiRateLimitError,
    )
    from rl.trajectory_buffer import TrajectoryBuffer
    from rl.transition_cache import TransitionCache as _BaseTransitionCache

    class TransitionCache(_BaseTransitionCache):
        """One bounded retry for transient SQLite writer contention."""

        def store(self, *store_args, **store_kwargs):
            try:
                return super().store(*store_args, **store_kwargs)
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                try:
                    self._get_conn().rollback()
                except sqlite3.Error:
                    pass
                delay = 1.0 + random.random()
                logger.warning(
                    "TransitionCache.store %s; rollback then retry once after %.3fs",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
                return super().store(*store_args, **store_kwargs)

    exact_mode = _validate_exact_flags(args)
    if exact_mode:
        data = load_authenticated_question_records(
            args.data_path, args.data_sha256,
            max_queries=args.max_queries, start_query=args.start_query,
        )
    else:
        data = load_question_records(
            args.data_path,
            max_queries=args.max_queries,
            start_query=args.start_query,
        )
    question_ids = [int(t["question_id"]) for t in data]
    question_map = question_map_from_records(data)
    cache_identity = None
    if exact_mode:
        embedding_endpoint = os.getenv("LLM_BASE_URL")
        if not embedding_endpoint:
            raise ValueError("exact mode requires LLM_BASE_URL for embedding endpoint identity")
        cache_identity = load_exact_cache_identity_manifest(
            args.cache_identity_manifest,
            question_map,
            operator_pkg=args.operator_pkg,
            project_root=_PROJECT_ROOT,
            db_root=args.db_root,
            db_mode=args.db_mode,
            embedding_endpoint=embedding_endpoint,
        )
    if not args.skip_db_validation:
        validate_database_paths(question_map, args.db_mode)

    normalizer = load_collection_stage_cost_normalizer(args)
    cache_kwargs, env_kwargs = exact_runtime_options(args, cache_identity)
    cache = TransitionCache(args.transition_db, **cache_kwargs)
    traj_buffer = TrajectoryBuffer(capacity=args.buffer_capacity)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    loaded_buffer_records = load_existing_buffer_if_requested(
        traj_buffer,
        args.output_path,
        append_buffer=args.append_buffer,
    )
    if loaded_buffer_records:
        logger.info(
            "Append buffer mode: loaded %d existing trajectory record(s) from %s",
            loaded_buffer_records,
            args.output_path,
        )
    logger.info("TransitionCache append/reuse mode: %s (%d entries)", args.transition_db, cache.count())

    def db_manager_factory(db_id):
        from runner.database_manager import DatabaseManager

        return DatabaseManager(db_mode=args.db_mode, db_id=db_id)

    env = EnvironmentWrapper(
        operator_pkg=args.operator_pkg,
        model=args.model_name,
        db_manager_factory=db_manager_factory,
        **env_kwargs,
    )
    env.set_question_map(question_map)

    reward_config = reward_config_from_args(args)
    reward_evaluator = RewardEvaluator(reward_config, normalizer=normalizer)

    collector = ExperienceCollector(
        env=env,
        cache=cache,
        traj_buffer=traj_buffer,
        reward_config=reward_config,
        reward_evaluator=reward_evaluator,
        collection_policy=args.collection_policy,
        min_routes_per_query=args.min_routes_per_query,
        max_routes_per_query=args.max_routes_per_query,
        stop_on_first_success=args.stop_on_first_success,
        audit_route_rate=args.audit_route_rate,
        route_templates=load_route_templates(args.route_library),
        route_names=parse_route_names_arg(args.route_names),
        n_workers=args.n_workers,
        autosave_path=args.output_path,
        autosave_every=args.autosave_every,
        failure_queue_path=args.failure_queue_path,
        retry_deferred_pairs=args.retry_deferred_pairs,
    )
    collection_started = time.monotonic()
    collection_error = None
    collection_outcome = "failure"
    try:
        metrics = collector.collect_from_questions(
            question_ids,
            max_queries=args.max_queries,
            stop_cache_hit_rate=args.stop_cache_hit_rate,
            trajectories_per_query=args.trajectories_per_query,
            min_routes_per_query=args.min_routes_per_query,
            max_routes_per_query=args.max_routes_per_query,
            stop_on_first_success=args.stop_on_first_success,
            audit_route_rate=args.audit_route_rate,
        )
        collection_outcome = "success"
    except ApiQuotaExhaustedError as exc:
        collection_error = exc
        collection_outcome = "quota_exhausted"
        save_buffer_if_nonempty(traj_buffer, args.output_path, reason="api_quota_exhausted")
        cache.close()
        logger.error(
            "API的钱用完了，已立即停止收集。当前buffer_size=%d，已保存到 %s。error=%s",
            len(traj_buffer),
            args.output_path,
            exc,
        )
        raise SystemExit(2) from exc
    except ApiRateLimitError as exc:
        collection_error = exc
        collection_outcome = "rate_limited"
        save_buffer_if_nonempty(traj_buffer, args.output_path, reason="api_rate_limited")
        cache.close()
        raise SystemExit(3) from exc
    except ApiInfrastructureError as exc:
        collection_error = exc
        collection_outcome = "infrastructure_error"
        save_buffer_if_nonempty(traj_buffer, args.output_path, reason="api_infrastructure_error")
        cache.close()
        raise SystemExit(4) from exc
    except BaseException as exc:
        collection_error = exc
        collection_outcome = "failure"
        save_buffer_if_nonempty(traj_buffer, args.output_path, reason="collection_failure")
        cache.close()
        raise
    finally:
        if args.collection_status_path and audit_tracker is not None:
            write_collection_status(
                args.collection_status_path,
                outcome=collection_outcome,
                elapsed_seconds=time.monotonic() - collection_started,
                tracker=audit_tracker,
                error=collection_error,
                batch_sha256=args.collection_batch_sha256,
                concurrency=args.collection_concurrency,
                invocation_id=args.collection_invocation_id,
            )

    save_buffer_if_nonempty(traj_buffer, args.output_path, reason="collection_finished")
    cache.close()
    logger.info(
        "Collected %d trajectories, decision_steps=%d, physical_stages=%d, cache_hit=%.1f%%, avg_reward=%.4f -> %s",
        metrics["n_collected"],
        NUM_RL_STAGES,
        NUM_PIPELINE_STAGES,
        metrics["cache_hit_rate"] * 100,
        metrics["avg_reward"],
        args.output_path,
    )


if __name__ == "__main__":
    main()
