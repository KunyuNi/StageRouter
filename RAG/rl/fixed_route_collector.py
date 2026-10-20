"""Resumable materialisation of fixed MultiHop-RAG route profiles.

The collector is intentionally small: the environment wrapper owns physical
stage execution and the two lower-level caches own transition/route identity.
This module only coordinates train records, appends route results, and keeps
the metadata-only ``new1`` trajectory buffer up to date.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Mapping

from .error_policy import (
    is_content_safety_error,
    is_fatal_collection_error,
    pair_error_category,
    pair_error_fingerprint,
)
from .failure_queue import FailedPairQueue
from .route_result_cache import RouteResultCache
from .stage_mapping import NUM_DECISION_STAGES, NUM_PHYSICAL_STAGES, route_key
from .trajectory_buffer import TrajectoryBuffer, TrajectoryRecord
from .transition_cache import TransitionCache, TransitionCacheIntegrityError


FIXED_ROUTES: dict[str, tuple[int, ...]] = {
    "all_t1": (0, 0, 0, 0),
    "all_t2": (1, 1, 1, 1),
    "all_t3": (2, 2, 2, 2),
}
BASELINE_ORDER: tuple[str, ...] = ("all_t3", "all_t2", "all_t1")
# Descriptive alias for callers ported from NL2SQL's evaluator module.
FIXED_BASELINE_ROUTES = FIXED_ROUTES
F1_METRIC_VERSION = "qa_f1_token_set_v1"
REWARD_VERSION = "raw_f1_passthrough_v1"


def _atomic_json_dump(value: Any, path: str | os.PathLike[str]) -> None:
    """Write JSON via the same replace-based durability boundary as the buffer."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _identity_string(identity: Any) -> str:
    if identity is None:
        return ""
    if isinstance(identity, Mapping):
        explicit = identity.get("workflow_identity")
        if explicit is not None:
            return str(explicit)
    explicit = getattr(identity, "workflow_identity", None)
    if explicit is not None:
        return str(explicit)
    digest = getattr(identity, "sha256", None)
    if digest is not None:
        return str(digest)
    return str(identity)


def _identity_dict(identity: Any) -> dict[str, Any]:
    value = getattr(identity, "to_json_dict", None)
    if callable(value):
        result = value()
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    if isinstance(identity, Mapping):
        return copy.deepcopy(dict(identity))
    return {}


def _qid(item: Mapping[str, Any]) -> int:
    value = item.get("question_id", item.get("qid"))
    if value is None:
        raise ValueError("dataset item requires question_id")
    return int(value)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalise_vector(value: Any, length: int) -> list[float]:
    if value is None:
        return [0.0] * length
    values = list(value)
    return [float(item) for item in values]


def _route_record(
    *,
    item: Mapping[str, Any],
    route_name: str,
    actions: list[int],
    source: Mapping[str, Any],
    workflow_identity: str,
    operator_version: int,
    route_result_hit: bool,
    split: str = "train",
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    qid = _qid(item)
    result = dict(source)
    stage_costs = _normalise_vector(result.get("stage_costs"), NUM_PHYSICAL_STAGES)
    decision_costs = _normalise_vector(
        result.get("decision_stage_costs"), NUM_DECISION_STAGES
    )
    fixed_costs = _normalise_vector(
        result.get("fixed_stage_costs"), NUM_PHYSICAL_STAGES
    )
    stage_latencies = _normalise_vector(
        result.get("stage_latencies_ms"), NUM_PHYSICAL_STAGES
    )
    decision_latencies = _normalise_vector(
        result.get("decision_stage_latencies_ms"), NUM_DECISION_STAGES
    )
    fixed_latencies = _normalise_vector(
        result.get("fixed_stage_latencies_ms"), NUM_PHYSICAL_STAGES
    )
    f1 = float(result.get("f1_score", 0.0) or 0.0)
    total_cost = float(result.get("total_cost", sum(stage_costs)) or 0.0)
    total_latency = float(result.get("total_latency_ms", sum(stage_latencies)) or 0.0)
    metadata = copy.deepcopy(dict(item.get("metadata") or {}))
    for field in ("question_type", "evidence_list"):
        if field in item:
            metadata[field] = copy.deepcopy(item[field])
    if isinstance(result.get("metadata"), Mapping):
        metadata.update(copy.deepcopy(dict(result["metadata"])))
    return {
        "split": str(split),
        "qid": qid,
        "question_id": qid,
        "route_name": route_name,
        "route_key": route_key(actions),
        "actions": list(actions),
        "status": status,
        "operator_version": int(operator_version),
        "workflow_identity": workflow_identity,
        "f1_score": f1,
        "f1_metric_version": F1_METRIC_VERSION,
        "reward": float(result.get("reward", f1) or 0.0),
        "reward_version": str(result.get("reward_version", REWARD_VERSION)),
        "answer": str(result.get("answer", "") or ""),
        "ground_truth": str(
            result.get("ground_truth", item.get("answer", "")) or ""
        ),
        "total_cost": total_cost,
        "stage_costs": stage_costs,
        "decision_stage_costs": decision_costs,
        "fixed_stage_costs": fixed_costs,
        "total_latency_ms": total_latency,
        "stage_latencies_ms": stage_latencies,
        "decision_stage_latencies_ms": decision_latencies,
        "fixed_stage_latencies_ms": fixed_latencies,
        "cache_hits": list(result.get("cache_hits", []) or []),
        "fixed_stage_idxs": list(result.get("fixed_stage_idxs", [1, 3, 6]) or []),
        "fixed_stage_default_tier": int(result.get("fixed_stage_default_tier", 0) or 0),
        "index_identity": str(result.get("index_identity", "") or ""),
        "metadata": metadata,
        "route_result_hit": bool(route_result_hit),
        "error": str(error or result.get("error", "") or ""),
        "created_at": str(result.get("created_at") or time.time()),
    }


def _trajectory_from_route(record: Mapping[str, Any]) -> TrajectoryRecord:
    return TrajectoryRecord(
        question_id=int(record["question_id"]),
        split=str(record.get("split", "train")),
        route_name=str(record["route_name"]),
        route_key=str(record["route_key"]),
        actions=[int(value) for value in record.get("actions", [])],
        f1_score=float(record.get("f1_score", 0.0)),
        reward=float(record.get("reward", record.get("f1_score", 0.0))),
        answer=str(record.get("answer", "") or ""),
        ground_truth=str(record.get("ground_truth", "") or ""),
        total_cost=float(record.get("total_cost", 0.0)),
        stage_costs=list(record.get("stage_costs", []) or []),
        decision_stage_costs=list(record.get("decision_stage_costs", []) or []),
        fixed_stage_costs=list(record.get("fixed_stage_costs", []) or []),
        total_latency_ms=float(record.get("total_latency_ms", 0.0)),
        stage_latencies_ms=list(record.get("stage_latencies_ms", []) or []),
        decision_stage_latencies_ms=list(
            record.get("decision_stage_latencies_ms", []) or []
        ),
        fixed_stage_latencies_ms=list(record.get("fixed_stage_latencies_ms", []) or []),
        metadata=dict(record.get("metadata") or {}),
        operator_version=int(record.get("operator_version", 1)),
        workflow_identity=str(record.get("workflow_identity", "")),
        index_identity=str(record.get("index_identity", "")),
    )


def _validate_route_metrics(record: Mapping[str, Any]) -> None:
    if not _finite(record.get("f1_score")) or not _finite(record.get("reward")):
        raise ValueError("non-finite F1/reward")
    if not math.isclose(
        float(record.get("reward", 0.0)),
        float(record.get("f1_score", 0.0)),
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        raise ValueError("reward != f1_score")
    if not _finite(record.get("total_cost")) or float(record.get("total_cost", 0.0)) < 0:
        raise ValueError("non-finite or negative route cost")


def _failed_route_record(
    *,
    item: Mapping[str, Any],
    route_name: str,
    actions: list[int],
    identity: str,
    operator_version: int,
    error: BaseException,
    split: str = "train",
) -> dict[str, Any]:
    stage_index = getattr(error, "stage_index", None)
    error_type = type(error).__name__
    record = _route_record(
        item=item,
        route_name=route_name,
        actions=actions,
        source={},
        workflow_identity=identity,
        operator_version=operator_version,
        route_result_hit=False,
        split=split,
        status="error",
        error=str(error),
    )
    record.update(
        {
            "failed_stage_idx": stage_index,
            "failed_stage_name": getattr(error, "stage_name", None),
            "failed_decision_stage_idx": getattr(
                error, "decision_stage_index", None
            ),
            "failed_decision_action": getattr(error, "decision_action", None),
            "failed_tier": getattr(error, "tier", None),
            "failed_tier_name": getattr(error, "tier_name", None),
            "error_type": error_type,
            "cause_type": str(getattr(error, "original_error_type", error_type)),
            "error_category": pair_error_category(error),
            "error_fingerprint": pair_error_fingerprint(
                error,
                error_type=error_type,
                stage_index=stage_index,
            ),
        }
    )
    return record


def _cached_route_record(
    item: Mapping[str, Any],
    route_name: str,
    route_cache: RouteResultCache,
    identities: Any,
    *,
    routes: Mapping[str, Iterable[int]],
    operator_version: int,
    split: str = "train",
) -> dict[str, Any] | None:
    """Return a normalized exact successful route-cache hit, if one exists."""

    if route_name not in routes:
        raise ValueError(f"Unknown route_name: {route_name}")
    actions = list(routes[route_name])
    identity = _identity_string(identities)
    cached = route_cache.get(
        split=split,
        qid=_qid(item),
        route_key=route_key(actions),
        operator_version=int(operator_version),
        workflow_identity=identity,
    )
    if cached is None or str(cached.get("status", "ok")) != "ok":
        return None
    record = _route_record(
        item=item,
        route_name=route_name,
        actions=actions,
        source=cached,
        workflow_identity=identity,
        operator_version=operator_version,
        route_result_hit=True,
        split=split,
    )
    _validate_route_metrics(record)
    return record


def _execute_missing_route(
    item: Mapping[str, Any],
    route_name: str,
    wrapper: Any,
    transition_cache: TransitionCache,
    identities: Any,
    *,
    routes: Mapping[str, Iterable[int]],
    operator_version: int,
    cache_only: bool,
    split: str = "train",
) -> dict[str, Any]:
    """Compute one missing route without mutating collector-owned state."""

    if route_name not in routes:
        raise ValueError(f"Unknown route_name: {route_name}")
    actions = list(routes[route_name])
    identity = _identity_string(identities)
    try:
        rollout = wrapper.rollout_with_cache(
            item,
            actions,
            transition_cache,
            operator_version=int(operator_version),
            cache_only=bool(cache_only),
        )
        if rollout is None:
            raise LookupError("rollout_with_cache returned None")
        if isinstance(rollout, Mapping) and str(rollout.get("status", "ok")) == "error":
            raise RuntimeError(str(rollout.get("error", "rollout failed")))
        record = _route_record(
            item=item,
            route_name=route_name,
            actions=actions,
            source=rollout,
            workflow_identity=identity,
            operator_version=operator_version,
            route_result_hit=False,
            split=split,
        )
        _validate_route_metrics(record)
        return record
    except TransitionCacheIntegrityError:
        raise
    except Exception as exc:
        if not is_content_safety_error(exc) and is_fatal_collection_error(exc):
            raise
        return _failed_route_record(
            item=item,
            route_name=route_name,
            actions=actions,
            identity=identity,
            operator_version=operator_version,
            error=exc,
            split=split,
        )


def collect_route(
    item: Mapping[str, Any],
    route_name: str,
    wrapper: Any,
    transition_cache: TransitionCache,
    route_cache: RouteResultCache,
    trajectory_buffer: TrajectoryBuffer,
    identities: Any,
    *,
    routes: Mapping[str, Iterable[int]] = FIXED_ROUTES,
    operator_version: int = 1,
    cache_only: bool = False,
    split: str = "train",
) -> dict[str, Any]:
    """Materialise one route, reusing a completed route result when present."""

    cached = _cached_route_record(
        item,
        route_name,
        route_cache,
        identities,
        routes=routes,
        operator_version=operator_version,
        split=split,
    )
    if cached is not None:
        trajectory_buffer.upsert_by_question_route(_trajectory_from_route(cached))
        return cached

    record = _execute_missing_route(
        item,
        route_name,
        wrapper,
        transition_cache,
        identities,
        routes=routes,
        operator_version=operator_version,
        cache_only=bool(cache_only),
        split=split,
    )
    route_cache.append(record)
    if record.get("status") == "ok":
        trajectory_buffer.upsert_by_question_route(_trajectory_from_route(record))
    return record


class FixedRouteCollector:
    """Single-process route-major collector with atomic ``new1`` snapshots."""

    def __init__(
        self,
        cache_root: str | os.PathLike[str],
        wrapper: Any,
        transition_cache: TransitionCache | None = None,
        route_cache: RouteResultCache | None = None,
        trajectory_buffer: TrajectoryBuffer | None = None,
        identities: Any = "",
        *,
        operator_version: int = 1,
        autosave_every: int = 10,
        expected_routes: int | None = None,
        routes: Mapping[str, Iterable[int]] | None = None,
        route_order: Iterable[str] | None = None,
        route_profile_name: str = "fixed_anchors_v1",
        route_profile_sha256: str = "",
        excluded_query_ids: Iterable[int] = (),
        n_workers: int = 1,
        max_inflight: int | None = None,
        split: str = "train",
        artifact_name: str = "new1",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.wrapper = wrapper
        self.split = str(split)
        self.artifact_name = str(artifact_name)
        if not self.split:
            raise ValueError("split must not be empty")
        if (
            not self.artifact_name
            or Path(self.artifact_name).name != self.artifact_name
            or self.artifact_name in {".", ".."}
        ):
            raise ValueError("artifact_name must be a plain file-name prefix")
        self.buffer_path = self.cache_root / f"{self.artifact_name}.json"
        self.db_path = self.cache_root / f"{self.artifact_name}.db"
        self.manifest_path = self.cache_root / f"{self.artifact_name}_manifest.json"
        self.failure_queue_path = (
            self.cache_root / f"{self.artifact_name}_failed_pairs.json"
        )
        self.identities = identities
        self.workflow_identity = _identity_string(identities)
        self.operator_version = int(operator_version)
        self.autosave_every = max(1, int(autosave_every))
        self.expected_routes = expected_routes
        self.n_workers = max(1, int(n_workers))
        self.max_inflight = (
            self.n_workers * 2 if max_inflight is None else int(max_inflight)
        )
        if self.max_inflight < self.n_workers:
            raise ValueError("max_inflight must be >= n_workers")
        self.scheduler = "route_major_qid_parallel_v1"
        configured_routes = FIXED_ROUTES if routes is None else routes
        self.routes: dict[str, tuple[int, ...]] = {}
        for route_name, raw_actions in configured_routes.items():
            actions = tuple(int(action) for action in raw_actions)
            if len(actions) != NUM_DECISION_STAGES:
                raise ValueError(
                    f"route {route_name!r} must contain exactly four actions"
                )
            if any(action < 0 or action > 3 for action in actions):
                raise ValueError(f"route {route_name!r} actions must be in 0..3")
            if actions[-1] == 3:
                raise ValueError(f"route {route_name!r} cannot skip the Answer stage")
            self.routes[str(route_name)] = actions
        configured_order = (
            BASELINE_ORDER
            if route_order is None and routes is None
            else (tuple(self.routes) if route_order is None else tuple(route_order))
        )
        self.route_order = tuple(str(route_name) for route_name in configured_order)
        if (
            len(self.route_order) != len(self.routes)
            or len(set(self.route_order)) != len(self.route_order)
            or set(self.route_order) != set(self.routes)
        ):
            raise ValueError(
                "route_order must contain every configured route exactly once"
            )
        self.route_profile_name = str(route_profile_name)
        self.route_profile_sha256 = str(route_profile_sha256)
        self.excluded_query_ids = {int(qid) for qid in excluded_query_ids}
        self._initial_excluded_query_ids = frozenset(self.excluded_query_ids)
        self.transition_cache = transition_cache or TransitionCache(
            self.db_path, workflow_identity=self.workflow_identity
        )
        self.trajectory_buffer = trajectory_buffer or TrajectoryBuffer(
            self.buffer_path,
            workflow_identity=self.workflow_identity,
            operator_version=self.operator_version,
        )
        self.trajectory_buffer.remove_question_ids(self.excluded_query_ids)
        self.failure_queue = FailedPairQueue(
            self.failure_queue_path,
            workflow_identity=self.workflow_identity,
            operator_version=self.operator_version,
        )
        self._systemic_error_threshold = 3
        self._pair_error_qids_by_fingerprint: dict[str, set[int]] = {}
        self.route_caches: dict[str, RouteResultCache] = {}
        if route_cache is not None:
            cache_route_name = (
                "all_t1"
                if "all_t1" in self.routes
                else (self.route_order[0] if self.route_order else "")
            )
            if cache_route_name:
                self.route_caches[cache_route_name] = route_cache
        self._new_successes = 0
        self._stop_reason = ""
        self._manifest_extra: dict[str, Any] = {}
        existing_manifest = self.manifest_path
        if existing_manifest.exists():
            try:
                old = json.loads(existing_manifest.read_text(encoding="utf-8"))
                old_identity = str(old.get("workflow_identity", ""))
                if old_identity and old_identity != self.workflow_identity:
                    raise ValueError(
                        "workflow identity mismatch: "
                        f"manifest={old_identity}, requested={self.workflow_identity}"
                    )
            except ValueError:
                raise
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid {self.artifact_name} manifest: {exc}"
                ) from exc

    def _route_cache(self, route_name: str) -> RouteResultCache:
        if route_name not in self.route_caches:
            self.route_caches[route_name] = RouteResultCache(
                self.cache_root / "baselines" / route_name / "results.jsonl"
            )
        return self.route_caches[route_name]

    def _effective_completed_routes(self) -> int:
        canonical_route_keys = {
            route_key(actions) for actions in self.routes.values()
        }
        return sum(
            1
            for record in self.trajectory_buffer
            if str(record.split) == self.split
            and str(record.route_key) in canonical_route_keys
            and int(record.operator_version) == self.operator_version
            and str(record.workflow_identity) == self.workflow_identity
            and int(record.question_id) not in self.excluded_query_ids
        )

    def _new_excluded_query_count(self) -> int:
        return len(
            self.excluded_query_ids.difference(self._initial_excluded_query_ids)
        )

    def _commit_records(
        self, route_name: str, records: Iterable[Mapping[str, Any]]
    ) -> None:
        """Persist worker results and update all main-thread-owned state."""

        materialized = [dict(record) for record in records]
        newly_executed = [
            record
            for record in materialized
            if not bool(record.get("route_result_hit"))
        ]
        if newly_executed:
            self._route_cache(route_name).append_many(newly_executed)

        added_exclusion = False
        for record in materialized:
            if record.get("status") == "ok":
                self.trajectory_buffer.upsert_by_question_route(
                    _trajectory_from_route(record)
                )
                self.failure_queue.resolve(record, resolution="success")
                if not record.get("route_result_hit"):
                    self._new_successes += 1
            elif is_content_safety_error(record.get("error", "")):
                qid = int(record.get("question_id", record.get("qid")))
                self.trajectory_buffer.remove_question_ids({qid})
                self.failure_queue.exclude_question(qid)
                if qid not in self.excluded_query_ids:
                    self.excluded_query_ids.add(qid)
                    added_exclusion = True
            elif record.get("status") == "error":
                self.failure_queue.record(record)

        if added_exclusion:
            self.save(status="running")
        if self._new_successes >= self.autosave_every:
            self.save()
            self._new_successes = 0

    def _is_systemic_pair_failure(self, record: Mapping[str, Any]) -> bool:
        if str(record.get("error_category", "")) != "pair_exception":
            return False
        fingerprint = str(record.get("error_fingerprint", ""))
        if not fingerprint:
            return False
        qid = int(record.get("question_id", record.get("qid")))
        qids = self._pair_error_qids_by_fingerprint.setdefault(fingerprint, set())
        qids.add(qid)
        return len(qids) >= self._systemic_error_threshold

    def collect_one(
        self,
        item: Mapping[str, Any],
        route_name: str,
        *,
        cache_only: bool = False,
    ) -> dict[str, Any]:
        if route_name not in self.routes:
            raise ValueError(f"Unknown route_name: {route_name}")
        qid = _qid(item)
        if qid in self.excluded_query_ids:
            return _route_record(
                item=item,
                route_name=route_name,
                actions=list(self.routes[route_name]),
                source={},
                workflow_identity=self.workflow_identity,
                operator_version=self.operator_version,
                route_result_hit=False,
                split=self.split,
                status="excluded",
                error="query excluded after content-safety failure",
            )
        result = collect_route(
            item,
            route_name,
            self.wrapper,
            self.transition_cache,
            self._route_cache(route_name),
            self.trajectory_buffer,
            self.identities,
            routes=self.routes,
            operator_version=self.operator_version,
            cache_only=bool(cache_only),
            split=self.split,
        )
        if result.get("status") == "error" and is_content_safety_error(
            result.get("error", "")
        ):
            self.trajectory_buffer.remove_question_ids({qid})
            self.failure_queue.exclude_question(qid)
            if qid not in self.excluded_query_ids:
                self.excluded_query_ids.add(qid)
                self.save(status="running")
        elif result.get("status") == "error":
            self.failure_queue.record(result)
        elif result.get("status") == "ok":
            self.failure_queue.resolve(result, resolution="success")
        if result.get("status") == "ok" and not result.get("route_result_hit"):
            self._new_successes += 1
        if self._new_successes >= self.autosave_every:
            self.save()
            self._new_successes = 0
        return result

    def collect(
        self,
        items: Iterable[Mapping[str, Any]],
        route_names: Iterable[str] | None = None,
        *,
        cache_only: bool = False,
    ) -> dict[str, Any]:
        materialized = list(items)
        selected_routes = list(self.route_order if route_names is None else route_names)
        self._cache_only = bool(cache_only)
        fatal_error: Exception | None = None
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for route_name in selected_routes:
                if route_name not in self.routes:
                    raise ValueError(f"Unknown route_name: {route_name}")

                canonical_route_key = route_key(self.routes[route_name])
                unique_items: list[
                    tuple[Mapping[str, Any], tuple[str, int, str, int, str]]
                ] = []
                seen_exact_pairs: set[tuple[str, int, str, int, str]] = set()
                for item in materialized:
                    qid = _qid(item)
                    if qid in self.excluded_query_ids:
                        continue
                    exact_pair = (
                        self.split,
                        qid,
                        canonical_route_key,
                        self.operator_version,
                        self.workflow_identity,
                    )
                    if exact_pair in seen_exact_pairs:
                        continue
                    seen_exact_pairs.add(exact_pair)
                    unique_items.append((item, exact_pair))

                remaining_items: deque[
                    tuple[Mapping[str, Any], tuple[str, int, str, int, str]]
                ] = deque()
                cached_records: list[dict[str, Any]] = []
                route_cache = self._route_cache(route_name)
                for item, exact_pair in unique_items:
                    cached = _cached_route_record(
                        item,
                        route_name,
                        route_cache,
                        self.identities,
                        routes=self.routes,
                        operator_version=self.operator_version,
                        split=self.split,
                    )
                    if cached is None:
                        remaining_items.append((item, exact_pair))
                    else:
                        cached_records.append(cached)
                self._commit_records(route_name, cached_records)

                stop = False
                deferred_items: list[Mapping[str, Any]] = []
                pending: dict[Future[dict[str, Any]], Mapping[str, Any]] = {}
                scheduled_keys: set[tuple[str, int, str, int, str]] = set()
                try:
                    while remaining_items or pending:
                        while (
                            remaining_items
                            and len(pending) < self.max_inflight
                            and not stop
                        ):
                            item, exact_pair = remaining_items.popleft()
                            if exact_pair in scheduled_keys:
                                continue
                            scheduled_keys.add(exact_pair)
                            future = executor.submit(
                                _execute_missing_route,
                                item,
                                route_name,
                                self.wrapper,
                                self.transition_cache,
                                self.identities,
                                routes=self.routes,
                                operator_version=self.operator_version,
                                cache_only=bool(cache_only),
                                split=self.split,
                            )
                            pending[future] = item

                        if not pending:
                            break
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        completed: list[dict[str, Any]] = []
                        for future in done:
                            item = pending.pop(future)
                            try:
                                record = future.result()
                                completed.append(record)
                                if (
                                    record.get("status") == "error"
                                    and not is_content_safety_error(
                                        record.get("error", "")
                                    )
                                ):
                                    deferred_items.append(item)
                            except Exception as exc:
                                if fatal_error is None:
                                    fatal_error = exc
                                if not self._stop_reason:
                                    self._stop_reason = str(exc)
                                stop = True
                            if stop:
                                for pending_future in list(pending):
                                    if pending_future.cancel():
                                        pending.pop(pending_future)
                        self._commit_records(route_name, completed)
                        for record in completed:
                            if (
                                record.get("status") == "error"
                                and not is_content_safety_error(
                                    record.get("error", "")
                                )
                                and self._is_systemic_pair_failure(record)
                            ):
                                fatal_error = RuntimeError(
                                    "systemic pair failure repeated across three qids: "
                                    f"stage={record.get('failed_stage_name')} "
                                    f"error={record.get('error_type')}"
                                )
                                self._stop_reason = str(fatal_error)
                                stop = True
                        if stop:
                            for pending_future in list(pending):
                                if pending_future.cancel():
                                    pending.pop(pending_future)

                    if not stop:
                        for item in deferred_items:
                            qid = _qid(item)
                            if qid in self.excluded_query_ids:
                                continue
                            try:
                                record = _execute_missing_route(
                                    item,
                                    route_name,
                                    self.wrapper,
                                    self.transition_cache,
                                    self.identities,
                                    routes=self.routes,
                                    operator_version=self.operator_version,
                                    cache_only=bool(cache_only),
                                    split=self.split,
                                )
                            except Exception as exc:
                                fatal_error = exc
                                self._stop_reason = str(exc)
                                stop = True
                                break
                            self._commit_records(route_name, [record])
                except KeyboardInterrupt:
                    self._stop_reason = "KeyboardInterrupt"
                    for pending_future in list(pending):
                        if pending_future.cancel():
                            pending.pop(pending_future)
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        completed = []
                        for future in done:
                            pending.pop(future)
                            if future.cancelled():
                                continue
                            try:
                                completed.append(future.result())
                            except Exception:
                                continue
                        self._commit_records(route_name, completed)
                    self.save(status="stopped")
                    raise

                self.save(status="stopped" if stop else "running")
                if stop:
                    break

        if fatal_error is not None:
            raise fatal_error
        effective_expected_routes = max(
            0,
            int(self.expected_routes or 0)
            - self._new_excluded_query_count() * len(self.routes),
        )
        complete = (
            (
                self.expected_routes is None
                or self.expected_routes <= 0
                or self._effective_completed_routes() >= effective_expected_routes
            )
        )
        unresolved_failures, _ = self.failure_queue.counts()
        final_status = (
            "completed"
            if complete and unresolved_failures == 0
            else ("completed_with_failures" if unresolved_failures else "running")
        )
        self.save(
            status=final_status
        )
        return self.manifest

    def _summary(self, route_name: str) -> dict[str, Any]:
        cache = self._route_cache(route_name)
        canonical_route_key = route_key(self.routes[route_name])
        rows = [
            row
            for row in getattr(cache, "_index", {}).values()
            if str(row.get("split", "")) == self.split
            and str(row.get("route_key", "")) == canonical_route_key
            and int(row.get("operator_version", -1)) == self.operator_version
            and str(row.get("workflow_identity", "")) == self.workflow_identity
            and int(row.get("question_id", row.get("qid")))
            not in self.excluded_query_ids
        ]
        ok = [row for row in rows if row.get("status") == "ok"]
        errors = [row for row in rows if row.get("status") != "ok"]
        total_f1 = sum(float(row.get("f1_score", 0.0) or 0.0) for row in ok)
        total_cost = sum(float(row.get("total_cost", 0.0) or 0.0) for row in ok)
        total_latency = sum(float(row.get("total_latency_ms", 0.0) or 0.0) for row in ok)
        return {
            "split": self.split,
            "route_name": route_name,
            "route_key": canonical_route_key,
            "actions": list(self.routes[route_name]),
            "operator_version": self.operator_version,
            "workflow_identity": self.workflow_identity,
            "n_ok": len(ok),
            "n_error": len(errors),
            "completed": False,
            "total_f1": total_f1,
            "avg_f1": total_f1 / len(ok) if ok else 0.0,
            "total_cost": total_cost,
            "avg_cost": total_cost / len(ok) if ok else 0.0,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(ok) if ok else 0.0,
        }

    def save(self, *, status: str = "running") -> None:
        self.trajectory_buffer.remove_question_ids(self.excluded_query_ids)
        self.trajectory_buffer.save(self.buffer_path)
        unresolved_failures, resolved_failures = self.failure_queue.counts()
        summaries: dict[str, Any] = {}
        expected_per_route = (
            max(
                0,
                int(self.expected_routes) // len(self.routes)
                - self._new_excluded_query_count(),
            )
            if self.expected_routes and self.routes
            else 0
        )
        for route_name in self.route_order:
            summary = self._summary(route_name)
            summary["completed"] = (
                summary["n_error"] == 0
                and (
                    expected_per_route <= 0
                    or summary["n_ok"] >= expected_per_route
                )
            )
            summaries[route_name] = summary
            _atomic_json_dump(
                summary,
                self.cache_root / "baselines" / route_name / "summary.json",
            )
        self.manifest = {
            "version": 1,
            "status": status,
            "workflow_identity": self.workflow_identity,
            "identity": _identity_dict(self.identities),
            "operator_version": self.operator_version,
            "route_profile_name": self.route_profile_name,
            "route_profile_sha256": self.route_profile_sha256,
            "route_count_per_query": len(self.routes),
            "excluded_query_ids": sorted(self.excluded_query_ids),
            "expected_routes": int(self.expected_routes or 0),
            "completed_routes": self._effective_completed_routes(),
            "total_cost": sum(float(item.get("total_cost", 0.0) or 0.0) for item in summaries.values()),
            "total_latency_ms": sum(float(item.get("total_latency_ms", 0.0) or 0.0) for item in summaries.values()),
            "baselines": summaries,
            "f1_metric_version": F1_METRIC_VERSION,
            "reward_version": REWARD_VERSION,
            "n_workers": self.n_workers,
            "max_inflight": self.max_inflight,
            "scheduler": self.scheduler,
            "failed_pairs_unresolved": unresolved_failures,
            "failed_pairs_resolved": resolved_failures,
            "dataset_split": self.split,
            "artifact_name": self.artifact_name,
            "failed_pair_queue": self.failure_queue_path.name,
            "stop_reason": self._stop_reason if status != "completed" else "",
            "updated_at": time.time(),
            **self._manifest_extra,
        }
        _atomic_json_dump(self.manifest, self.manifest_path)


__all__ = [
    "FIXED_ROUTES",
    "FIXED_BASELINE_ROUTES",
    "BASELINE_ORDER",
    "F1_METRIC_VERSION",
    "REWARD_VERSION",
    "FixedRouteCollector",
    "collect_route",
]
