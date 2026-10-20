"""Query-level exclusion and finalisation for fixed-route RAG buffers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .error_policy import is_content_safety_error
from .route_result_cache import RouteResultCache


ROUTE_NAMES = ("all_t1", "all_t2", "all_t3")
QUERY_FILTER_POLICY = "exclude_query_if_any_route_error_v1"


def latest_route_statuses(
    cache_root: str | os.PathLike[str],
    *,
    route_names: Iterable[str] = ROUTE_NAMES,
    split: str = "train",
    workflow_identity: str = "",
    operator_version: int | None = None,
) -> dict[int, dict[str, str]]:
    """Return the latest status for every qid/route in the route caches."""

    statuses: dict[int, dict[str, str]] = {}
    root = Path(cache_root)
    for route_name in route_names:
        cache = RouteResultCache(root / "baselines" / route_name / "results.jsonl")
        for row in getattr(cache, "_index", {}).values():
            if str(row.get("split", "")) != str(split):
                continue
            if workflow_identity and str(row.get("workflow_identity", "")) != str(
                workflow_identity
            ):
                continue
            if operator_version is not None and int(row.get("operator_version", 1)) != int(
                operator_version
            ):
                continue
            raw_qid = row.get("qid", row.get("question_id"))
            if raw_qid is None:
                continue
            qid = int(raw_qid)
            status = str(row.get("status", "ok"))
            if status == "ok":
                value = "ok"
            else:
                error = str(row.get("error", "") or "")
                value = error if error.strip() else status
            statuses.setdefault(qid, {})[route_name] = value
    return statuses


def content_safety_query_ids(
    statuses: Mapping[int, Mapping[str, object]],
) -> set[int]:
    """Return qids with a deterministic content-safety route failure."""

    return {
        int(qid)
        for qid, routes in statuses.items()
        if any(is_content_safety_error(status) for status in routes.values())
    }


def complete_query_ids(
    statuses: Mapping[int, Mapping[str, object]],
    route_names: Iterable[str],
) -> set[int]:
    """Return qids whose required routes all have successful latest rows."""

    required = tuple(route_names)
    return {
        int(qid)
        for qid, routes in statuses.items()
        if all(str(routes.get(name, "missing")) == "ok" for name in required)
    }


def invalid_query_ids(
    cache_root: str | os.PathLike[str],
    *,
    route_names: Iterable[str] = ROUTE_NAMES,
    split: str = "train",
    workflow_identity: str = "",
    operator_version: int | None = None,
) -> set[int]:
    """Return qids with any non-success route status."""

    statuses = latest_route_statuses(
        cache_root,
        route_names=route_names,
        split=split,
        workflow_identity=workflow_identity,
        operator_version=operator_version,
    )
    return {
        qid
        for qid, route_status in statuses.items()
        if any(status != "ok" for status in route_status.values())
    }


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
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def filter_trajectory_file(
    buffer_path: str | os.PathLike[str], excluded_query_ids: Iterable[int]
) -> tuple[int, int]:
    """Remove every trajectory belonging to an excluded qid atomically."""

    path = Path(buffer_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("trajectory buffer must contain a JSON list")
    excluded = {int(qid) for qid in excluded_query_ids}
    kept: list[Mapping[str, Any]] = []
    for value in data:
        if not isinstance(value, Mapping):
            raise ValueError("trajectory buffer records must be objects")
        raw_qid = value.get("question_id", value.get("qid"))
        if raw_qid is None or int(raw_qid) not in excluded:
            kept.append(dict(value))
    _atomic_json_dump(kept, path)
    return len(data), len(kept)


__all__ = [
    "QUERY_FILTER_POLICY",
    "ROUTE_NAMES",
    "complete_query_ids",
    "content_safety_query_ids",
    "filter_trajectory_file",
    "invalid_query_ids",
    "latest_route_statuses",
]
