"""Small offline verifier for the RAG ``new1`` cache.

The verifier intentionally checks consistency and arithmetic only.  It does
not invoke an LLM, embedding service, or retrieval quality benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from RAG.rl.counterfactual_route_ladder import (
    RouteProfile,
    RouteTemplate,
    enumerate_legal_routes,
    get_route_profile,
)
from RAG.rl.route_result_cache import route_key


_DEFAULT_PROFILE = get_route_profile("anchors")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_route_plan(path: Path) -> RouteProfile:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("route plan must contain an object")
    if str(payload.get("schema_version", "")) == "rag_observed_route_plan_v3":
        from RAG.scripts.collect_fixed_routes import _load_observed_route_plan
        return _load_observed_route_plan(path)
    if str(payload.get("schema_version", "")) != "rag_random_route_plan_v1":
        raise ValueError("unsupported route plan schema_version")
    legal_payload = json.dumps(
        enumerate_legal_routes(), separators=(",", ":")
    ).encode("utf-8")
    expected_legal_sha = hashlib.sha256(legal_payload).hexdigest()
    if str(payload.get("legal_route_sha256", "")) != expected_legal_sha:
        raise ValueError("route plan legal universe sha256 mismatch")
    raw_routes = payload.get("routes")
    if not isinstance(raw_routes, list):
        raise ValueError("route plan routes must be a list")
    routes: list[RouteTemplate] = []
    for expected_rank, row in enumerate(raw_routes):
        if not isinstance(row, Mapping):
            raise ValueError("route plan entries must be objects")
        rank = int(row.get("rank", -1))
        actions = tuple(int(value) for value in row.get("actions", []))
        if rank != expected_rank:
            raise ValueError("route plan ranks must be contiguous and ordered")
        if (
            len(actions) != 4
            or any(action not in (0, 1, 2, 3) for action in actions[:3])
            or actions[3] not in (0, 1, 2)
        ):
            raise ValueError(f"illegal RAG route actions: {actions}")
        routes.append(RouteTemplate(rank, str(row.get("name", "")), actions))
    if int(payload.get("count", -1)) != len(routes):
        raise ValueError("route plan count mismatch")
    if len({route.name for route in routes}) != len(routes):
        raise ValueError("route plan contains duplicate names")
    if len({route.actions for route in routes}) != len(routes):
        raise ValueError("route plan contains duplicate action tuples")
    profile = RouteProfile(str(payload.get("route_profile_name", "")), tuple(routes))
    if not profile.name:
        raise ValueError("route plan requires route_profile_name")
    if str(payload.get("route_profile_sha256", "")) != profile.sha256:
        raise ValueError("route plan profile sha256 mismatch")
    return profile


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(dict(value))
    return rows


def _finite(value: Any, *, nonnegative: bool = False) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and (not nonnegative or number >= 0.0)


def _close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-10)
    except (TypeError, ValueError):
        return False


def _sum(values: Iterable[Any]) -> float:
    return sum(float(value or 0.0) for value in values)


def _latest_route_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        qid = row.get("qid", row.get("question_id"))
        if qid is None:
            continue
        actions = row.get("actions") or []
        key = (
            str(row.get("split", "")),
            int(qid),
            str(row.get("route_key") or route_key(actions)),
            int(row.get("operator_version", 1)),
            str(row.get("workflow_identity", "")),
        )
        latest[key] = dict(row)
    return latest


def _verify_sqlite(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "missing", ""
    connection = sqlite3.connect(path)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        row = connection.execute(
            "SELECT value FROM cache_meta WHERE key='workflow_identity'"
        ).fetchone()
        return integrity, str(row[0]) if row else ""
    except sqlite3.DatabaseError as exc:
        return f"error: {exc}", ""
    finally:
        connection.close()


def verify_new1(
    cache_root: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    expected_routes: int | None = None,
    *,
    routes: Mapping[str, Iterable[int]] | None = None,
    profile_name: str | None = None,
    profile_sha256: str | None = None,
    artifact_name: str = "new1",
    route_plan_file: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe report; ``ok`` is false for any consistency error."""

    root = Path(cache_root)
    errors: list[str] = []
    route_map = {
        str(name): tuple(int(action) for action in actions)
        for name, actions in (
            _DEFAULT_PROFILE.route_map if routes is None else routes
        ).items()
    }
    resolved_profile_name = str(profile_name or _DEFAULT_PROFILE.name)
    resolved_profile_sha256 = str(profile_sha256 or _DEFAULT_PROFILE.sha256)
    approved_by_key: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name, actions in route_map.items():
        try:
            key = route_key(actions)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid approved route {name}: {exc}")
            continue
        if key in approved_by_key:
            errors.append(f"duplicate approved route key: {key}")
        approved_by_key[key] = (name, actions)
    artifact = str(artifact_name)
    if not artifact or Path(artifact).name != artifact:
        raise ValueError("artifact_name must be a plain file stem")
    manifest_path = root / f"{artifact}_manifest.json"
    buffer_path = root / f"{artifact}.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded = _load_json(manifest_path)
            if isinstance(loaded, Mapping):
                manifest = dict(loaded)
            else:
                errors.append(f"{manifest_path.name} is not an object")
        except Exception as exc:
            errors.append(f"manifest: {exc}")
    else:
        errors.append(f"missing {manifest_path.name}")

    records: list[dict[str, Any]] = []
    if buffer_path.exists():
        try:
            loaded = _load_json(buffer_path)
            if not isinstance(loaded, list):
                errors.append(f"{buffer_path.name} is not a list")
            else:
                records = [dict(value) for value in loaded if isinstance(value, Mapping)]
        except Exception as exc:
            errors.append(f"{buffer_path.name}: {exc}")
    else:
        errors.append(f"missing {buffer_path.name}")

    if route_plan_file is not None:
        plan_path = Path(route_plan_file)
        declared_plan_sha = str(manifest.get("route_plan_file_sha256", ""))
        actual_plan_sha = _sha256(plan_path)
        if declared_plan_sha != actual_plan_sha:
            errors.append("manifest route plan file sha256 mismatch")

    train_rows = _load_json(Path(train_path)) if Path(train_path).exists() else []
    test_rows = _load_json(Path(test_path)) if Path(test_path).exists() else []
    train_qids = {
        int(row.get("question_id", row.get("qid")))
        for row in train_rows
        if isinstance(row, Mapping) and row.get("question_id", row.get("qid")) is not None
    }
    test_qids = {
        int(row.get("question_id", row.get("qid")))
        for row in test_rows
        if isinstance(row, Mapping) and row.get("question_id", row.get("qid")) is not None
    }
    raw_excluded = manifest.get("excluded_query_ids", [])
    try:
        excluded_query_ids = {int(value) for value in raw_excluded}
    except (TypeError, ValueError):
        errors.append("manifest excluded_query_ids is invalid")
        excluded_query_ids = set()
    valid_train_qids = train_qids - excluded_query_ids
    if str(manifest.get("route_profile_name", "")) != resolved_profile_name:
        errors.append("manifest route profile name mismatch")
    if str(manifest.get("route_profile_sha256", "")) != resolved_profile_sha256:
        errors.append("manifest route profile sha256 mismatch")
    try:
        manifest_route_count = int(manifest.get("route_count_per_query", -1))
    except (TypeError, ValueError):
        manifest_route_count = -1
    if manifest_route_count != len(route_map):
        errors.append("manifest route_count_per_query mismatch")
    effective_full_count = len(valid_train_qids) * len(route_map)
    try:
        declared_expected_routes = int(
            manifest.get("expected_routes", effective_full_count)
        )
    except (TypeError, ValueError):
        errors.append("manifest expected_routes is invalid")
        declared_expected_routes = effective_full_count
    if expected_routes is None:
        # Collectors retain the pre-exclusion target in the manifest.  Once a
        # content-safety qid is excluded, the valid full-profile target shrinks
        # by one complete route block and must be verified at that effective size.
        expected_routes = min(declared_expected_routes, effective_full_count)
    else:
        expected_routes = int(expected_routes)

    expected_identity = str(manifest.get("workflow_identity", ""))
    try:
        expected_operator_version = int(manifest.get("operator_version", 1))
    except (TypeError, ValueError):
        errors.append("manifest operator_version is invalid")
        expected_operator_version = 1
    index_manifest_path = root / "vector_index" / "manifest.json"
    expected_index_identity = ""
    if index_manifest_path.exists():
        try:
            index_manifest = _load_json(index_manifest_path)
            expected_index_identity = str(index_manifest.get("index_identity", ""))
            manifest_identity = manifest.get("identity")
            if isinstance(manifest_identity, Mapping):
                declared_index = str(manifest_identity.get("index_identity", ""))
                if declared_index and expected_index_identity and declared_index != expected_index_identity:
                    errors.append("index identity mismatch between manifests")
        except Exception as exc:
            errors.append(f"vector index manifest: {exc}")
    route_pairs: dict[tuple[int, str], dict[str, Any]] = {}
    invalid_records: list[int] = []
    duplicate_pairs: list[list[Any]] = []
    test_overlap: list[int] = []
    for index, record in enumerate(records):
        if record.get("status", "ok") != "ok":
            continue
        try:
            raw_qid = record.get("question_id")
            if raw_qid is None:
                raw_qid = record.get("qid")
            qid = int(raw_qid)
            actions = [int(value) for value in record.get("actions", [])]
            rkey = str(record.get("route_key") or route_key(actions))
        except (KeyError, TypeError, ValueError):
            invalid_records.append(index)
            continue
        pair = (qid, rkey)
        if pair in route_pairs:
            duplicate_pairs.append([qid, rkey])
        route_pairs[pair] = record
        approved_route = approved_by_key.get(rkey)
        if approved_route is None:
            errors.append(f"record {index} has extra route key {rkey}")
        else:
            expected_name, expected_actions = approved_route
            if (
                str(record.get("route_name", "")) != expected_name
                or tuple(actions) != expected_actions
                or rkey != route_key(actions)
            ):
                errors.append(f"record {index} has wrong route name/actions/key")
        if str(record.get("split", "train")) != "train":
            errors.append(f"record {index} has wrong split")
        if int(record.get("operator_version", 1)) != expected_operator_version:
            errors.append(f"record {index} operator version mismatch")
        if qid in excluded_query_ids:
            errors.append(f"excluded qid {qid} present in {artifact} buffer")
        if qid in test_qids:
            test_overlap.append(qid)
        if expected_identity and str(record.get("workflow_identity", "")) != expected_identity:
            errors.append(f"record {index} workflow identity mismatch")
        if expected_index_identity and str(record.get("index_identity", "")) != expected_index_identity:
            errors.append(f"record {index} index identity mismatch")
        for metric in ("f1_score", "reward", "total_cost", "total_latency_ms"):
            if not _finite(record.get(metric), nonnegative=metric != "f1_score"):
                invalid_records.append(index)
        if not _close(record.get("reward"), record.get("f1_score")):
            errors.append(f"record {index} reward != f1_score")
        stage_costs = list(record.get("stage_costs", []) or [])
        decision_costs = list(record.get("decision_stage_costs", []) or [])
        fixed_costs = list(record.get("fixed_stage_costs", []) or [])
        stage_latencies = list(record.get("stage_latencies_ms", []) or [])
        decision_latencies = list(
            record.get("decision_stage_latencies_ms", []) or []
        )
        fixed_latencies = list(record.get("fixed_stage_latencies_ms", []) or [])
        cost_vectors_valid = (
            len(stage_costs) == 7
            and len(decision_costs) == 4
            and len(fixed_costs) == 7
            and all(
                _finite(value, nonnegative=True)
                for vector in (stage_costs, decision_costs, fixed_costs)
                for value in vector
            )
        )
        latency_vectors_valid = (
            len(stage_latencies) == 7
            and len(decision_latencies) == 4
            and len(fixed_latencies) == 7
            and all(
                _finite(value, nonnegative=True)
                for vector in (stage_latencies, decision_latencies, fixed_latencies)
                for value in vector
            )
        )
        if not cost_vectors_valid or not latency_vectors_valid:
            invalid_records.append(index)
        if cost_vectors_valid:
            if not _close(record.get("total_cost"), _sum(stage_costs)):
                errors.append(f"record {index} total_cost != sum(stage_costs)")
            if not _close(
                record.get("total_cost"), _sum(decision_costs) + _sum(fixed_costs)
            ):
                errors.append(f"record {index} total_cost != decision + fixed costs")

    # A materialized qid is useful only when its complete profile is present.
    # Partial smoke collections remain valid when expected_routes is smaller.
    successful_qids = {qid for qid, _ in route_pairs}
    expected_qids = (
        valid_train_qids
        if expected_routes == len(valid_train_qids) * len(route_map)
        else successful_qids
    )
    missing_routes: list[int] = []
    approved = set(approved_by_key)
    for qid in sorted(expected_qids):
        keys = {key for (item_qid, key) in route_pairs if item_qid == qid}
        if keys != approved:
            missing_routes.append(qid)
    if expected_routes and len(route_pairs) != int(expected_routes):
        errors.append(f"successful route count {len(route_pairs)} != {expected_routes}")
    if invalid_records:
        errors.append(f"invalid records: {sorted(set(invalid_records))[:20]}")
    if duplicate_pairs:
        errors.append(f"duplicate route pairs: {duplicate_pairs[:20]}")
    if missing_routes:
        errors.append(f"qids missing approved routes: {missing_routes[:20]}")
    if test_overlap:
        errors.append(f"test qids present: {sorted(set(test_overlap))[:20]}")
    unexpected_train = sorted(successful_qids - valid_train_qids)
    if unexpected_train:
        errors.append(f"non-train qids present: {unexpected_train[:20]}")

    # Check baseline files and summaries against their latest cache identity.
    baseline_reports: dict[str, Any] = {}
    for name, actions in route_map.items():
        results_path = root / "baselines" / name / "results.jsonl"
        summary_path = root / "baselines" / name / "summary.json"
        rows = _load_jsonl(results_path)
        latest = _latest_route_rows(rows)
        expected_key = route_key(actions)
        eligible = [
            row
            for (
                row_split,
                qid,
                row_route_key,
                row_version,
                row_identity,
            ), row in latest.items()
            if row_split == "train"
            and qid in valid_train_qids
            and row_route_key == expected_key
            and row_version == expected_operator_version
            and row_identity == expected_identity
        ]
        effective = [row for row in eligible if row.get("status") == "ok"]
        effective_error_count = len(eligible) - len(effective)
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = dict(_load_json(summary_path))
            except Exception as exc:
                errors.append(f"{name} summary: {exc}")
        else:
            errors.append(f"missing {name}/summary.json")
        if (
            str(summary.get("route_name", "")) != name
            or str(summary.get("route_key", "")) != expected_key
            or list(summary.get("actions", []) or []) != list(actions)
        ):
            errors.append(f"{name} summary route mismatch")
        try:
            summary_operator_version = int(summary.get("operator_version", -1))
        except (TypeError, ValueError):
            summary_operator_version = -1
        if summary_operator_version != expected_operator_version:
            errors.append(f"{name} summary operator version mismatch")
        if str(summary.get("workflow_identity", "")) != expected_identity:
            errors.append(f"{name} summary workflow identity mismatch")
        for row in effective:
            if (
                row.get("route_name") != name
                or row.get("route_key") != expected_key
                or list(row.get("actions", [])) != list(actions)
            ):
                errors.append(f"{name} contains wrong route")
            if int(row.get("operator_version", 1)) != expected_operator_version:
                errors.append(f"{name} operator version mismatch")
            if expected_identity and str(row.get("workflow_identity", "")) != expected_identity:
                errors.append(f"{name} workflow identity mismatch")
            if expected_index_identity and str(row.get("index_identity", "")) != expected_index_identity:
                errors.append(f"{name} index identity mismatch")
        total_cost = sum(float(row.get("total_cost", 0.0) or 0.0) for row in effective)
        if int(summary.get("n_ok", -1)) != len(effective):
            errors.append(f"{name} summary n_ok mismatch")
        if int(summary.get("n_error", -1)) != effective_error_count:
            errors.append(f"{name} summary n_error mismatch")
        if not _close(summary.get("total_cost"), total_cost):
            errors.append(f"{name} summary total_cost mismatch")
        expected_baseline_complete = (
            effective_error_count == 0
            and (
                int(expected_routes) <= 0
                or len(effective)
                >= (
                    len(valid_train_qids)
                    if int(expected_routes) == len(valid_train_qids) * len(route_map)
                    else int(expected_routes) // len(route_map)
                )
            )
        )
        if bool(summary.get("completed", False)) != expected_baseline_complete:
            errors.append(f"{name} summary completed mismatch")
        baseline_reports[name] = {
            "records": len(rows),
            "effective_ok": len(effective),
            "n_error": effective_error_count,
        }

    sqlite_integrity, sqlite_identity = _verify_sqlite(root / f"{artifact}.db")
    if sqlite_integrity != "ok":
        errors.append(f"SQLite integrity_check: {sqlite_integrity}")
    if expected_identity and sqlite_identity and sqlite_identity != expected_identity:
        errors.append("SQLite workflow identity mismatch")

    return {
        "ok": not errors,
        "errors": errors,
        "artifact_name": artifact,
        "declared_expected_routes": declared_expected_routes,
        "expected_routes": int(expected_routes),
        "route_profile_name": resolved_profile_name,
        "route_profile_sha256": resolved_profile_sha256,
        "successful_routes": len(route_pairs),
        "train_qids": len(train_qids),
        "excluded_query_ids": sorted(excluded_query_ids),
        "excluded_query_count": len(excluded_query_ids),
        "valid_train_qids": len(valid_train_qids),
        "successful_qids": len(successful_qids),
        "workflow_identity": expected_identity,
        "index_identity": expected_index_identity,
        "sqlite_integrity": sqlite_integrity,
        "baselines": baseline_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="RAG/cache")
    parser.add_argument("--train", default="RAG/data/train.json")
    parser.add_argument("--test", default="RAG/data/test.json")
    parser.add_argument("--expected-routes", type=int, default=None)
    parser.add_argument(
        "--profile", choices=("anchors",), default=None
    )
    parser.add_argument("--artifact-name", default="new1")
    parser.add_argument("--route-plan-file", type=Path)
    args = parser.parse_args(argv)
    if args.route_plan_file is not None and args.profile is not None:
        raise SystemExit("--profile cannot be combined with --route-plan-file")
    profile = (
        _load_route_plan(args.route_plan_file)
        if args.route_plan_file is not None
        else get_route_profile(args.profile or "anchors")
    )
    report = verify_new1(
        args.cache_root,
        args.train,
        args.test,
        args.expected_routes,
        routes=profile.route_map,
        profile_name=profile.name,
        profile_sha256=profile.sha256,
        artifact_name=args.artifact_name,
        route_plan_file=args.route_plan_file,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
