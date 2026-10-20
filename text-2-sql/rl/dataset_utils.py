"""Dataset loading helpers for ARTIST NL2SQL scripts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable

QUESTION_ID_KEYS = ("question_id", "id", "idx", "index")


def load_question_records(
    data_path: str | Path,
    max_queries: int | None = None,
    start_query: int = 0,
) -> list[dict[str, Any]]:
    """Load NL2SQL records and ensure every item has an integer question_id."""
    if start_query < 0:
        raise ValueError("start_query must be >= 0")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    records = _as_record_list(raw_data)
    records = records[start_query:]
    if max_queries is not None:
        records = records[:max_queries]
    return normalize_question_records(records, fallback_start=start_query)


def normalize_question_records(
    records: Iterable[dict[str, Any]],
    fallback_start: int = 0,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row_idx, raw in enumerate(records, start=fallback_start):
        item = dict(raw)
        question_id = _coerce_question_id(item, row_idx)
        if question_id in seen:
            raise ValueError(f"Duplicate question_id after normalization: {question_id}")
        seen.add(question_id)
        item["question_id"] = question_id
        normalized.append(item)
    return normalized


def question_map_from_records(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["question_id"]): item for item in records}


def _as_record_list(raw_data: Any) -> list[dict[str, Any]]:
    if isinstance(raw_data, list):
        return raw_data
    if isinstance(raw_data, dict):
        for key in ("data", "records", "questions", "examples"):
            value = raw_data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Expected dataset JSON to be a list of records or a dict containing a record list")


def _coerce_question_id(item: dict[str, Any], fallback_idx: int) -> int:
    for key in QUESTION_ID_KEYS:
        value = item.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            item[f"source_{key}"] = value
            break
    return int(fallback_idx)


def db_schema_fingerprint(
    db_root: str | Path, db_mode: str, db_ids: Iterable[str]
) -> str:
    """Hash the selected database files and descriptions for cache identity."""
    root = Path(db_root).resolve()
    mode_directory = f"{db_mode}_databases"
    databases_root = root if root.name.casefold() == mode_directory.casefold() else root / mode_directory
    files: list[Path] = []
    for db_id in sorted({str(value) for value in db_ids}):
        directory = databases_root / db_id
        sqlite_path = directory / f"{db_id}.sqlite"
        if not sqlite_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
        files.append(sqlite_path)
        descriptions = directory / "database_description"
        if descriptions.is_dir():
            files.extend(path for path in descriptions.glob("*.csv") if path.is_file())

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(databases_root).as_posix()):
        digest.update(path.relative_to(databases_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
