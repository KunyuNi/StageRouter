"""Content identity for an offline SQLite transition cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


_COLUMNS = (
    "task_type", "question_id", "stage_idx", "tier", "input_hash",
    "operator_version", "semantic_hash", "output_payload", "stored_hash", "source",
)
_PRIMARY_KEY = ("question_id", "stage_idx", "tier", "input_hash", "operator_version")


def sqlite_transition_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    file_digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            file_digest.update(block)
    logical_digest = hashlib.sha256()
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        available = {str(row[1]) for row in connection.execute("PRAGMA table_info(transitions)")}
        if set(_COLUMNS) - available:
            raise ValueError("transition DB is missing logical columns")
        query = f"SELECT {','.join(_COLUMNS)} FROM transitions ORDER BY {','.join(_PRIMARY_KEY)}"
        count = 0
        for raw in connection.execute(query):
            payload = {name: raw[name] for name in _COLUMNS}
            try:
                payload["output_payload"] = json.loads(payload["output_payload"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("transition output_payload is not valid JSON") from exc
            logical_digest.update(json.dumps(payload, ensure_ascii=True, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False).encode("ascii"))
            logical_digest.update(b"\n")
            count += 1
    finally:
        connection.close()
    return {"source_file_sha256": file_digest.hexdigest(),
            "logical_sha256": logical_digest.hexdigest(), "rows": count,
            "integrity": integrity}
