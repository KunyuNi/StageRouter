"""Exact stage-transition cache for the MultiHop-RAG workflow.

Unlike the older NL2SQL cache this module intentionally has no semantic or
cross-tier lookup.  A tier is an action, so a hit is valid only when every
component of :class:`TransitionKey` matches exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .stage_mapping import PHYSICAL_STAGE_NAMES
from .workflow_identity import WorkflowIdentity

STAGE_NAMES = PHYSICAL_STAGE_NAMES
STAGES_IN_CACHE = tuple(range(len(PHYSICAL_STAGE_NAMES)))


class TransitionCacheIntegrityError(ValueError):
    """Raised when a stored transition payload fails integrity validation."""


@dataclass(frozen=True, init=False)
class TransitionKey:
    """Primary key: qid, physical stage, tier, semantic input hash, version."""

    question_id: int
    stage_idx: int
    tier: int
    input_hash: str
    operator_version: int

    def __init__(
        self,
        question_id: int,
        stage_idx: int | None = None,
        tier: int = 0,
        input_hash: str | None = None,
        operator_version: int = 1,
        *,
        physical_stage_idx: int | None = None,
        input_payload_hash: str | None = None,
    ) -> None:
        if stage_idx is None:
            stage_idx = physical_stage_idx
        if stage_idx is None:
            raise TypeError("TransitionKey requires stage_idx or physical_stage_idx")
        if input_hash is None:
            input_hash = input_payload_hash
        if input_hash is None:
            raise TypeError("TransitionKey requires input_hash")
        object.__setattr__(self, "question_id", int(question_id))
        object.__setattr__(self, "stage_idx", int(stage_idx))
        object.__setattr__(self, "tier", int(tier))
        object.__setattr__(self, "input_hash", str(input_hash))
        object.__setattr__(self, "operator_version", int(operator_version))
        if not 0 <= self.stage_idx < len(PHYSICAL_STAGE_NAMES):
            raise ValueError(f"invalid RAG physical stage index: {self.stage_idx}")
        if self.tier < 0:
            raise ValueError(f"tier must be non-negative, got {self.tier}")
        if self.operator_version < 1:
            raise ValueError("operator_version must be positive")

    @property
    def physical_stage_idx(self) -> int:
        return self.stage_idx

    @property
    def input_payload_hash(self) -> str:
        """Compatibility alias for code using NL2SQL's field name."""

        return self.input_hash


class TransitionCache:
    """Thread-safe SQLite cache stored in ``RAG/cache/new1.db``."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] = "RAG/cache/new1.db",
        workflow_identity: str | WorkflowIdentity = "default",
        readonly: bool = False,
    ) -> None:
        self.db_path = str(db_path)
        self.readonly = bool(readonly)
        self.workflow_identity = (
            workflow_identity.sha256
            if isinstance(workflow_identity, WorkflowIdentity)
            else str(workflow_identity)
        )
        self._lock = threading.RLock()
        self._closed = False
        if self.readonly:
            resolved = Path(self.db_path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"transition cache not found: {resolved}")
            self._conn = sqlite3.connect(
                resolved.as_uri() + "?mode=ro", uri=True, check_same_thread=False
            )
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.readonly:
            self._validate_readonly()
        else:
            self._initialize()

    def _validate_readonly(self) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cache_meta WHERE key='workflow_identity'"
            ).fetchone()
            existing = str(row["value"]) if row else ""
            if existing != self.workflow_identity:
                raise ValueError(
                    "workflow identity mismatch: "
                    f"cache={existing}, requested={self.workflow_identity}"
                )
            self._conn.execute("SELECT 1 FROM transitions LIMIT 1").fetchone()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    task_type TEXT NOT NULL DEFAULT 'multi-hop-rag',
                    question_id INTEGER NOT NULL,
                    stage_idx INTEGER NOT NULL,
                    tier INTEGER NOT NULL,
                    input_hash TEXT NOT NULL,
                    operator_version INTEGER NOT NULL,
                    output_payload TEXT NOT NULL,
                    stored_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
                    PRIMARY KEY (question_id, stage_idx, tier, input_hash,
                                operator_version)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transitions_stage "
                "ON transitions(stage_idx, tier, operator_version)"
            )
            row = self._conn.execute(
                "SELECT value FROM cache_meta WHERE key='workflow_identity'"
            ).fetchone()
            existing = str(row["value"]) if row else ""
            if existing and existing != self.workflow_identity:
                raise ValueError(
                    "workflow identity mismatch: "
                    f"cache={existing}, requested={self.workflow_identity}"
                )
            if not existing:
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache_meta(key,value) VALUES(?,?)",
                    ("workflow_identity", self.workflow_identity),
                )
            self._conn.commit()

    @staticmethod
    def _payload_json(payload: Mapping[str, Any]) -> tuple[str, str]:
        if not isinstance(payload, Mapping):
            raise TypeError("transition output_payload must be a mapping")
        encoded = json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def lookup(self, key: TransitionKey) -> dict[str, Any] | None:
        """Return an exact payload match, or ``None``."""

        with self._lock:
            if self._closed:
                raise RuntimeError("TransitionCache is closed")
            row = self._conn.execute(
                """SELECT output_payload, stored_hash FROM transitions
                   WHERE question_id=? AND stage_idx=? AND tier=?
                     AND input_hash=? AND operator_version=?""",
                (
                    key.question_id,
                    key.stage_idx,
                    key.tier,
                    key.input_hash,
                    key.operator_version,
                ),
            ).fetchone()
            if row is None:
                return None
            try:
                payload = json.loads(row["output_payload"])
                encoded, digest = self._payload_json(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise TransitionCacheIntegrityError(
                    "corrupt transition cache payload"
                ) from exc
            if digest != row["stored_hash"] or encoded != row["output_payload"]:
                raise TransitionCacheIntegrityError(
                    "corrupt transition cache payload"
                )
            return payload

    def __len__(self) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("TransitionCache is closed")
            row = self._conn.execute("SELECT COUNT(*) AS n FROM transitions").fetchone()
            return int(row["n"])

    def store(
        self,
        key: TransitionKey,
        output_payload: Mapping[str, Any],
        source: str = "unknown",
    ) -> None:
        if self.readonly:
            raise RuntimeError("transition cache is read-only")
        payload_json, stored_hash = self._payload_json(output_payload)
        with self._lock:
            if self._closed:
                raise RuntimeError("TransitionCache is closed")
            self._conn.execute(
                """INSERT OR REPLACE INTO transitions
                   (task_type, question_id, stage_idx, tier, input_hash,
                    operator_version, output_payload, stored_hash, source,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "multi-hop-rag",
                    key.question_id,
                    key.stage_idx,
                    key.tier,
                    key.input_hash,
                    key.operator_version,
                    payload_json,
                    stored_hash,
                    str(source),
                    time.time(),
                ),
            )
            self._conn.commit()

    def count(self, stage_idx: int | None = None, tier: int | None = None) -> int:
        with self._lock:
            query = "SELECT COUNT(*) FROM transitions"
            params: list[Any] = []
            clauses: list[str] = []
            if stage_idx is not None:
                clauses.append("stage_idx=?")
                params.append(int(stage_idx))
            if tier is not None:
                clauses.append("tier=?")
                params.append(int(tier))
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            row = self._conn.execute(query, params).fetchone()
            return int(row[0])

    def integrity_check(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA integrity_check").fetchone()
            return str(row[0]) if row else ""

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "TransitionCache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
