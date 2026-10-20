"""Atomic metadata-only trajectory buffer for ``RAG/cache/new1.json``."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .stage_mapping import route_key as make_route_key


@dataclass
class TrajectoryRecord:
    question_id: int
    split: str
    route_name: str
    route_key: str
    actions: list[int]
    f1_score: float
    reward: float
    answer: str
    ground_truth: str
    total_cost: float = 0.0
    stage_costs: list[float] = field(default_factory=list)
    decision_stage_costs: list[float] = field(default_factory=list)
    fixed_stage_costs: list[float] = field(default_factory=list)
    total_latency_ms: float = 0.0
    stage_latencies_ms: list[float] = field(default_factory=list)
    decision_stage_latencies_ms: list[float] = field(default_factory=list)
    fixed_stage_latencies_ms: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    operator_version: int = 1
    workflow_identity: str = ""
    index_identity: str = ""

    def __post_init__(self) -> None:
        self.question_id = int(self.question_id)
        self.actions = [int(action) for action in self.actions]
        self.operator_version = int(self.operator_version)
        self.route_key = str(self.route_key or make_route_key(self.actions))
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryRecord":
        data = dict(value)
        # Old/new route logs may keep identity in metadata; accept both forms.
        metadata = dict(data.get("metadata") or {})
        data.setdefault("operator_version", metadata.get("operator_version", 1))
        data.setdefault("workflow_identity", metadata.get("workflow_identity", ""))
        data.setdefault("index_identity", metadata.get("index_identity", ""))
        if not data.get("route_key"):
            data["route_key"] = make_route_key(data.get("actions", []))
        return cls(**data)


class TrajectoryBuffer:
    """In-memory index with atomic persistence and route-level upsert."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        workflow_identity: str = "",
        operator_version: int = 1,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.workflow_identity = str(workflow_identity)
        self.operator_version = int(operator_version)
        self.records: list[TrajectoryRecord] = []
        self._index: dict[tuple[str, int, str, int, str], int] = {}
        self._revision = 0
        self._saved_revision = 0
        self._lock = threading.RLock()
        if self.path is not None and self.path.exists():
            self.load()

    def _key(self, record: TrajectoryRecord) -> tuple[str, int, str, int, str]:
        metadata = record.metadata or {}
        operator_version = int(
            getattr(record, "operator_version", None)
            or metadata.get("operator_version", 1)
        )
        workflow_identity = str(
            getattr(record, "workflow_identity", None)
            or metadata.get("workflow_identity", "")
            or self.workflow_identity
        )
        return (
            str(record.split),
            int(record.question_id),
            str(record.route_key),
            operator_version,
            workflow_identity,
        )

    def _rebuild_index(self) -> None:
        self._index = {}
        for index, record in enumerate(self.records):
            self._index.setdefault(self._key(record), index)

    def upsert_by_question_route(self, record: TrajectoryRecord) -> bool:
        """Insert a record or replace the exact qid/route/version identity.

        Returns ``True`` when an existing record was replaced and ``False`` for
        a newly inserted record, matching NL2SQL's useful behavior.
        """

        if not isinstance(record, TrajectoryRecord):
            record = TrajectoryRecord.from_dict(record)
        if not record.workflow_identity and self.workflow_identity:
            record.workflow_identity = self.workflow_identity
        key = self._key(record)
        with self._lock:
            index = self._index.get(key)
            if index is not None:
                if self.records[index] != record:
                    self.records[index] = record
                    self._revision += 1
                return True
            self._index[key] = len(self.records)
            self.records.append(record)
            self._revision += 1
            return False

    def get_exact(
        self,
        split: str,
        question_id: int,
        route_key: str,
        operator_version: int,
        workflow_identity: str,
    ) -> TrajectoryRecord | None:
        """Return the exact split-aware route result, if materialized."""

        key = (
            str(split),
            int(question_id),
            str(route_key),
            int(operator_version),
            str(workflow_identity),
        )
        with self._lock:
            index = self._index.get(key)
            if index is None:
                return None
            return self.records[index]

    def remove_question_ids(self, question_ids: set[int] | list[int] | tuple[int, ...]) -> int:
        """Remove materialized trajectories for excluded queries."""

        excluded = {int(question_id) for question_id in question_ids}
        if not excluded:
            return 0
        with self._lock:
            before = len(self.records)
            self.records = [
                record
                for record in self.records
                if int(record.question_id) not in excluded
            ]
            self._rebuild_index()
            removed = before - len(self.records)
            if removed:
                self._revision += 1
            return removed

    def save(self, file_path: str | os.PathLike[str] | None = None) -> None:
        target = Path(file_path) if file_path is not None else self.path
        if target is None:
            raise ValueError("TrajectoryBuffer.save requires a file path")
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            same_target = self.path is not None and target == self.path.resolve()
            if (
                same_target
                and target.exists()
                and self._revision == self._saved_revision
            ):
                return
            data = [record.to_dict() for record in self.records]
            revision = self._revision
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        self.path = target
        with self._lock:
            if self._revision == revision:
                self._saved_revision = revision

    def load(self, file_path: str | os.PathLike[str] | None = None) -> int:
        target = Path(file_path) if file_path is not None else self.path
        if target is None:
            raise ValueError("TrajectoryBuffer.load requires a file path")
        if not target.exists():
            return 0
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("trajectory buffer must contain a JSON list")
        records = [TrajectoryRecord.from_dict(item) for item in data]
        with self._lock:
            self.records = records
            self._rebuild_index()
            self._revision = 0
            self._saved_revision = 0
        self.path = target
        return len(records)

    def __len__(self) -> int:
        with self._lock:
            return len(self.records)

    def __iter__(self):
        with self._lock:
            return iter(tuple(self.records))
