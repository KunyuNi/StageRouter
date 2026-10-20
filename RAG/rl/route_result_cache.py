"""Append-only JSONL cache for resumable route results."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def route_key(actions: Sequence[int]) -> str:
    values = [int(action) for action in actions]
    if len(values) != 4:
        raise ValueError(f"RAG route requires 4 actions, got {len(values)}")
    if any(value not in (0, 1, 2, 3) for value in values):
        raise ValueError(f"RAG actions must be in 0..3, got {values!r}")
    return "-".join(str(value) for value in values)


def _record_key(record: Mapping[str, Any]) -> tuple[str, int, str, int, str]:
    qid = record.get("qid", record.get("question_id"))
    if qid is None:
        raise KeyError("qid")
    if "route_key" not in record:
        actions = record.get("actions")
        if actions is None:
            raise KeyError("route_key")
        route = route_key(actions)
    else:
        route = str(record["route_key"])
    if "operator_version" not in record:
        raise KeyError("operator_version")
    if "workflow_identity" not in record:
        raise KeyError("workflow_identity")
    return (
        str(record.get("split", "")),
        int(qid),
        route,
        int(record["operator_version"]),
        str(record["workflow_identity"]),
    )


class RouteResultCache:
    """Index route JSONL records by split/qid/route/version/workflow identity."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._index: dict[tuple[str, int, str, int, str], dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _parse(raw: bytes) -> dict[tuple[str, int, str, int, str], dict[str, Any]]:
        if not raw:
            return {}
        if not raw.endswith(b"\n"):
            raise ValueError("route result cache has an incomplete final line")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("route result cache is not valid UTF-8") from exc
        index: dict[tuple[str, int, str, int, str], dict[str, Any]] = {}
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"route result cache line {line_number} is blank")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"route result cache line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"route result cache line {line_number} must be an object")
            record = dict(value)
            try:
                index[_record_key(record)] = record
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"route result cache line {line_number} has an invalid cache key"
                ) from exc
        return index

    def _load(self) -> None:
        if self.path.exists():
            self._index = self._parse(self.path.read_bytes())

    def get(
        self,
        *,
        split: str,
        qid: int | None = None,
        route_key: str,
        operator_version: int,
        workflow_identity: str,
        question_id: int | None = None,
    ) -> dict[str, Any] | None:
        if qid is None:
            qid = question_id
        if qid is None:
            raise TypeError("RouteResultCache.get requires qid")
        return self._index.get(
            (str(split), int(qid), str(route_key), int(operator_version), str(workflow_identity))
        )

    def append_many(self, records: Sequence[Mapping[str, Any]]) -> None:
        items: list[dict[str, Any]] = []
        keys: list[tuple[str, int, str, int, str]] = []
        for record in records:
            item = dict(record)
            if "qid" not in item and "question_id" in item:
                item["qid"] = item["question_id"]
            keys.append(_record_key(item))
            items.append(item)
        if not items:
            return

        encoded = b"".join(
            (json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n").encode(
                "utf-8"
            )
            for item in items
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.read_bytes() if self.path.exists() else b""
        next_index = self._parse(existing)
        for key, item in zip(keys, items):
            next_index[key] = item

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(existing)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        self._index = next_index

    def append(self, record: Mapping[str, Any]) -> None:
        self.append_many([record])

    def __len__(self) -> int:
        return len(self._index)
