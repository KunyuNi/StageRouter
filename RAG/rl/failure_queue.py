"""Atomic unresolved/resolved queue for incomplete RAG route pairs."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


QUEUE_VERSION = "rag_failed_pair_queue_v1"
EVALUATION_QUEUE_VERSION = "rag_policy_failure_queue_v2"


class FailedPairQueue:
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        workflow_identity: str,
        operator_version: int,
    ) -> None:
        self.path = Path(path)
        self.workflow_identity = str(workflow_identity)
        self.operator_version = int(operator_version)
        self._lock = threading.RLock()
        if self.path.exists():
            self._read()

    def _empty(self) -> dict[str, Any]:
        return {
            "version": QUEUE_VERSION,
            "workflow_identity": self.workflow_identity,
            "operator_version": self.operator_version,
            "split": "train",
            "unresolved": [],
            "resolved": [],
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("version") != QUEUE_VERSION
            or not isinstance(payload.get("unresolved"), list)
            or not isinstance(payload.get("resolved"), list)
        ):
            raise ValueError(f"invalid RAG failed-pair queue: {self.path}")
        if str(payload.get("workflow_identity", "")) != self.workflow_identity:
            raise ValueError("failed-pair queue workflow identity mismatch")
        if int(payload.get("operator_version", -1)) != self.operator_version:
            raise ValueError("failed-pair queue operator version mismatch")
        return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _identity(record: Mapping[str, Any]) -> tuple[str, int, str, int, str]:
        return (
            str(record.get("split", "train")),
            int(record.get("question_id", record.get("qid"))),
            str(record["route_key"]),
            int(record["operator_version"]),
            str(record["workflow_identity"]),
        )

    def record(self, route_record: Mapping[str, Any]) -> None:
        identity = self._identity(route_record)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            payload = self._read()
            prior = next(
                (
                    item
                    for item in payload["unresolved"]
                    if self._identity(item) == identity
                ),
                None,
            )
            payload["unresolved"] = [
                item
                for item in payload["unresolved"]
                if self._identity(item) != identity
            ]
            payload["resolved"] = [
                item
                for item in payload["resolved"]
                if self._identity(item) != identity
            ]
            record = {
                "split": identity[0],
                "question_id": identity[1],
                "route_name": str(route_record.get("route_name", "unknown")),
                "route_key": identity[2],
                "actions": [int(value) for value in route_record.get("actions", [])],
                "operator_version": identity[3],
                "workflow_identity": identity[4],
                "stage_index": route_record.get("failed_stage_idx"),
                "stage_name": route_record.get("failed_stage_name"),
                "decision_stage_index": route_record.get("failed_decision_stage_idx"),
                "decision_action": route_record.get("failed_decision_action"),
                "tier": route_record.get("failed_tier"),
                "tier_name": route_record.get("failed_tier_name"),
                "error_category": str(
                    route_record.get("error_category", "pair_exception")
                ),
                "error_type": str(route_record.get("error_type", "Exception")),
                "cause_type": str(
                    route_record.get(
                        "cause_type", route_record.get("error_type", "Exception")
                    )
                ),
                "error_message": str(route_record.get("error", "")),
                "error_fingerprint": str(
                    route_record.get("error_fingerprint", "")
                ),
                "attempts": int(prior.get("attempts", 0)) + 1 if prior else 1,
                "first_failed_at": prior.get("first_failed_at", now) if prior else now,
                "last_failed_at": now,
            }
            payload["unresolved"].append(record)
            payload["unresolved"].sort(key=self._identity)
            self._write(payload)

    def resolve(self, route_record: Mapping[str, Any], *, resolution: str) -> None:
        identity = self._identity(route_record)
        with self._lock:
            payload = self._read()
            moved = [
                item
                for item in payload["unresolved"]
                if self._identity(item) == identity
            ]
            if not moved:
                return
            payload["unresolved"] = [
                item
                for item in payload["unresolved"]
                if self._identity(item) != identity
            ]
            resolved = {
                self._identity(item): item for item in payload["resolved"]
            }
            now = datetime.now(timezone.utc).isoformat()
            for item in moved:
                resolved[identity] = {
                    **item,
                    "resolution": str(resolution),
                    "resolved_at": now,
                }
            payload["resolved"] = [resolved[key] for key in sorted(resolved)]
            self._write(payload)

    def exclude_question(self, question_id: int) -> None:
        qid = int(question_id)
        with self._lock:
            payload = self._read()
            identities = [
                self._identity(item)
                for item in payload["unresolved"]
                if int(item["question_id"]) == qid
            ]
            if not identities:
                return
            resolved = {
                self._identity(item): item for item in payload["resolved"]
            }
            now = datetime.now(timezone.utc).isoformat()
            for item in payload["unresolved"]:
                identity = self._identity(item)
                if identity in identities:
                    resolved[identity] = {
                        **item,
                        "resolution": "excluded_content_safety",
                        "resolved_at": now,
                    }
            payload["unresolved"] = [
                item
                for item in payload["unresolved"]
                if self._identity(item) not in identities
            ]
            payload["resolved"] = [resolved[key] for key in sorted(resolved)]
            self._write(payload)

    def counts(self) -> tuple[int, int]:
        with self._lock:
            payload = self._read()
            return len(payload["unresolved"]), len(payload["resolved"])


class EvaluationFailureQueue:
    """Shared Dev/Test policy failure queue with safe v1 migration."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        split: str,
        policy_id: str,
        workflow_identity: str = "",
        operator_version: int = 1,
    ) -> None:
        self.path = Path(path)
        self.split = str(split)
        self.policy_id = str(policy_id)
        self.workflow_identity = str(workflow_identity)
        self.operator_version = int(operator_version)
        self._lock = threading.RLock()
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("version") == QUEUE_VERSION:
                self._migrate_v1(raw)
            else:
                self._validate(raw)

    @staticmethod
    def _identity(record: Mapping[str, Any]) -> tuple[str, int, str, int, str]:
        return (
            str(record["split"]),
            int(record.get("question_id", record.get("qid"))),
            str(record["policy_id"]),
            int(record["operator_version"]),
            str(record["workflow_identity"]),
        )

    def _empty(self) -> dict[str, Any]:
        return {
            "version": EVALUATION_QUEUE_VERSION,
            "split": self.split,
            "workflow_identity": self.workflow_identity,
            "operator_version": self.operator_version,
            "unresolved": [],
            "resolved": [],
        }

    def _validate(self, payload: Mapping[str, Any]) -> None:
        if payload.get("version") != EVALUATION_QUEUE_VERSION:
            raise ValueError(f"invalid RAG policy failure queue: {self.path}")
        if str(payload.get("split")) != self.split:
            raise ValueError("policy failure queue split mismatch")
        if not isinstance(payload.get("unresolved"), list) or not isinstance(
            payload.get("resolved"), list
        ):
            raise ValueError("policy failure queue rows must be lists")
        existing_workflow = str(payload.get("workflow_identity", ""))
        if self.workflow_identity and existing_workflow != self.workflow_identity:
            raise ValueError("policy failure queue workflow identity mismatch")
        existing_version = int(payload.get("operator_version", -1))
        if existing_version != self.operator_version:
            raise ValueError("policy failure queue operator version mismatch")
        if not self.workflow_identity:
            self.workflow_identity = existing_workflow

    def _migrate_v1(self, payload: Mapping[str, Any]) -> None:
        unresolved = list(payload.get("unresolved") or [])
        if unresolved:
            raise ValueError(
                "resolve legacy unresolved rows before migrating Dev failure queue"
            )
        resolved = []
        for raw in payload.get("resolved") or []:
            row = dict(raw)
            row["split"] = self.split
            row["policy_id"] = "baseline"
            row["operator_version"] = int(
                row.get("operator_version", payload.get("operator_version", 1))
            )
            row["workflow_identity"] = str(
                row.get(
                    "workflow_identity", payload.get("workflow_identity", "")
                )
            )
            resolved.append(row)
        if not self.workflow_identity:
            self.workflow_identity = str(payload.get("workflow_identity", ""))
        self.operator_version = int(payload.get("operator_version", 1))
        migrated = self._empty()
        migrated["resolved"] = resolved
        self._write(migrated)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._validate(payload)
        return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @property
    def payload(self) -> dict[str, Any]:
        with self._lock:
            return self._read()

    @property
    def unresolved_count(self) -> int:
        return self.counts()[0]

    def record_failure(self, record: Mapping[str, Any]) -> None:
        from .error_policy import pair_error_category, pair_error_fingerprint

        error = record.get("error", "")
        row = {
            "split": self.split,
            "question_id": int(record.get("question_id", record.get("qid"))),
            "policy_id": str(record.get("policy_id", self.policy_id)),
            "operator_version": int(
                record.get("operator_version", self.operator_version)
            ),
            "workflow_identity": str(
                record.get("workflow_identity", self.workflow_identity)
            ),
            "selected_prefix": [
                int(value) for value in record.get("selected_prefix", [])
            ],
            "route_key": str(record.get("route_key", "")),
            "stage_index": record.get("stage_index"),
            "stage_name": record.get("stage_name"),
            "decision_stage_index": record.get("decision_stage_index"),
            "decision_action": record.get("decision_action"),
            "tier": record.get("tier"),
            "tier_name": record.get("tier_name"),
            "error_category": str(
                record.get("error_category") or pair_error_category(error)
            ),
            "error_type": str(record.get("error_type") or type(error).__name__),
            "error_message_digest": pair_error_fingerprint(
                error,
                error_type=str(record.get("error_type") or type(error).__name__),
                stage_index=record.get("stage_index"),
            ),
        }
        identity = self._identity(row)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            payload = self._read()
            prior = next(
                (
                    item
                    for item in payload["unresolved"]
                    if self._identity(item) == identity
                ),
                None,
            )
            payload["unresolved"] = [
                item
                for item in payload["unresolved"]
                if self._identity(item) != identity
            ]
            row["attempts"] = int(prior.get("attempts", 0)) + 1 if prior else 1
            row["first_failed_at"] = prior.get("first_failed_at", now) if prior else now
            row["last_failed_at"] = now
            payload["unresolved"].append(row)
            payload["unresolved"].sort(key=self._identity)
            self._write(payload)

    def resolve(self, question_id: int, *, resolution: str = "completed") -> None:
        target = (
            self.split,
            int(question_id),
            self.policy_id,
            self.operator_version,
            self.workflow_identity,
        )
        with self._lock:
            payload = self._read()
            moved = [
                item
                for item in payload["unresolved"]
                if self._identity(item) == target
            ]
            if not moved:
                return
            payload["unresolved"] = [
                item
                for item in payload["unresolved"]
                if self._identity(item) != target
            ]
            resolved = {self._identity(item): item for item in payload["resolved"]}
            now = datetime.now(timezone.utc).isoformat()
            for item in moved:
                resolved[target] = {
                    **item,
                    "resolution": str(resolution),
                    "resolved_at": now,
                }
            payload["resolved"] = [resolved[key] for key in sorted(resolved)]
            self._write(payload)

    def exclude_question(self, question_id: int) -> None:
        """Move a policy's safety failure out of unresolved without retrying it."""

        target = (
            self.split,
            int(question_id),
            self.policy_id,
            self.operator_version,
            self.workflow_identity,
        )
        with self._lock:
            payload = self._read()
            moved = [
                item for item in payload["unresolved"] if self._identity(item) == target
            ]
            if not moved:
                return
            payload["unresolved"] = [
                item for item in payload["unresolved"] if self._identity(item) != target
            ]
            resolved = {self._identity(item): item for item in payload["resolved"]}
            now = datetime.now(timezone.utc).isoformat()
            for item in moved:
                resolved[target] = {
                    **item,
                    "resolution": "excluded_content_safety",
                    "resolved_at": now,
                }
            payload["resolved"] = [resolved[key] for key in sorted(resolved)]
            self._write(payload)

    def counts(self) -> tuple[int, int]:
        payload = self.payload
        return len(payload["unresolved"]), len(payload["resolved"])


__all__ = [
    "EVALUATION_QUEUE_VERSION",
    "EvaluationFailureQueue",
    "FailedPairQueue",
    "QUEUE_VERSION",
]
