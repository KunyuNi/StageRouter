"""Frozen train-buffer cost scales shared by state encoding and reward."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DECISION_STAGE_NAMES = ("S1", "S4", "S5", "S6", "S7", "S8")
GLOBAL_STAGE_COST_NORMALIZER_V1 = "global_train_p99_v1"
DEFAULT_PERCENTILE = 99.0


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_costs(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != len(DECISION_STAGE_NAMES):
        raise ValueError("decision_stage_costs must contain exactly 6 values")
    costs = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in costs):
        raise ValueError("decision_stage_costs must be finite and non-negative")
    return costs


@dataclass(frozen=True)
class GlobalStageCostNormalizer:
    version: str
    source_buffer_sha256: str
    source_records: int
    source_qids: int
    percentile: float
    stage_clips: tuple[float, ...]
    stage_stats: tuple[Mapping[str, float], ...]
    artifact_sha256: str

    def normalize(self, decision_stage_costs: Sequence[float]) -> tuple[float, ...]:
        costs = _validated_costs(decision_stage_costs)
        return tuple(
            min(cost / clip, 1.0)
            for cost, clip in zip(costs, self.stage_clips)
        )

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_buffer_sha256": self.source_buffer_sha256,
            "source_records": self.source_records,
            "source_qids": self.source_qids,
            "percentile": self.percentile,
            "stage_names": list(DECISION_STAGE_NAMES),
            "stage_clips": list(self.stage_clips),
            "stage_stats": [dict(stats) for stats in self.stage_stats],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def fit_global_stage_cost_normalizer(
    records: Sequence[Any],
    source_buffer_sha256: str,
    percentile: float = DEFAULT_PERCENTILE,
) -> GlobalStageCostNormalizer:
    if not records:
        raise ValueError("cannot fit stage cost normalizer from an empty buffer")
    if not source_buffer_sha256:
        raise ValueError("source_buffer_sha256 must not be empty")
    if not math.isfinite(percentile) or not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be finite and in (0, 100]")

    columns: list[list[float]] = [[] for _ in DECISION_STAGE_NAMES]
    qids: set[int] = set()
    for record in records:
        costs = _validated_costs(record.decision_stage_costs)
        qids.add(int(record.question_id))
        for stage_idx, cost in enumerate(costs):
            columns[stage_idx].append(cost)

    clips = tuple(float(np.percentile(values, percentile)) for values in columns)
    if any(not math.isfinite(value) or value <= 0.0 for value in clips):
        raise ValueError("every fitted stage cost clip must be finite and positive")

    stats: list[Mapping[str, float]] = []
    for values, clip in zip(columns, clips):
        stats.append(
            {
                "count": float(len(values)),
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50.0)),
                "p90": float(np.percentile(values, 90.0)),
                "p95": float(np.percentile(values, 95.0)),
                "p99": float(np.percentile(values, 99.0)),
                "p995": float(np.percentile(values, 99.5)),
                "max": float(max(values)),
                "saturation_rate": float(sum(value >= clip for value in values) / len(values)),
            }
        )

    normalizer = GlobalStageCostNormalizer(
        version=GLOBAL_STAGE_COST_NORMALIZER_V1,
        source_buffer_sha256=str(source_buffer_sha256),
        source_records=len(records),
        source_qids=len(qids),
        percentile=float(percentile),
        stage_clips=clips,
        stage_stats=tuple(stats),
        artifact_sha256="",
    )
    return replace(
        normalizer,
        artifact_sha256=_artifact_sha256(normalizer._payload_without_hash()),
    )


def load_global_stage_cost_normalizer(path: str | Path) -> GlobalStageCostNormalizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_hash = str(payload.pop("artifact_sha256", ""))
    actual_hash = _artifact_sha256(payload)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(
            "stage cost normalizer SHA256 mismatch: "
            f"expected={expected_hash or '<missing>'} actual={actual_hash}"
        )
    if tuple(payload.get("stage_names", ())) != DECISION_STAGE_NAMES:
        raise ValueError("stage cost normalizer has incompatible decision stage names")

    clips = tuple(float(value) for value in payload["stage_clips"])
    if len(clips) != len(DECISION_STAGE_NAMES) or any(
        not math.isfinite(value) or value <= 0.0 for value in clips
    ):
        raise ValueError("stage cost normalizer must contain 6 finite positive clips")

    return GlobalStageCostNormalizer(
        version=str(payload["version"]),
        source_buffer_sha256=str(payload["source_buffer_sha256"]),
        source_records=int(payload["source_records"]),
        source_qids=int(payload["source_qids"]),
        percentile=float(payload["percentile"]),
        stage_clips=clips,
        stage_stats=tuple(dict(stats) for stats in payload["stage_stats"]),
        artifact_sha256=expected_hash,
    )
