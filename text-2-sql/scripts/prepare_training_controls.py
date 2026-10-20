#!/usr/bin/env python3
"""Prepare the replay schedule and seeded policy initialization."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from config.new2_state_v26 import (
    CONTROL_ROOT,
    LOCKED_CONFIG,
    LOCKED_RUN_NAME,
    NEW2_STATE_V26_EXPERIMENT_FAMILY,
)
from rl.state_v26_experiment_factory import (
    create_state_v26_initialization_bundle,
    state_v26_schema_identity,
)
from rl.state_v26_replay import build_cycle_schedule


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.preparing")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _save_schedule(path: Path, schedule: Any) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, route_indices=schedule.route_indices)
    _atomic_write(path, stream.getvalue())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--source_identity")
    parser.add_argument(
        "--output_root",
        default=CONTROL_ROOT,
    )
    parser.add_argument(
        "--base_root",
        default=f"{CONTROL_ROOT}/base",
    )
    parser.add_argument("--buffer", default="cache/new2.json")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    base = root / args.base_root
    required = (
        base / "train_semantics.db",
        base / "train_semantics.manifest.json",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("State V26 base preparation is incomplete")

    buffer_path = root / args.buffer
    records = json.loads(buffer_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("State V26 family requires a nonempty buffer")
    qids = sorted({int(record["question_id"]) for record in records})
    if len(qids) < 4:
        raise ValueError("State V26 family requires at least four requests")

    if int(LOCKED_CONFIG["batch_size"]) % 4:
        raise ValueError("batch size must be divisible by routes per request")
    formal_schedule = build_cycle_schedule(
        records,
        steps=int(LOCKED_CONFIG["epochs"]) * int(LOCKED_CONFIG["n_batches"]),
        qids_per_step=int(LOCKED_CONFIG["batch_size"]) // 4,
        routes_per_qid=4,
        seed=int(LOCKED_CONFIG["seed"]),
    )
    schedule_path = output / "td_schedule_v1.npz"
    _save_schedule(schedule_path, formal_schedule)
    _atomic_write(
        schedule_path.with_suffix(".manifest.json"),
        _canonical(
            {
                "format_version": 1,
                "name": "formal",
                "shape": list(formal_schedule.route_indices.shape),
                "canonical_sha256": formal_schedule.canonical_sha256,
                "file_sha256": _sha256(schedule_path),
            }
        ),
    )

    source_buffer_sha = _sha256(buffer_path)

    initialization = create_state_v26_initialization_bundle(seed=int(LOCKED_CONFIG["seed"]))
    init_path = output / "shared_initialization_v1.pt"
    torch.save(initialization, init_path)
    _atomic_write(
        output / "shared_initialization_v1.json",
        _canonical(
            {
                "format_version": 1,
                "seed": int(LOCKED_CONFIG["seed"]),
                "state_schema_hash": state_v26_schema_identity(),
                "file_sha256": _sha256(init_path),
                "tensor_sha256": initialization["tensor_sha256"],
            }
        ),
    )

    artifact_paths = [
        schedule_path,
        schedule_path.with_suffix(".manifest.json"),
        init_path,
        output / "shared_initialization_v1.json",
    ]
    route_costs = np.asarray([record["total_cost"] for record in records], dtype=np.float64)
    if not np.isfinite(route_costs).all() or np.any(route_costs < 0) or route_costs.mean() <= 0:
        raise ValueError("buffer route costs must be finite, non-negative, and have positive mean")
    family = {
        "format_version": 1,
        "experiment_family": NEW2_STATE_V26_EXPERIMENT_FAMILY,
        "source_identity": args.source_identity or source_buffer_sha[:40],
        "run_names": [LOCKED_RUN_NAME],
        "lambda_run_names": [],
        "buffer_path": args.buffer,
        "buffer_sha256": source_buffer_sha,
        "route_count": len(records),
        "qid_count": len(qids),
        "c_ref": float(route_costs.mean()),
        "reward_version": "terminal_quality_mean_buffer_cost_v1",
        "state_schema_hash": state_v26_schema_identity(),
        "base_semantic_db": str(
            (base / "train_semantics.db").relative_to(root)
        ).replace("\\", "/"),
        "artifact_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in artifact_paths
        },
        "static_validation_policy": "once_at_process_start",
    }
    _atomic_write(output / "family_control_v1.json", _canonical(family))
    print(_canonical(family).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
