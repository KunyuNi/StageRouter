#!/usr/bin/env python3
"""Train the locked routing policy from the offline buffer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parent.parent
ENV_ROOT = SOURCE_ROOT / "env"
for path in (SOURCE_ROOT, ENV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import get_run_config
from config.new2_state_v26 import (
    LOCKED_RUN_NAME,
    NEW2_STATE_V26_EXPERIMENT_FAMILY,
)
from rl.dataset_utils import load_question_records, question_map_from_records
from rl.env_wrapper import EnvironmentWrapper
from rl.state_v25_encoder import StateV25Encoder
from rl.state_v26_encoder import StateV26Encoder, StateV26FeatureFlags
from rl.state_v26_experiment_factory import (
    build_state_v26_network,
    load_state_v26_initialization,
)
from rl.state_v26_objective import StateV26ObjectiveConfig
from rl.state_v26_replay import (
    CachedStateV26RouteMaterializer,
    FrozenCycleSchedule,
    StateV26ReplayBuilder,
)
from rl.state_v26_trainer import OfflineStateV26Trainer
from rl.trajectory_buffer import TrajectoryBuffer
from rl.offline_runtime import ReadOnlyTransitionCache, offline_execution_guard


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_identity(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    manifest_path = root / str(config["family_control_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_family") != NEW2_STATE_V26_EXPERIMENT_FAMILY:
        raise ValueError("State V26 family manifest mismatch")
    if manifest.get("run_names") != [LOCKED_RUN_NAME]:
        raise ValueError("State V26 family manifest does not match the locked run")
    for relative, expected in manifest["artifact_sha256"].items():
        path = root / relative
        if _sha256(path) != expected:
            raise ValueError(f"State V26 startup artifact drift: {relative}")
    return manifest, manifest_path


def run_training(
    *,
    run_name: str,
    device: str,
    project_root: str | Path,
    checkpoint_dir: str | Path | None,
    smoke_steps: int | None,
) -> dict[str, Any]:
    config = get_run_config(run_name)
    if config["experiment_family"] != NEW2_STATE_V26_EXPERIMENT_FAMILY:
        raise ValueError("train_router received a non-State-V26 run")
    if smoke_steps is not None:
        if smoke_steps <= 0 or checkpoint_dir is None:
            raise ValueError("smoke runs require positive steps and a separate checkpoint directory")
    root = Path(project_root).resolve()
    manifest, manifest_path = _load_identity(root, config)
    if manifest.get("reward_version") != config["reward_version"]:
        raise ValueError("training manifest reward version mismatch")
    output = (
        root / str(checkpoint_dir)
        if checkpoint_dir is not None
        else root / "model" / run_name
    ).resolve()
    if smoke_steps is not None and smoke_steps > int(config["epochs"]) * int(config["n_batches"]):
        raise ValueError("smoke steps exceed the prepared replay schedule")
    output.mkdir(parents=True, exist_ok=False)

    buffer = TrajectoryBuffer(capacity=200000)
    buffer.load(str(root / str(config["buffer_path"])))
    route_costs = np.asarray([record.total_cost for record in buffer.records], dtype=np.float64)
    if route_costs.size == 0 or not np.isfinite(route_costs).all() or np.any(route_costs < 0):
        raise ValueError("training buffer must contain finite non-negative route costs")
    c_ref = float(route_costs.mean())
    if c_ref <= 0:
        raise ValueError("mean training-buffer route cost must be positive")
    if c_ref != float(manifest["c_ref"]) or _sha256(root / str(config["buffer_path"])) != manifest["buffer_sha256"]:
        raise ValueError("training buffer or mean cost differs from the prepared controls")
    questions = load_question_records(root / "data/train/train.json")
    question_map = question_map_from_records(questions)
    os.environ["NL2SQL_DB_ROOT"] = str((root / "data/train").resolve())
    os.environ["NL2SQL_DB_MODE"] = "train"
    from env.runner.database_manager import DatabaseManager

    env = EnvironmentWrapper(
        operator_pkg="env.chess",
        db_manager_factory=lambda db_id: DatabaseManager(
            db_mode="train", db_id=db_id
        ),
        exact_replay_only=True,
    )
    env.set_question_map(question_map)
    cache = ReadOnlyTransitionCache(root / str(config["transition_db"]))

    schedule_path = root / str(config["schedule_path"])
    with np.load(schedule_path) as payload:
        schedule_matrix = np.asarray(payload["route_indices"], dtype=np.int32)
    if schedule_matrix.shape != (int(config["epochs"]) * int(config["n_batches"]), int(config["batch_size"])):
        raise ValueError("replay schedule does not match configured training steps or batch size")
    schedule = FrozenCycleSchedule(
        schedule_matrix,
        hashlib.sha256(schedule_matrix.astype("<i4").tobytes()).hexdigest(),
    )
    init_bundle = torch.load(
        root / str(config["initialization_bundle_path"]),
        map_location="cpu",
        weights_only=False,
    )
    mc_weight = float(config["mc_weight"])

    try:
        v25 = StateV25Encoder(
            db_root=root / "data/train",
            db_mode="train",
            plm_model_name=config["plm_model"],
            device=device,
            semantic_cache_path=root / manifest["base_semantic_db"],
            semantic_cache_writable=False,
        )
        encoder = StateV26Encoder(
            v25,
            None,
            StateV26FeatureFlags(
                cost_context=bool(config["cost_context_enabled"]),
                s4_difficulty_context=bool(config["difficulty_context_enabled"]),
            ),
        )
        materializer = CachedStateV26RouteMaterializer(
            env,
            cache,
            encoder,
            lambda_cost=float(config["lambda_cost"]),
            c_ref=c_ref,
        )
        builder = StateV26ReplayBuilder(
            buffer.records, schedule, materializer
        )
        online = build_state_v26_network(config).to(device)
        target = build_state_v26_network(config).to(device)
        load_state_v26_initialization(online, init_bundle)
        load_state_v26_initialization(target, init_bundle)
        optimizer = torch.optim.Adam(online.parameters(), lr=float(config["lr"]))
        identity = {
            "run_name": run_name,
            "family_manifest_sha256": _sha256(manifest_path),
            "schedule_sha256": _sha256(schedule_path),
            "source_identity": manifest["source_identity"],
            "lambda_cost": float(config["lambda_cost"]),
            "c_ref": c_ref,
            "reward_version": "terminal_quality_mean_buffer_cost_v1",
            "feature_flags": {
                "cost_context": bool(config["cost_context_enabled"]),
                "difficulty_context": bool(config["difficulty_context_enabled"]),
            },
        }
        identity["mc_weight"] = mc_weight
        trainer = OfflineStateV26Trainer(
            online,
            target,
            optimizer,
            StateV26ObjectiveConfig(
                lambda_cost=float(config["lambda_cost"]),
                c_ref=c_ref,
                gamma=float(config["gamma"]),
                mc_weight=mc_weight,
            ),
            target_update_interval=int(config["target_update_interval"]),
            grad_clip=float(config["grad_clip"]),
            identity=identity,
        )
        gradient_ratios: dict[str, float] = {}
        if mc_weight > 0:
            with offline_execution_guard():
                calibration_batch = builder.sample_batch(0)
                gradient_ratios = trainer.auxiliary_gradient_ratios(calibration_batch)
            mc_ratio = gradient_ratios.get("mc_to_td")
            if mc_weight > 0:
                if mc_ratio is None or not math.isfinite(mc_ratio) or mc_ratio <= 0:
                    raise ValueError(
                        f"State V26 MC gradient ratio is invalid: {mc_ratio}"
                    )
            (output / "gradient_ratio_preflight.json").write_text(
                json.dumps(
                    {
                        "run_name": run_name,
                        "schedule_step": 0,
                        "mc_weight": mc_weight,
                        **gradient_ratios,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )

        total_steps = smoke_steps if smoke_steps is not None else int(config["epochs"]) * int(config["n_batches"])
        metric_path = output / ("smoke_metrics.jsonl" if smoke_steps else "train_metrics.jsonl")
        metric_stream = metric_path.open("w", encoding="utf-8")
        try:
            with offline_execution_guard():
                for step in range(total_steps):
                    batch = builder.sample_batch(step)
                    metrics = trainer.train_batch(batch)
                    metric_stream.write(json.dumps(metrics, sort_keys=True) + "\n")
                    if (step + 1) % int(config["n_batches"]) == 0:
                        trainer.epoch += 1
                        trainer.save_checkpoint(
                            output / f"ckpt_epoch_{trainer.epoch:03d}.pt"
                        )
                    if (step + 1) % 20 == 0:
                        metric_stream.flush()
                        print(
                            f"{run_name} step={step + 1}/{total_steps} "
                            f"loss={metrics['loss']:.6f}",
                            flush=True,
                        )
        finally:
            metric_stream.close()
        final_checkpoint = output / (
            "smoke_checkpoint.pt" if smoke_steps is not None else "ckpt_final.pt"
        )
        trainer.save_checkpoint(final_checkpoint)
        trainer.load_checkpoint(final_checkpoint)
        summary = {
            "run_name": run_name,
            "execution_mode": "smoke" if smoke_steps else "formal",
            "global_step": trainer.global_step,
            "epoch": trainer.epoch,
            "final_checkpoint": str(final_checkpoint),
            "finite_metrics": True,
            "mc_weight": mc_weight,
            "c_ref": c_ref,
            "reward_version": "terminal_quality_mean_buffer_cost_v1",
            "gradient_ratios": gradient_ratios,
            "s4_skip_legal": False,
            "api_call_count": 0,
            "cache_write_count": 0,
            "semantic_write_count": 0,
        }
        if "sensitivity_parameter" in config:
            summary["sensitivity_parameter"] = config["sensitivity_parameter"]
            summary["sensitivity_value"] = float(config["sensitivity_value"])
        (output / ("smoke_report.json" if smoke_steps else "training_summary.json")).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary
    finally:
        cache.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--smoke_steps", type=int)
    args = parser.parse_args(argv)
    run_training(
        run_name=args.run_name,
        device=args.device,
        project_root=args.project_root,
        checkpoint_dir=args.checkpoint_dir,
        smoke_steps=args.smoke_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
