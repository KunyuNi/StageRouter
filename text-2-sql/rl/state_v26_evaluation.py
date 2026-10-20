"""Cache-only State V26 policy loading and rollout helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from config import get_run_config
from config.new2_state_v26 import LOCKED_RUN_NAME, NEW2_STATE_V26_EXPERIMENT_FAMILY
from rl.stage_mapping import decision_to_pipeline_stage_idx
from rl.state_v25_encoder import StateV25Encoder
from rl.state_v26_experiment_factory import build_state_v26_network
from rl.state_v26_policy import StateV26Policy


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_state_dict(state: Mapping[str, Any]) -> bool:
    return all(
        isinstance(value, torch.Tensor) and torch.isfinite(value).all().item()
        for value in state.values()
    )


def _action_mask(valid_actions: Sequence[Any]) -> np.ndarray:
    mask = np.zeros(4, dtype=bool)
    try:
        candidates = list(valid_actions)
    except TypeError as exc:
        raise ValueError(
            "State V26 evaluation received an invalid action mask"
        ) from exc
    for value in candidates:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("State V26 evaluation received an invalid action mask")
        try:
            action = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "State V26 evaluation received an invalid action mask"
            ) from exc
        if action != value or action < 0 or action >= 4:
            raise ValueError("State V26 evaluation received an invalid action mask")
        mask[action] = True
    if not mask.any():
        raise ValueError("State V26 evaluation received an invalid action mask")
    return mask


@dataclass
class StateV26EvaluationBundle:
    config: Mapping[str, Any]
    policy: StateV26Policy
    encoder: StateV25Encoder | Any
    checkpoint_sha256: str
    checkpoint_path: Path

    def close(self) -> None:
        self.encoder.v25.semantic_cache.close()

    def __enter__(self) -> "StateV26EvaluationBundle":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _family_manifest(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path = root / str(config["family_control_manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_family") != NEW2_STATE_V26_EXPERIMENT_FAMILY:
        raise ValueError("State V26 family manifest mismatch")
    if manifest.get("reward_version") != config["reward_version"] or manifest.get("run_names") != [LOCKED_RUN_NAME]:
        raise ValueError("State V26 paper run manifest mismatch")
    if manifest.get("buffer_path") != config["buffer_path"] or _sha256(root / str(config["buffer_path"])) != manifest.get("buffer_sha256"):
        raise ValueError("State V26 training buffer identity mismatch")
    if not math.isfinite(float(manifest.get("c_ref", 0))) or float(manifest["c_ref"]) <= 0:
        raise ValueError("State V26 cost reference is invalid")
    for relative, expected in manifest["artifact_sha256"].items():
        if _sha256(root / relative) != expected:
            raise ValueError(f"State V26 evaluation artifact drift: {relative}")
    return manifest, path


def load_state_v26_evaluation_bundle(
    *,
    project_root: str | Path,
    run_name: str,
    device: str,
    split: str,
    manager_factory: Callable[[str], Any],
    checkpoint_path: str | Path | None = None,
) -> StateV26EvaluationBundle:
    root = Path(project_root).resolve()
    config = get_run_config(run_name)
    if config.get("experiment_family") != NEW2_STATE_V26_EXPERIMENT_FAMILY:
        raise ValueError("State V26 evaluator received an unrelated run")
    manifest, manifest_path = _family_manifest(root, config)
    selected = (
        Path(checkpoint_path).resolve()
        if checkpoint_path is not None
        else (root / "model" / run_name / "ckpt_final.pt").resolve()
    )
    checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("algorithm") != "state_v26_scalar_double_dqn"
        or checkpoint.get("epoch") != 15
        or checkpoint.get("global_step") != 7500
    ):
        raise ValueError("State V26 final checkpoint progress/format mismatch")
    expected_objective = {
        "lambda_cost": float(config["lambda_cost"]),
        "c_ref": float(manifest["c_ref"]),
        "gamma": float(config["gamma"]),
        "mc_weight": float(config["mc_weight"]),
    }
    if checkpoint.get("objective") != expected_objective:
        raise ValueError("State V26 checkpoint reward or loss settings mismatch")
    identity = checkpoint.get("identity")
    expected_identity = {
        "run_name": run_name,
        "family_manifest_sha256": _sha256(manifest_path),
        "schedule_sha256": _sha256(root / str(config["schedule_path"])),
        "source_identity": manifest["source_identity"],
        "lambda_cost": float(config["lambda_cost"]),
        "c_ref": float(manifest["c_ref"]),
        "mc_weight": float(config["mc_weight"]),
        "reward_version": config["reward_version"],
        "feature_flags": {
            "cost_context": bool(config["cost_context_enabled"]),
            "difficulty_context": bool(config["difficulty_context_enabled"]),
        },
    }
    if identity != expected_identity:
        raise ValueError("State V26 checkpoint identity mismatch")
    online = checkpoint.get("online_state_dict")
    if not isinstance(online, Mapping) or not _finite_state_dict(online):
        raise ValueError("State V26 checkpoint online weights are invalid")
    network = build_state_v26_network(config)
    network.load_state_dict(online, strict=True)

    semantic_path = (
        root / manifest["base_semantic_db"]
        if split == "train"
        else root / str(config["dev_semantic_cache_path"])
    )
    db_root = root / ("data/train" if split == "train" else str(config["dev_db_root"]))
    v25 = StateV25Encoder(
        db_root=db_root,
        db_mode=split,
        plm_model_name=config["plm_model"],
        device=device,
        semantic_cache_path=semantic_path,
        semantic_cache_writable=split == "dev",
    )
    from rl.state_v26_encoder import StateV26Encoder, StateV26FeatureFlags

    if split not in ("train", "dev"):
        raise ValueError("State V26 evaluation split must be train or dev")
    encoder = StateV26Encoder(
        v25,
        None,
        StateV26FeatureFlags(
            cost_context=bool(config["cost_context_enabled"]),
            s4_difficulty_context=bool(config["difficulty_context_enabled"]),
        ),
    )
    return StateV26EvaluationBundle(
        config=config,
        policy=StateV26Policy(network, device),
        encoder=encoder,
        checkpoint_sha256=_sha256(selected),
        checkpoint_path=selected,
    )


def cache_only_policy_rollout(
    *,
    item: Mapping[str, Any],
    env: Any,
    transition_cache: Any,
    encoder: Any,
    policy: StateV26Policy,
    force_s4_action: int | None = None,
) -> dict[str, Any]:
    from env._shared.state_payload import CostTracker, StatePayload

    qid = int(item["question_id"])
    db_id = str(item.get("db_id", "") or "")
    payload = StatePayload(
        question_id=qid,
        db_id=db_id,
        question=str(item.get("question", "") or ""),
        evidence=str(item.get("evidence", "") or ""),
        gold_sql=str(item.get("SQL", "") or ""),
        cost_tracker=CostTracker(),
    )
    manager = env._get_db_manager(db_id)
    question_embedding = encoder.encode_question(payload.question, payload.evidence)
    actions: list[int] = []
    decision_costs: list[float] = []
    stage_costs = [0.0] * 8
    decisions: list[dict[str, Any]] = []
    total_cost = 0.0
    for stage_idx in range(6):
        mask = _action_mask(
            env.get_action_mask(stage_idx, payload, max(0.0, 10.0 - total_cost))
        )
        state = encoder.encode(
            payload,
            stage_idx,
            max(0.0, 10.0 - total_cost),
            actions,
            decision_costs,
            question_embedding,
            mask,
        )
        q_values = policy.q_values(state, mask)
        action = (
            int(force_s4_action)
            if stage_idx == 1 and force_s4_action is not None
            else policy.select_action(state, mask)
        )
        if action not in range(4) or not mask[action]:
            raise ValueError("State V26 policy selected an illegal action")
        if stage_idx == 1 and (not np.array_equal(mask, [1, 1, 1, 0]) or action == 3):
            raise ValueError("State V26 S4 must expose exactly T1/T2/T3")
        decisions.append(
            {
                "stage_idx": stage_idx,
                "action": action,
                "q_values": q_values.tolist(),
                "action_mask": mask.astype(int).tolist(),
            }
        )
        actions.append(action)
        physical = decision_to_pipeline_stage_idx(stage_idx)
        payload, raw_cost, _hit = env.run_physical_stage(
            stage_idx=physical,
            tier=action,
            payload=payload,
            cache=transition_cache,
            question_id=qid,
            db_manager=manager,
            operator_version=1,
            cache_only=True,
        )
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("State V26 evaluation observed an invalid stage cost")
        decision_costs.append(cost)
        stage_costs[physical] = cost
        total_cost += cost
        payload, fixed = env.run_fixed_stages_after_decision(
            decision_idx=stage_idx,
            payload=payload,
            cache=transition_cache,
            question_id=qid,
            db_manager=manager,
            operator_version=1,
            cache_only=True,
        )
        for fixed_stage, fixed_cost, _hit, _latency in fixed:
            value = float(fixed_cost)
            if not math.isfinite(value) or value < 0:
                raise ValueError("State V26 evaluation observed an invalid fixed cost")
            stage_costs[int(fixed_stage)] = value
            total_cost += value
    terminal = env.step_s9(payload, manager)
    final_ex = int(terminal["exec_res"])
    if final_ex not in (0, 1) or not math.isfinite(total_cost):
        raise ValueError("State V26 terminal evaluation is invalid")
    return {
        "question_id": qid,
        "db_id": db_id,
        "actions": actions,
        "final_ex": final_ex,
        "total_cost": total_cost,
        "decision_stage_costs": decision_costs,
        "stage_costs": stage_costs,
        "decisions": decisions,
        "status": "ok",
        "system_error": "",
    }
