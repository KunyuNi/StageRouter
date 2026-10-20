"""Locked StageRouter configuration for the New2 Text-to-SQL run."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


NEW2_STATE_V26_EXPERIMENT_FAMILY = "new2_stage_router_paper_v1"
LOCKED_RUN_NAME = "state_v26_paper_tdmc_lam01_s1145_v1"
CONTROL_ROOT = f"cache/experiment_controls/{NEW2_STATE_V26_EXPERIMENT_FAMILY}"

LOCKED_CONFIG: dict[str, Any] = {
    "algorithm": "state_v26_scalar_double_dqn",
    "experiment_family": NEW2_STATE_V26_EXPERIMENT_FAMILY,
    "state_version": "state_v26",
    "state_dim": 5256,
    "hidden_dim": 512,
    "repr_dim": 128,
    "plm_model": os.getenv("NL2SQL_SEMANTIC_MODEL", "models/gte-large-en-v1-5"),
    "buffer_path": "cache/new2.json",
    "transition_db": "cache/new2.db",
    "dev_data_path": "data/dev/dev.json",
    "dev_db_root": "data/dev",
    "dev_transition_db": "exp/dev.db",
    "dev_trajectory_buffer_path": "exp/dev.json",
    "dev_semantic_cache_path": "exp/state_v25_semantics.db",
    "dev_exact_cache_contract": "exp/dev_exact_cache_contract.json",
    "dev_stage_cost_normalizer_path": "exp/stage_cost_normalization.json",
    "operator_pkg": "env.chess",
    "exact_replay_only": True,
    "initialization_bundle_path": f"{CONTROL_ROOT}/shared_initialization_v1.pt",
    "family_control_manifest_path": f"{CONTROL_ROOT}/family_control_v1.json",
    "reward_version": "terminal_quality_mean_buffer_cost_v1",
    "gamma": 1.0,
    "target_update_interval": 500,
    "epochs": 15,
    "batch_size": 64,
    "n_batches": 500,
    "lr": 1e-4,
    "grad_clip": 1.0,
    "seed": 1145,
    "lambda_cost": 0.1,
    "mc_weight": 0.3,
    "cost_context_enabled": False,
    "difficulty_context_enabled": False,
    "schedule_path": f"{CONTROL_ROOT}/td_schedule_v1.npz",
}

NEW2_STATE_V26_RUN_CONFIGS = {LOCKED_RUN_NAME: LOCKED_CONFIG}


def validate_new2_state_v26_config(config: Mapping[str, Any]) -> None:
    if dict(config) != LOCKED_CONFIG:
        raise ValueError("Only the locked State V26 New2 configuration is supported")
