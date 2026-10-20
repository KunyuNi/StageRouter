"""Submission configuration: scalar Stage-FiLM Dueling Double DQN with TD+MC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    arm: str
    run_name: str
    q_formulation: str
    mc_enabled: bool
    pairwise_enabled: bool
    head_type: str = "dueling"


EXPERIMENTS = {
    "S1": ExperimentConfig("S1", "rag_state_v1_scalar_td_mc", "scalar", True, False),
}


@dataclass(frozen=True)
class TrainerConfig:
    seed: int = 1145
    total_steps: int = 7500
    logging_epoch_steps: int = 500
    checkpoint_steps: int = 500
    qids_per_step: int = 16
    routes_per_qid: int = 4
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    gamma: float = 1.0
    mc_weight: float = 0.25
    target_update_steps: int = 500


def run_id(experiment: ExperimentConfig, seed: int) -> str:
    return f"{experiment.run_name}_s{int(seed)}"


__all__ = ["EXPERIMENTS", "ExperimentConfig", "TrainerConfig", "run_id"]
