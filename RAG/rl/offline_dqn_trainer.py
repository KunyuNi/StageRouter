"""Deterministic offline trainer for scalar Stage-FiLM Dueling Double DQN + MC."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .dqn_experiments import ExperimentConfig, TrainerConfig
from .dqn_networks import ScalarStageFiLMDuelingDQN
from .dqn_objective import scalar_double_dqn_targets
from .dqn_replay import OfflineReplayDataset, OfflineTransition
from .state_v1_schema import STATE_V1_SCHEMA


def component_seed(seed: int, family: str, component_path: str) -> int:
    digest = hashlib.sha256(
        f"{int(seed)}:{family}:{component_path}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _initialize_components(
    model: torch.nn.Module, *, seed: int, family: str
) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for name, module in model.named_modules():
        if not name or not hasattr(module, "reset_parameters"):
            continue
        value = component_seed(seed, family, name)
        seeds[name] = value
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(value)
            module.reset_parameters()
    # FiLM must remain an exact identity after generic Linear initialization.
    torch.nn.init.zeros_(model.task_film[-1].weight)
    torch.nn.init.zeros_(model.task_film[-1].bias)
    torch.nn.init.zeros_(model.workspace_film[-1].weight)
    torch.nn.init.zeros_(model.workspace_film[-1].bias)
    return seeds


class OfflineRAGDQNTrainer:
    def __init__(
        self,
        *,
        experiment: ExperimentConfig,
        replay: OfflineReplayDataset,
        pair_sampler: None = None,
        stage_embeddings: torch.Tensor,
        output_dir: str | Path,
        config: TrainerConfig = TrainerConfig(),
        identities: Mapping[str, str],
        scalar_lambda_cost: float = 0.01,
        scalar_c_ref: float = 0.00333879,
        fusion_hidden_dim: int = 256,
        device: str | torch.device = "cpu",
    ) -> None:
        self.experiment = experiment
        self.replay = replay
        if experiment.arm != "S1" or experiment.q_formulation != "scalar" or not experiment.mc_enabled or experiment.pairwise_enabled or experiment.head_type != "dueling":
            raise ValueError("submission trainer only supports S1 scalar Dueling TD+MC")
        if pair_sampler is not None:
            raise ValueError("pairwise sampling is not part of the submission method")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.mc_weight = float(config.mc_weight)
        if not math.isfinite(self.mc_weight) or self.mc_weight < 0:
            raise ValueError("mc_weight must be finite and nonnegative")
        self.scalar_lambda_cost = float(scalar_lambda_cost)
        self.scalar_c_ref = float(scalar_c_ref)
        if self.scalar_lambda_cost < 0.0:
            raise ValueError("scalar_lambda_cost must be nonnegative")
        if self.scalar_c_ref <= 0.0:
            raise ValueError("scalar_c_ref must be positive")
        if fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be positive")
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.identities = {str(name): str(value) for name, value in identities.items()}
        self.device = torch.device(device)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.model = ScalarStageFiLMDuelingDQN(
            STATE_V1_SCHEMA,
            stage_embeddings,
            fusion_hidden_dim=self.fusion_hidden_dim,
        )
        self.component_seeds = _initialize_components(
            self.model, seed=config.seed, family="scalar"
        )
        self.model.to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        self.global_step = 0
        self.logging_epoch = 0
        self.metrics: list[dict[str, float]] = []
        self.last_batch_identity = ""

    def _batch_tensors(self, rows: tuple[OfflineTransition, ...]) -> dict[str, torch.Tensor]:
        states = torch.from_numpy(np.stack([row.state.vector for row in rows])).to(
            self.device
        )
        legal = torch.from_numpy(
            np.stack([row.state.legal_action_mask for row in rows])
        ).to(self.device)
        actions = torch.tensor([row.action for row in rows], device=self.device)
        terminal = torch.tensor(
            [row.terminal for row in rows], dtype=torch.bool, device=self.device
        )
        next_vectors = np.stack(
            [
                row.next_state.vector
                if row.next_state is not None
                else np.zeros_like(row.state.vector)
                for row in rows
            ]
        )
        next_legal_values = np.stack(
            [
                row.next_state.legal_action_mask
                if row.next_state is not None
                else row.state.legal_action_mask
                for row in rows
            ]
        )
        next_states = torch.from_numpy(next_vectors).to(self.device)
        next_legal = torch.from_numpy(next_legal_values).to(self.device)
        cost_slice = STATE_V1_SCHEMA.slices["cost_history"]
        next_cost_block = next_states[:, cost_slice].reshape(-1, 17)
        # State stores candidate costs as SKIP,T1,T2,T3; Q actions use T1,T2,T3,SKIP.
        empirical = next_cost_block[:, [9, 10, 11, 8]]
        reward_cost_values = [float(row.reward_cost) for row in rows]
        reward_scalar_values = [
            (float(row.final_f1) if row.terminal else 0.0)
            - self.scalar_lambda_cost * cost
            for row, cost in zip(rows, reward_cost_values)
        ]
        mc_scalar_values = [
            float(row.final_f1)
            - self.scalar_lambda_cost
            * float(row.remaining_cost)
            / self.scalar_c_ref
            for row in rows
        ]
        return {
            "states": states,
            "legal": legal,
            "actions": actions,
            "terminal": terminal,
            "next_states": next_states,
            "next_legal": next_legal,
            "empirical": empirical,
            "reward_scalar": torch.tensor(
                reward_scalar_values, device=self.device
            ),
            "mc_scalar": torch.tensor(mc_scalar_values, device=self.device),
        }

    def _losses(self, rows: tuple[OfflineTransition, ...]) -> dict[str, torch.Tensor]:
        batch = self._batch_tensors(rows)
        actions = batch["actions"]
        selected_legal = batch["legal"].gather(1, actions.unsqueeze(1)).squeeze(1)
        if not bool(torch.all(selected_legal)):
            raise ValueError("offline replay contains an illegal sampled action")
        current = self.model(
            batch["states"], batch["legal"], mask_illegal=False
        ).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_next = self.model(batch["next_states"], batch["next_legal"])
            target_next = self.target_model(batch["next_states"], batch["next_legal"])
            target = scalar_double_dqn_targets(
                rewards=batch["reward_scalar"],
                online_next_q=online_next,
                target_next_q=target_next,
                legal_mask=batch["next_legal"],
                terminal=batch["terminal"],
                empirical_cost=batch["empirical"],
                gamma=self.config.gamma,
            )
        return {
            "td_scalar": F.smooth_l1_loss(current, target, beta=1.0),
            "mc_scalar": F.smooth_l1_loss(current, batch["mc_scalar"], beta=1.0),
        }

    def _combined_loss(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return losses["td_scalar"] + self.mc_weight * losses["mc_scalar"]

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite {name}")

    def train_to(self, target_step: int) -> None:
        target_step = int(target_step)
        if target_step > self.config.total_steps:
            raise ValueError("target step exceeds frozen training budget")
        self.model.train()
        while self.global_step < target_step:
            rows = self.replay.next_batch()
            self.last_batch_identity = self.replay.last_batch_identity
            losses = self._losses(rows)
            total = self._combined_loss(losses)
            self._require_finite("loss", total)
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    self._require_finite(f"gradient {name}", parameter.grad)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self._require_finite("gradient norm", torch.as_tensor(gradient_norm))
            self.optimizer.step()
            for name, parameter in self.model.named_parameters():
                self._require_finite(f"parameter {name}", parameter)
            self.global_step += 1
            if self.global_step % self.config.target_update_steps == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            if self.global_step % self.config.logging_epoch_steps == 0:
                self.logging_epoch += 1
            row = {name: float(value.detach().cpu()) for name, value in losses.items()}
            row["loss_total"] = float(total.detach().cpu())
            row["gradient_norm"] = float(torch.as_tensor(gradient_norm).cpu())
            row["step"] = float(self.global_step)
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError("non-finite training metric")
            self.metrics.append(row)
            if self.global_step % self.config.checkpoint_steps == 0:
                self.save_checkpoint(
                    self.output_dir
                    / "checkpoints"
                    / f"step_{self.global_step:06d}.pt"
                )

    def _checkpoint_data(self) -> dict[str, Any]:
        return {
            "experiment": asdict(self.experiment),
            "config": asdict(self.config),
            "mc_weight": self.mc_weight,
            "identities": dict(self.identities),
            "component_seeds": dict(self.component_seeds),
            "model_config": {"fusion_hidden_dim": self.fusion_hidden_dim},
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "logging_epoch": self.logging_epoch,
            "metrics": list(self.metrics),
            "last_batch_identity": self.last_batch_identity,
            "replay": self.replay.state_dict(),
            "pair_sampler": None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(self._checkpoint_data(), temporary)
        temporary.replace(target)
        return target

    def resume(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        gates = {
            "experiment": asdict(self.experiment),
            "config": asdict(self.config),
            "mc_weight": self.mc_weight,
            "identities": dict(self.identities),
            "component_seeds": dict(self.component_seeds),
        }
        for name, expected in gates.items():
            if data[name] != expected:
                raise ValueError(f"checkpoint {name} identity mismatch")
        checkpoint_model_config = data.get(
            "model_config", {"fusion_hidden_dim": 256}
        )
        if checkpoint_model_config != {"fusion_hidden_dim": self.fusion_hidden_dim}:
            raise ValueError("checkpoint model_config identity mismatch")
        self.model.load_state_dict(data["model"])
        self.target_model.load_state_dict(data["target_model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.global_step = int(data["global_step"])
        self.logging_epoch = int(data["logging_epoch"])
        self.metrics = list(data["metrics"])
        self.last_batch_identity = str(data["last_batch_identity"])
        self.replay.load_state_dict(data["replay"])
        if data.get("pair_sampler") is not None:
            raise ValueError("pairwise checkpoint is not supported by the submission trainer")
        random.setstate(data["python_rng"])
        np.random.set_state(data["numpy_rng"])
        torch.set_rng_state(data["torch_rng"])
        if data["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(data["cuda_rng"])


__all__ = ["OfflineRAGDQNTrainer", "component_seed"]
