"""Run one frozen offline RAG DQN arm, with exact checkpoint resume."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from RAG.rl.dqn_experiments import EXPERIMENTS, TrainerConfig, run_id
from RAG.rl.dqn_replay import DeterministicRouteSchedule
from RAG.rl.offline_dqn_trainer import OfflineRAGDQNTrainer
from RAG.rl.prepared_artifacts import PreparedReplayDataset, PreparedReplayStore


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_embeddings(path: Path):
    import torch

    return torch.tensor(_json(path)["stage_embeddings"], dtype=torch.float32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--state-root", type=Path, default=Path("RAG/cache/rl_state_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("RAG/exp/rl_runs"))
    parser.add_argument("--seed", type=int, default=1145)
    parser.add_argument("--target-step", type=int, default=7500)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.target_step <= 0 or args.target_step > 7500:
        raise SystemExit("target-step must be in 1..7500")

    state_root = args.state_root.resolve()
    manifest = _json(state_root / "state_manifest.json")
    experiment = EXPERIMENTS[args.arm]
    lambda_cost = 0.01

    routes_payload = _json(state_root / "replay_schedule.json")
    schedule = DeterministicRouteSchedule(
        {int(qid): tuple(routes) for qid, routes in routes_payload["routes_by_qid"].items()},
        seed=args.seed,
        qids_per_step=16,
        routes_per_qid=4,
    )
    schedule.load_state_dict(routes_payload["initial_state"])
    store = PreparedReplayStore(
        state_root / "offline_replay.db",
        identity=str(manifest["prepared_identity"]),
        readonly=True,
    )
    replay = PreparedReplayDataset(store, schedule)
    current_run_id = run_id(experiment, args.seed)
    output_dir = args.output_root.resolve() / current_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainerConfig(seed=args.seed)
    identities = {
        "prepared_identity": str(manifest["prepared_identity"]),
        "state_schema_id": str(manifest["schema_id"]),
        "normalizer_identity": str(manifest["normalizer_identity"]),
        "semantic_model_identity": str(manifest["semantic_model_identity"]),
        "prompt_identity": str(manifest["prompt_identity"]),
        "cost_priors_identity": str(manifest["cost_priors_identity"]),
        "replay_schedule_identity": str(routes_payload["identity"]),
        "scalar_lambda_cost": str(lambda_cost),
        "scalar_mc_weight": str(config.mc_weight),
    }
    trainer = OfflineRAGDQNTrainer(
        experiment=experiment,
        replay=replay,
        pair_sampler=None,
        stage_embeddings=_stage_embeddings(state_root / "prompt_embeddings.json"),
        output_dir=output_dir,
        config=config,
        identities=identities,
        scalar_lambda_cost=lambda_cost,
        fusion_hidden_dim=256,
        device=args.device,
    )
    if args.resume:
        trainer.resume(args.resume)
    trainer.train_to(args.target_step)
    if trainer.global_step == config.total_steps:
        final_checkpoint = output_dir / "checkpoints" / f"step_{config.total_steps:06d}.pt"
        if not final_checkpoint.exists():
            raise FileNotFoundError(f"missing final checkpoint: {final_checkpoint}")
        shutil.copyfile(final_checkpoint, output_dir / "final.pt")
    run_config = {
        "run_id": current_run_id,
        "arm": asdict(experiment),
        "trainer_config": asdict(config),
        "mc_weight": config.mc_weight,
        "objective": {
            "c_ref": 0.00333879,
            "lambda_cost": lambda_cost,
            "mc_weight": config.mc_weight,
        },
        "model_config": {"fusion_hidden_dim": 256},
        "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
        "identities": identities,
        "component_seeds": trainer.component_seeds,
        "global_step": trainer.global_step,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trainer.metrics),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"run_id": current_run_id, "global_step": trainer.global_step, "metrics": trainer.metrics[-1:]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    store.close()
    print(json.dumps(run_config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
