"""Cache-first evaluation of one frozen RAG DQN policy on the Dev/Test split."""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from RAG.env._shared.llm_adapter import get_adapter
from RAG.env._shared.retriever import get_retriever
from RAG.rl.dqn_networks import ScalarStageFiLMDuelingDQN
from RAG.rl.dqn_policy import ScalarGreedyPolicy
from RAG.rl.dev_evaluator import TrajectoryBufferWriter, evaluate_policy
from RAG.rl.env_wrapper import RAGEnvironmentWrapper
from RAG.rl.failure_queue import EvaluationFailureQueue
from RAG.rl.semantic_config import SemanticModelConfig
from RAG.rl.state_v1_encoder import StateV1Encoder, StateV1Normalizer
from RAG.rl.state_v1_features import DecisionCostPriors
from RAG.rl.state_v1_semantic import LocalGTESemanticEncoder, SemanticEmbeddingCache
from RAG.rl.state_v1_schema import STATE_V1_SCHEMA
from RAG.rl.trajectory_buffer import TrajectoryBuffer
from RAG.rl.transition_cache import TransitionCache


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset(path: Path) -> list[dict[str, Any]]:
    rows = _json(path)
    if not isinstance(rows, list):
        raise ValueError("evaluation dataset must be a JSON list")
    return [dict(row) for row in rows]


def _model_and_policy(
    checkpoint: Path,
    state_root: Path,
    policy_name: str,
    device: str,
):
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    experiment = checkpoint_data["experiment"]
    fusion_hidden_dim = int(
        checkpoint_data.get("model_config", {}).get("fusion_hidden_dim", 256)
    )
    stage_embeddings = torch.tensor(
        _json(state_root / "prompt_embeddings.json")["stage_embeddings"],
        dtype=torch.float32,
    )
    if (policy_name != "scalar" or experiment.get("arm") != "S1"
            or experiment.get("q_formulation") != "scalar"
            or not experiment.get("mc_enabled")
            or experiment.get("pairwise_enabled")
            or experiment.get("head_type", "dueling") != "dueling"):
        raise ValueError("submission evaluation requires an S1 scalar TD+MC Dueling checkpoint")
    model = ScalarStageFiLMDuelingDQN(
        STATE_V1_SCHEMA,
        stage_embeddings,
        fusion_hidden_dim=fusion_hidden_dim,
    )
    policy = ScalarGreedyPolicy()
    model.load_state_dict(checkpoint_data["model"])
    model.to(device)
    model.eval()
    return model, policy, checkpoint_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy", choices=("scalar",), default="scalar")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("RAG/data/test.json"))
    parser.add_argument("--manifest", type=Path, default=Path("RAG/exp/dev_manifest.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("RAG/exp"))
    parser.add_argument("--state-root", type=Path, default=Path("RAG/cache/rl_state_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("RAG/exp/rl_eval"))
    parser.add_argument("--qid-file", type=Path)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--n-workers", type=int, default=64)
    parser.add_argument("--autosave-every", type=int, default=10)
    parser.add_argument("--operator-version", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.n_workers <= 0 or args.autosave_every <= 0:
        raise SystemExit("worker and autosave counts must be positive")

    manifest = _json(args.manifest.resolve())
    workflow_identity = str(manifest["workflow_identity"])
    operator_version = int(manifest.get("operator_version", args.operator_version))
    if operator_version != args.operator_version:
        raise ValueError("operator version differs from Dev manifest")
    rows = _dataset(args.dataset.resolve())
    excluded = {int(value) for value in manifest.get("excluded_query_ids", [])}
    shared_safety_path = args.output_root.resolve() / "shared_excluded_query_ids.json"
    if shared_safety_path.exists():
        shared_safety = _json(shared_safety_path)
        excluded.update(
            int(value)
            for value in shared_safety.get("excluded_query_ids", shared_safety)
        )
    if args.qid_file:
        requested = {
            int(value.strip())
            for value in args.qid_file.read_text(encoding="utf-8").splitlines()
            if value.strip()
        }
        rows = [row for row in rows if int(row.get("question_id", row.get("qid"))) in requested]
    rows = [
        row for row in rows
        if int(row.get("question_id", row.get("qid"))) not in excluded
    ]
    model, policy, checkpoint_data = _model_and_policy(
        args.checkpoint.resolve(), args.state_root.resolve(), args.policy, args.device
    )
    state_manifest = _json(args.state_root.resolve() / "state_manifest.json")
    normalizer = StateV1Normalizer.load(args.state_root.resolve() / "train_normalizer.json")
    semantic_config = SemanticModelConfig(**state_manifest["semantic_config"])
    # Keep evaluation text embeddings separate from the frozen Train State
    # artifact.  Route/transition reuse still happens through the shared Dev
    # JSON/SQLite caches below, but Test/Dev text must not mutate Train cache.
    eval_semantic_path = args.output_root.resolve() / "semantic_embeddings.db"
    eval_semantic_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_cache = SemanticEmbeddingCache(eval_semantic_path)
    semantic = LocalGTESemanticEncoder(
        semantic_config,
        cache=semantic_cache,
        model_fingerprint=str(state_manifest["model_fingerprint"]),
    )
    encoder = StateV1Encoder(
        semantic_encoder=semantic,
        cost_priors=DecisionCostPriors(
            means=tuple(tuple(float(x) for x in row) for row in _json(args.state_root.resolve() / "cost_priors.json")["means"]),
            identity=str(_json(args.state_root.resolve() / "cost_priors.json")["identity"]),
        ),
        normalizer=normalizer,
        prompt_identity=str(state_manifest["prompt_identity"]),
    )
    dev_buffer_path = args.cache_root.resolve() / "dev.json"
    trajectory_buffer = TrajectoryBuffer(
        dev_buffer_path,
        workflow_identity=workflow_identity,
        operator_version=operator_version,
    )
    writer = TrajectoryBufferWriter(trajectory_buffer, autosave_every=args.autosave_every)
    failure_queue = EvaluationFailureQueue(
        args.cache_root.resolve() / "dev_failed_pairs.json",
        split="test",
        policy_id=args.run_id,
        workflow_identity=workflow_identity,
        operator_version=operator_version,
    )
    transition_cache = TransitionCache(
        args.cache_root.resolve() / "dev.db",
        workflow_identity=workflow_identity,
        readonly=bool(args.cache_only),
    )
    adapter = None if args.cache_only else get_adapter()
    retriever = None if args.cache_only else get_retriever()
    env = RAGEnvironmentWrapper(
        adapter=adapter,
        retriever=retriever,
        index_identity=str(manifest.get("identity", {}).get("index_identity", "")),
    )
    try:
        summary = evaluate_policy(
            items=rows,
            model=model,
            policy=policy,
            env=env,
            transition_cache=transition_cache,
            trajectory_buffer=trajectory_buffer,
            writer=writer,
            failure_queue=failure_queue,
            operator_version=operator_version,
            workflow_identity=workflow_identity,
            state_encoder=encoder,
            split="test",
            policy_id=args.run_id,
            cache_only=args.cache_only,
            n_workers=args.n_workers,
        )
        output = args.output_root.resolve() / args.run_id / args.policy
        output.mkdir(parents=True, exist_ok=True)
        predictions = [asdict(row) for row in summary.predictions]
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        queue = failure_queue.payload
        safety_path = args.output_root.resolve() / "shared_excluded_query_ids.json"
        prior_safety: set[int] = set()
        if safety_path.exists():
            prior = _json(safety_path)
            prior_safety.update(
                int(value)
                for value in prior.get("excluded_query_ids", prior)
            )
        prior_safety.update(
            int(row["question_id"])
            for row in queue["resolved"]
            if row.get("resolution") == "excluded_content_safety"
        )
        safety_path.parent.mkdir(parents=True, exist_ok=True)
        safety_path.write_text(
            json.dumps(
                {"excluded_query_ids": sorted(prior_safety)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "policy": args.policy,
                    "completed": summary.completed,
                    "failed": summary.failed,
                    "unresolved": len(queue["unresolved"]),
                    "resolved": len(queue["resolved"]),
                    "safety_excluded_qids": sorted(prior_safety),
                    "checkpoint_global_step": checkpoint_data["global_step"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"completed": summary.completed, "failed": summary.failed}, ensure_ascii=False))
        return 0
    finally:
        transition_cache.close()
        semantic_cache.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
