"""Build and audit cache-only State V1 artifacts for offline RAG training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from RAG.rl.decision_session import RAGDecisionSession
from RAG.rl.dqn_replay import DeterministicRouteSchedule, reconstruct_offline_transitions
from RAG.rl.env_wrapper import RAGEnvironmentWrapper
from RAG.rl.prepared_artifacts import PreparedReplayStore
from RAG.rl.semantic_config import SemanticModelConfig, fingerprint_model_directory
from RAG.rl.state_v1_encoder import StateV1Encoder, StateV1Normalizer
from RAG.rl.state_v1_features import DecisionCostPriors
from RAG.rl.state_v1_schema import STATE_V1_DIM, STATE_V1_SCHEMA_ID
from RAG.rl.state_v1_semantic import (
    LocalGTESemanticEncoder,
    SemanticEmbeddingCache,
    load_stage_prompt_embeddings,
)
from RAG.rl.trajectory_buffer import TrajectoryBuffer, TrajectoryRecord
from RAG.rl.transition_cache import TransitionCache, TransitionKey


C_REF = 0.00333879
LAMBDA_COST = 0.01


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(value: Any, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_dataset(path: Path) -> dict[int, dict[str, Any]]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"dataset must be a JSON list: {path}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("dataset rows must be JSON objects")
        qid = int(row.get("question_id", row.get("qid")))
        if qid in result:
            raise ValueError(f"duplicate dataset qid: {qid}")
        result[qid] = dict(row)
    return result


def _load_train_snapshot(
    buffer_path: Path,
    dataset_path: Path,
) -> tuple[list[TrajectoryRecord], dict[int, dict[str, Any]], str, int, str]:
    buffer = TrajectoryBuffer(buffer_path)
    records = sorted(
        tuple(buffer), key=lambda row: (int(row.question_id), str(row.route_key))
    )
    if not records:
        raise ValueError("training trajectory buffer is empty")
    workflows = {str(row.workflow_identity) for row in records}
    versions = {int(row.operator_version) for row in records}
    splits = {str(row.split) for row in records}
    if len(workflows) != 1 or "" in workflows:
        raise ValueError("training buffer must have one nonempty workflow identity")
    if len(versions) != 1 or len(splits) != 1 or splits != {"train"}:
        raise ValueError("training buffer must have one operator version and split=train")
    by_pair = {(row.question_id, row.route_key) for row in records}
    if len(by_pair) != len(records):
        raise ValueError("training buffer contains duplicate qid/route identities")
    dataset = _load_dataset(dataset_path)
    missing = sorted({row.question_id for row in records} - set(dataset))
    if missing:
        raise ValueError(f"training dataset is missing buffered qids: {missing[:10]}")
    return records, dataset, next(iter(workflows)), next(iter(versions)), next(iter(splits))


def _transition_key_identity(key: TransitionKey) -> tuple[Any, ...]:
    return (
        int(key.stage_idx),
        int(key.tier),
        str(key.input_hash),
        int(key.operator_version),
    )


class _CostPriorAccumulator:
    def __init__(self) -> None:
        self._values: dict[tuple[Any, ...], float] = {}

    def add(
        self,
        decision_idx: int,
        action: int,
        transition_keys: Sequence[TransitionKey],
        cost: float,
    ) -> None:
        bundle = tuple(_transition_key_identity(key) for key in transition_keys)
        identity = (int(decision_idx), int(action), bundle)
        value = float(cost)
        existing = self._values.get(identity)
        if existing is not None and abs(existing - value) > 1e-12:
            raise ValueError("one exact transition bundle has inconsistent restored cost")
        self._values.setdefault(identity, value)

    def freeze(self) -> tuple[DecisionCostPriors, dict[str, Any]]:
        grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
        for (decision_idx, action, _bundle), value in self._values.items():
            grouped[(decision_idx, action)].append(value)
        means: list[list[float]] = []
        counts: list[list[int]] = []
        # State cost-prior order is SKIP,T1,T2,T3.
        for decision_idx in range(4):
            row_means: list[float] = []
            row_counts: list[int] = []
            for action in (3, 0, 1, 2):
                values = grouped.get((decision_idx, action), [])
                if not values and not (decision_idx == 3 and action == 3):
                    raise ValueError(
                        f"cost-prior cell has no support: decision={decision_idx}, action={action}"
                    )
                row_means.append(float(np.mean(values)) if values else 0.0)
                row_counts.append(len(values))
            means.append(row_means)
            counts.append(row_counts)
        evidence = {
            "action_order": ["SKIP", "T1", "T2", "T3"],
            "means": means,
            "unique_bundle_counts": counts,
        }
        identity = _canonical_sha(evidence)
        return (
            DecisionCostPriors(
                means=tuple(tuple(value for value in row) for row in means),
                identity=identity,
            ),
            {**evidence, "identity": identity},
        )


def _validate_replay_result(record: TrajectoryRecord, result: Mapping[str, Any]) -> None:
    if str(result["route_key"]) != str(record.route_key):
        raise ValueError("cache-only replay route key does not match trajectory buffer")
    if abs(float(result["f1_score"]) - float(record.f1_score)) > 1e-12:
        raise ValueError("cache-only replay F1 does not match trajectory buffer")
    if abs(float(result["total_cost"]) - float(record.total_cost)) > 1e-9:
        raise ValueError("cache-only replay cost does not match trajectory buffer")


def _raw_state_stream(
    *,
    records: Sequence[TrajectoryRecord],
    dataset: Mapping[int, Mapping[str, Any]],
    env: RAGEnvironmentWrapper,
    cache: TransitionCache,
    encoder: StateV1Encoder,
    operator_version: int,
    cost_accumulator: _CostPriorAccumulator,
) -> Iterable[Any]:
    for record in records:
        session = RAGDecisionSession.start(
            item=dataset[int(record.question_id)],
            env=env,
            cache=cache,
            operator_version=operator_version,
            cache_only=True,
        )
        for action in record.actions:
            yield encoder.raw_blocks(session.observation())
            step = session.apply_action(int(action))
            cost_accumulator.add(
                step.decision_idx,
                step.action,
                step.transition_keys,
                step.bundled_cost,
            )
        _validate_replay_result(record, session.finalize())


def _read_tsv(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        label, separator, prompt = line.partition("\t")
        if not separator or not prompt.strip():
            raise ValueError(f"invalid prompt TSV row {line_number}: {path}")
        rows.append((label.strip(), prompt.strip()))
    return rows


def _prompt_artifact(
    *,
    stage_prompt: Path,
    action_prompt: Path,
    semantic_encoder: LocalGTESemanticEncoder,
) -> tuple[dict[str, Any], str, np.ndarray]:
    stage_rows = _read_tsv(stage_prompt)
    action_rows = _read_tsv(action_prompt)
    if [label for label, _prompt in stage_rows] != ["S1", "S2", "S3", "S4"]:
        raise ValueError("stage prompt rows must be exactly S1..S4")
    if [label for label, _prompt in action_rows] != ["T1", "T2", "T3", "SKIP"]:
        raise ValueError("action prompt rows must be exactly T1,T2,T3,SKIP")
    stage_embeddings = load_stage_prompt_embeddings(stage_prompt, semantic_encoder)
    action_embeddings = semantic_encoder.encode_texts([row[1] for row in action_rows])
    artifact = {
        "semantic_model_identity": semantic_encoder.model_identity,
        "stage_prompt_sha256": _sha256_file(stage_prompt),
        "action_prompt_sha256": _sha256_file(action_prompt),
        "stage_labels": [row[0] for row in stage_rows],
        "action_labels": [row[0] for row in action_rows],
        "stage_embeddings": stage_embeddings.tolist(),
        "action_embeddings": action_embeddings.tolist(),
    }
    identity = _canonical_sha(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"stage_embeddings", "action_embeddings"}
        }
    )
    artifact["identity"] = identity
    return artifact, identity, stage_embeddings


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records, dataset, workflow_identity, operator_version, split = _load_train_snapshot(
        args.train_buffer.resolve(), args.dataset.resolve()
    )
    if args.expected_qids > 0 and len({int(row.question_id) for row in records}) != args.expected_qids:
        raise ValueError(
            "training qid cardinality mismatch: "
            f"expected {args.expected_qids}, found {len({int(row.question_id) for row in records})}"
        )
    route_counts = defaultdict(int)
    for row in records:
        route_counts[int(row.question_id)] += 1
    manifest_path = args.train_buffer.with_name(
        f"{args.train_buffer.stem}_manifest.json"
    )
    source_manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    expected_routes = args.expected_routes_per_qid or int(source_manifest.get("route_count_per_query", 0))
    if expected_routes <= 0 or set(route_counts.values()) != {expected_routes}:
        raise ValueError(
            "training route cardinality mismatch: "
            f"expected {expected_routes} per qid, found {sorted(set(route_counts.values()))}"
        )
    index_identity = str(
        source_manifest.get("identity", {}).get("index_identity", "")
        if isinstance(source_manifest, Mapping)
        else ""
    )

    semantic_config = SemanticModelConfig(model_path=str(args.gte_model.resolve()))
    model_fingerprint = fingerprint_model_directory(args.gte_model.resolve())
    semantic_cache = SemanticEmbeddingCache(output_root / "semantic_embeddings.db")
    semantic_encoder = LocalGTESemanticEncoder(
        semantic_config,
        cache=semantic_cache,
        model_fingerprint=model_fingerprint,
    )
    prompt_artifact, prompt_identity, stage_embeddings = _prompt_artifact(
        stage_prompt=args.stage_prompt.resolve(),
        action_prompt=args.action_prompt.resolve(),
        semantic_encoder=semantic_encoder,
    )
    _atomic_json(prompt_artifact, output_root / "prompt_embeddings.json")

    env = RAGEnvironmentWrapper(index_identity=index_identity)
    cache = TransitionCache(
        args.train_cache.resolve(),
        workflow_identity=workflow_identity,
        readonly=True,
    )
    try:
        dummy_priors = DecisionCostPriors(
            means=((0.0, 0.0, 0.0, 0.0),) * 4,
            identity="normalizer-pass-no-cost-priors",
        )
        raw_encoder = StateV1Encoder(
            semantic_encoder=semantic_encoder,
            cost_priors=dummy_priors,
            normalizer=StateV1Normalizer.identity_transform(),
            prompt_identity=prompt_identity,
            c_ref=C_REF,
        )
        accumulator = _CostPriorAccumulator()
        normalizer = StateV1Normalizer.fit(
            _raw_state_stream(
                records=records,
                dataset=dataset,
                env=env,
                cache=cache,
                encoder=raw_encoder,
                operator_version=operator_version,
                cost_accumulator=accumulator,
            )
        )
        if normalizer.count != len(records) * 4:
            raise ValueError("normalizer did not consume every trajectory decision")
        normalizer.save(output_root / "train_normalizer.json")
        cost_priors, cost_artifact = accumulator.freeze()
        _atomic_json(cost_artifact, output_root / "cost_priors.json")

        source_identity = {
            "schema_id": STATE_V1_SCHEMA_ID,
            "train_buffer_sha256": _sha256_file(args.train_buffer.resolve()),
            "train_dataset_sha256": _sha256_file(args.dataset.resolve()),
            "workflow_identity": workflow_identity,
            "operator_version": operator_version,
            "split": split,
            "normalizer_identity": normalizer.identity,
            "semantic_model_identity": semantic_encoder.model_identity,
            "model_fingerprint": model_fingerprint,
            "prompt_identity": prompt_identity,
            "cost_priors_identity": cost_priors.identity,
            "c_ref": C_REF,
            "lambda_cost": LAMBDA_COST,
            "seed": int(args.seed),
        }
        prepared_identity = _canonical_sha(source_identity)
        encoder = StateV1Encoder(
            semantic_encoder=semantic_encoder,
            cost_priors=cost_priors,
            normalizer=normalizer,
            prompt_identity=prompt_identity,
            c_ref=C_REF,
        )
        replay_path = output_root / "offline_replay.db"
        routes_by_qid: dict[int, list[str]] = defaultdict(list)
        with PreparedReplayStore(replay_path, identity=prepared_identity) as store:
            for index, record in enumerate(records, 1):
                transitions = reconstruct_offline_transitions(
                    item=dataset[int(record.question_id)],
                    trajectory=record.to_dict(),
                    env=env,
                    cache=cache,
                    encoder=encoder,
                    operator_version=operator_version,
                    c_ref=C_REF,
                    lambda_cost=LAMBDA_COST,
                )
                store.put_trajectory(transitions)
                routes_by_qid[int(record.question_id)].append(str(record.route_key))
                if index % int(args.commit_every) == 0:
                    store.commit()
            store.commit()
            counts = {
                "trajectory_count": store.trajectory_count(),
                "transition_count": store.transition_count(),
                "unique_state_count": store.state_count(),
            }

        schedule = DeterministicRouteSchedule(
            {qid: tuple(sorted(set(routes))) for qid, routes in routes_by_qid.items()},
            seed=int(args.seed),
            qids_per_step=16,
            routes_per_qid=4,
        )
        schedule_artifact = {
            "routes_by_qid": {
                str(qid): list(routes)
                for qid, routes in sorted(schedule.routes_by_qid.items())
            },
            "initial_state": schedule.state_dict(),
            "identity": schedule.identity,
        }
        _atomic_json(schedule_artifact, output_root / "replay_schedule.json")
        manifest = {
            "version": 1,
            "status": "completed",
            "prepared_identity": prepared_identity,
            **source_identity,
            **counts,
            "qid_count": len(routes_by_qid),
            "routes_per_qid": sorted({len(set(value)) for value in routes_by_qid.values()}),
            "replay_schedule_identity": schedule.identity,
            "state_dimension": STATE_V1_DIM,
            "semantic_config": asdict(semantic_config),
            "stage_embeddings_shape": list(stage_embeddings.shape),
            "artifacts": {
                "normalizer": "train_normalizer.json",
                "semantic_embeddings": "semantic_embeddings.db",
                "prompt_embeddings": "prompt_embeddings.json",
                "cost_priors": "cost_priors.json",
                "offline_replay": "offline_replay.db",
                "replay_schedule": "replay_schedule.json",
            },
        }
        _atomic_json(manifest, output_root / "state_manifest.json")
        return manifest
    finally:
        cache.close()
        semantic_cache.close()


def _load_cost_priors(path: Path) -> DecisionCostPriors:
    value = _load_json(path)
    return DecisionCostPriors(
        means=tuple(tuple(float(item) for item in row) for row in value["means"]),
        identity=str(value["identity"]),
    )


def verify_sample(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest = _load_json(output_root / "state_manifest.json")
    records, dataset, workflow_identity, operator_version, _split = _load_train_snapshot(
        args.train_buffer.resolve(), args.dataset.resolve()
    )
    if workflow_identity != str(manifest["workflow_identity"]):
        raise ValueError("verification workflow identity differs from State manifest")
    semantic_config = SemanticModelConfig(**manifest["semantic_config"])
    semantic_cache = SemanticEmbeddingCache(output_root / "semantic_embeddings.db")
    semantic_encoder = LocalGTESemanticEncoder(
        semantic_config,
        cache=semantic_cache,
        model_fingerprint=str(manifest["model_fingerprint"]),
    )
    normalizer = StateV1Normalizer.load(output_root / "train_normalizer.json")
    encoder = StateV1Encoder(
        semantic_encoder=semantic_encoder,
        cost_priors=_load_cost_priors(output_root / "cost_priors.json"),
        normalizer=normalizer,
        prompt_identity=str(manifest["prompt_identity"]),
        c_ref=C_REF,
    )
    by_qid: dict[int, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        by_qid[int(record.question_id)].append(record)
    rng = np.random.default_rng(int(args.seed))
    qids = sorted(by_qid)
    if len(qids) < int(args.sample_qids):
        raise ValueError("sample_qids exceeds prepared Train qid count")
    selected_qids = sorted(
        int(value)
        for value in rng.choice(qids, size=int(args.sample_qids), replace=False)
    )
    cache = TransitionCache(
        args.train_cache.resolve(), workflow_identity=workflow_identity, readonly=True
    )
    env = RAGEnvironmentWrapper(index_identity=str(manifest.get("index_identity", "")))
    report = {
        "sampled_qids": len(selected_qids),
        "sampled_routes": 0,
        "checked_states": 0,
        "cache_misses": 0,
        "shape_failures": 0,
        "non_finite": 0,
        "hash_mismatches": 0,
        "causal_visibility_failures": 0,
    }
    try:
        for qid in selected_qids:
            candidates = sorted(by_qid[qid], key=lambda row: row.route_key)
            with_skip = next((row for row in candidates if 3 in row.actions[:3]), None)
            without_skip = next((row for row in candidates if 3 not in row.actions), None)
            chosen = [row for row in (with_skip, without_skip) if row is not None]
            chosen = list({row.route_key: row for row in chosen}.values())
            if len(chosen) != int(args.routes_per_qid):
                raise ValueError(f"qid={qid} lacks the required SKIP/non-SKIP sample routes")
            report["sampled_routes"] += len(chosen)
            prefix_hashes: dict[tuple[int, tuple[int, ...]], tuple[str, str]] = {}
            for record in chosen:
                first = reconstruct_offline_transitions(
                    item=dataset[qid], trajectory=record.to_dict(), env=env, cache=cache,
                    encoder=encoder, operator_version=operator_version,
                    c_ref=C_REF, lambda_cost=LAMBDA_COST,
                )
                second = reconstruct_offline_transitions(
                    item=dataset[qid], trajectory=record.to_dict(), env=env, cache=cache,
                    encoder=encoder, operator_version=operator_version,
                    c_ref=C_REF, lambda_cost=LAMBDA_COST,
                )
                for left, right in zip(first, second):
                    report["checked_states"] += 1
                    if left.state.vector.shape != (STATE_V1_DIM,) or left.state.vector.dtype != np.float32:
                        report["shape_failures"] += 1
                    if not np.isfinite(left.state.vector).all():
                        report["non_finite"] += 1
                    if (
                        left.state.logical_state_hash != right.state.logical_state_hash
                        or left.state.tensor_state_hash != right.state.tensor_state_hash
                        or not np.array_equal(left.state.vector, right.state.vector)
                    ):
                        report["hash_mismatches"] += 1
                    prefix_key = (left.decision_idx, tuple(record.actions[: left.decision_idx]))
                    hashes = (left.state.logical_state_hash, left.state.tensor_state_hash)
                    existing = prefix_hashes.setdefault(prefix_key, hashes)
                    if existing != hashes:
                        report["causal_visibility_failures"] += 1
        _atomic_json(report, output_root / "verification_sample.json")
        failures = sum(
            int(report[name])
            for name in (
                "cache_misses", "shape_failures", "non_finite",
                "hash_mismatches", "causal_visibility_failures",
            )
        )
        if failures:
            raise ValueError(f"State V1 verification failed: {report}")
        return report
    except LookupError:
        report["cache_misses"] += 1
        _atomic_json(report, output_root / "verification_sample.json")
        raise
    finally:
        cache.close()
        semantic_cache.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "verify-sample"), required=True)
    parser.add_argument("--train-buffer", type=Path, default=Path("RAG/cache/new1.json"))
    parser.add_argument("--train-cache", type=Path, default=Path("RAG/cache/new1.db"))
    parser.add_argument("--dataset", type=Path, default=Path("RAG/data/train.json"))
    parser.add_argument("--output-root", type=Path, default=Path("RAG/cache/rl_state_v1"))
    parser.add_argument("--gte-model", type=Path, required=True)
    parser.add_argument("--stage-prompt", type=Path, default=Path("RAG/LLM/stage_prompt.tsv"))
    parser.add_argument("--action-prompt", type=Path, default=Path("RAG/LLM/action_prompt.tsv"))
    parser.add_argument("--seed", type=int, default=1145)
    parser.add_argument("--sample-qids", type=int, default=100)
    parser.add_argument("--routes-per-qid", type=int, default=2)
    parser.add_argument("--commit-every", type=int, default=100)
    parser.add_argument("--expected-qids", type=int, default=0)
    parser.add_argument("--expected-routes-per-qid", type=int, default=0)
    args = parser.parse_args(argv)
    if args.sample_qids <= 0 or args.routes_per_qid <= 0 or args.commit_every <= 0:
        raise SystemExit("sample/commit counts must be positive")
    result = build(args) if args.mode == "build" else verify_sample(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
