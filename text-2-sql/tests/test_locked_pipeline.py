"""Offline checks for the published Text-to-SQL routing configuration."""

import torch
import json
import socket

import pytest
import numpy as np

from config import DEFAULT_RUN_NAME, get_run_config
from rl.route_ladder import RouteTemplate
from rl.env_wrapper import EnvironmentWrapper
from rl.experience_collector import ExperienceCollector
from rl.trajectory_buffer import TrajectoryBuffer
from rl.state_v26_network import StateV26ScalarDuelingNetwork
from rl.state_v26_experiment_factory import (
    build_state_v26_network,
    create_state_v26_initialization_bundle,
    load_state_v26_initialization,
)
from rl.state_v26_objective import (
    StateV26ObjectiveConfig,
    mc_huber_loss,
    return_to_go,
    scalar_transition_rewards,
)
from rl.state_v26_trainer import compute_state_v26_double_dqn_target
from rl.state_v26_replay import CachedStateV26RouteMaterializer
from rl.offline_runtime import OfflineNetworkError, offline_execution_guard
from rl.state_v26_encoder import StateV26Encoder, StateV26FeatureFlags
from rl.state_v25_schema import STATE_V25_DIM
from scripts._dev_runtime import _validated_cached


def test_locked_configuration_and_stage_constraints():
    config = get_run_config(DEFAULT_RUN_NAME)
    assert config["algorithm"] == "state_v26_scalar_double_dqn"
    assert config["lambda_cost"] == 0.1
    assert config["reward_version"] == "terminal_quality_mean_buffer_cost_v1"
    assert "c_ref" not in config
    assert config["lr"] == 1e-4
    assert config["hidden_dim"] == 512
    assert config["mc_weight"] == 0.3
    assert "cost_context_path" not in config
    assert "pair_index_path" not in config
    assert EnvironmentWrapper(operator_pkg="env.chess").non_skippable_decision_stage_idxs == (1, 4)
    StateV26ObjectiveConfig(lambda_cost=0.1, c_ref=0.012, mc_weight=0.3).validate()
    with pytest.raises(ValueError, match="Unknown run_name"):
        get_run_config("state_v25_experimental")


def test_terminal_quality_reward_and_undiscounted_mc_return():
    rewards = scalar_transition_rewards(
        np.asarray([0.002, 0.001, 0.0, 0.003, 0.002, 0.004]),
        final_ex=1,
        lambda_cost=0.1,
        c_ref=0.012,
    )
    assert np.allclose(rewards[:-1], [-1 / 60, -1 / 120, 0, -1 / 40, -1 / 60])
    assert rewards[-1] == pytest.approx(1 - 1 / 30)
    assert return_to_go(rewards)[0] == pytest.approx(0.9)


def test_replay_reward_accounts_for_intervening_fixed_costs():
    class CachedEnvironment:
        def rollout_with_cache(self, **_kwargs):
            return {
                "states": np.zeros((7, 5256), dtype=np.float32),
                "action_masks": np.asarray(
                    [[1, 1, 1, 1], [1, 1, 1, 0]] + [[1, 1, 1, 1]] * 4,
                    dtype=bool,
                ),
            }

    record = {
        "question_id": 1,
        "actions": [0] * 6,
        "exec_res": 1,
        "decision_stage_costs": [0.001] * 6,
        "fixed_stage_costs": [0, 0.002, 0.002, 0, 0, 0, 0, 0],
        "total_cost": 0.01,
    }
    materialize = CachedStateV26RouteMaterializer(
        CachedEnvironment(), None, None, lambda_cost=0.1, c_ref=0.01
    )
    batch = materialize(record)
    assert batch.rewards[0] == pytest.approx(-0.05)
    assert batch.return_to_go[0] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="total route cost"):
        materialize({**record, "total_cost": 0.011})


def test_fixed_route_collection_does_not_require_a_policy_network():
    route = RouteTemplate("anchor_t1", [0] * 6, 0)

    class FixedRouteEnvironment:
        def rollout_with_cache(self, **kwargs):
            assert kwargs == {
                "question_id": 1,
                "action_sequence": list(route.actions),
                "cache": None,
            }
            return {
                "actions": list(route.actions),
                "exec_res": 1,
                "total_cost": 0.01,
                "stage_costs": [0.0] * 8,
                "decision_stage_costs": [0.0] * 6,
            }

    buffer = TrajectoryBuffer(capacity=2)
    collector = ExperienceCollector(
        env=FixedRouteEnvironment(),
        cache=None,
        traj_buffer=buffer,
        route_templates=(route,),
        route_names=[route.name],
        n_workers=1,
    )
    metrics = collector.collect_from_questions([1], max_routes_per_query=1)
    assert metrics["n_collected"] == 1
    assert len(buffer) == 1


def test_v26_state_keeps_inactive_context_dimensions_zero():
    class PrefixEncoder:
        def encode(self, *_args):
            return np.ones(STATE_V25_DIM, dtype=np.float32)

    encoder = StateV26Encoder(
        PrefixEncoder(), None,
        StateV26FeatureFlags(cost_context=False, s4_difficulty_context=False),
    )
    state = encoder.encode(None, 0, 10.0, [], [], None)
    assert state.shape == (5256,)
    assert np.all(state[:STATE_V25_DIM] == 1)
    assert np.all(state[STATE_V25_DIM:] == 0)


def test_offline_guard_blocks_network():
    with offline_execution_guard():
        with pytest.raises(OfflineNetworkError):
            socket.create_connection(("example.com", 443))


def test_dev_cache_rejects_inconsistent_stage_costs():
    from rl.route_ladder import route_key

    actions = [0, 0, 0, 0, 0, 0]
    record = {
        "qid": 1,
        "db_id": "toy",
        "actions": actions,
        "route_key": route_key(actions),
        "exec_res": 1,
        "total_cost": 1.0,
        "stage_costs": [0.0] * 8,
        "decision_stage_costs": [0.0] * 6,
        "fixed_stage_costs": [0.0] * 8,
        "total_latency_ms": 0.0,
        "stage_latencies_ms": [0.0] * 8,
        "decision_stage_latencies_ms": [0.0] * 6,
        "fixed_stage_latencies_ms": [0.0] * 8,
        "final_sql": "SELECT 1",
    }
    with pytest.raises(ValueError, match="cost mismatch"):
        _validated_cached(record, qid=1, db_id="toy", actions=actions)


def test_value_advantage_network_and_mc_loss():
    torch.set_num_threads(1)
    network = StateV26ScalarDuelingNetwork()
    assert network.hidden_dim == 512
    assert len((network.request_encoder, network.artifact_encoder, network.trajectory_encoder)) == 3
    states = torch.zeros(2, 5256)
    from rl.state_v25_schema import STRUCTURED_SLICES

    states[:, STRUCTURED_SLICES["control"].start] = 1
    legal = torch.tensor([[True, True, True, True], [True, True, True, False]])
    q = network(states, legal)
    value, _ = network.debug_value_advantage(states, legal)
    assert q.shape == (2, 4)
    assert torch.isfinite(q).all()
    assert torch.allclose(q[0].mean(), value[0, 0])
    assert torch.allclose(q[1, :3].mean(), value[1, 0])
    assert torch.isfinite(mc_huber_loss(q[:, 0], torch.zeros(2)))


def test_seeded_initialization_matches_single_policy_configuration():
    config = get_run_config(DEFAULT_RUN_NAME)
    bundle = create_state_v26_initialization_bundle(seed=config["seed"])
    network = build_state_v26_network(config)
    load_state_v26_initialization(network, bundle)
    assert network.hidden_dim == config["hidden_dim"]
    assert network.request_encoder[0].out_features == config["repr_dim"]


def test_double_q_target_uses_online_selection_and_target_evaluation():
    class FixedQ(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, states, mask):
            return self.values.expand(states.shape[0], -1)

    online = FixedQ([1, 3, 9, 0])
    target = FixedQ([10, 20, 30, 40])
    result = compute_state_v26_double_dqn_target(
        online, target, torch.zeros(2, 1),
        torch.tensor([[True, True, False, False], [True, False, False, False]]),
        torch.tensor([-0.1, 0.5]), torch.tensor([0.0, 1.0]), 1.0,
    )
    assert torch.allclose(result, torch.tensor([19.9, 0.5]))


def test_chess_tiers_resolve_to_the_requested_models():
    import hashlib
    from pathlib import Path

    import env.chess as chess
    from env._shared.tier_registry import TierRegistry
    from env._shared.model_config import LLM_PRICING_DOLLAR_PER_1M, TIER_LLM_MODELS

    assert LLM_PRICING_DOLLAR_PER_1M == {
        "qwen-plus": {"input": 0.80, "output": 2.00},
        "qwen3.5-flash": {"input": 0.20, "output": 2.00},
        "qwen-flash": {"input": 0.15, "output": 1.50},
    }

    for tier, model_name in TIER_LLM_MODELS.items():
        operator = TierRegistry.get("candidate_generation", tier)
        assert operator is not None
        assert operator.__new2_source__ == "env.chess.candidate_generation"
        assert operator.__new2_model__ == model_name
    assert chess.stage_functions["candidate_generation"] is chess.candidate_generation_tier1
    source = Path(chess.candidate_generation_tier1.__code__.co_filename)
    identity = {
        "stage_source": {
            "module": "env.chess.candidate_generation",
            "callable_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "effective_model_config": {
            "model": TIER_LLM_MODELS[1],
            "temperature": 0,
            "enable_thinking": False,
        },
    }
    resolved = EnvironmentWrapper(operator_pkg="env.chess")._get_exact_operator(6, 0, identity)
    assert resolved.__new2_model__ == TIER_LLM_MODELS[1]


def test_dollar_cost_fields_and_legacy_cache_rejection():
    import copy

    import pytest

    from env._shared.state_payload import CostTracker, StatePayload

    tracker = CostTracker()
    tracker.record_llm_call("generation", "qwen-plus", 1_000_000, 0, 10.0)
    tracker.record_embedding_call("embedding", "text-embedding-v3", 1_000_000, 5.0)
    assert tracker.total_cost_dollar() == pytest.approx(1.3)
    assert tracker.summary()["total_cost_dollar"] == pytest.approx(1.3)
    assert tracker.summary()["total_cost_dollar_per_1k"] == pytest.approx(1300.0)

    serialized = StatePayload(cost_tracker=tracker).to_json_dict()
    restored = StatePayload.from_json_dict(serialized)
    assert restored.cost_tracker.total_cost_dollar() == pytest.approx(1.3)
    assert all("cost_dollar" in entry for entry in restored.cost_tracker.stage_costs)

    legacy = copy.deepcopy(serialized)
    entry = legacy["cost_tracker"]["stage_costs"][0]
    entry["cost_yuan"] = entry.pop("cost_dollar")
    with pytest.raises(ValueError, match="cost_dollar"):
        StatePayload.from_json_dict(legacy)
    with pytest.raises(ValueError, match="cost_dollar"):
        EnvironmentWrapper._merge_cached_stage_cost(CostTracker(), [entry])


def test_exact_contract_is_local_and_valid(tmp_path):
    from pathlib import Path
    import shutil
    from scripts.build_cache_contract import build_contract
    from rl.exact_cache_contract import STAGE_NAMES, PROMPT_STAGES

    source = Path(__file__).resolve().parents[1]
    for stage in STAGE_NAMES:
        destination = tmp_path / "env/chess" / f"{stage}.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / "env/chess" / f"{stage}.py").read_bytes())
    for index in PROMPT_STAGES:
        stage = STAGE_NAMES[index]
        destination = tmp_path / "env/templates" / f"template_{stage}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / "env/templates" / f"template_{stage}.txt").read_bytes())
    parser = tmp_path / "env/llm/parsers.py"
    parser.parent.mkdir(parents=True, exist_ok=True)
    parser.write_bytes((source / "env/llm/parsers.py").read_bytes())
    database = tmp_path / "data/train/train_databases/toy"
    for relative in (
        "toy.sqlite", "preprocessed/toy_lsh.pkl",
        "preprocessed/toy_minhashes.pkl", "database_description/columns.csv",
    ):
        path = database / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    (database / "context_vector_db").mkdir()
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps([{"question_id": 1, "db_id": "toy"}]), encoding="utf-8")
    contract = build_contract(tmp_path, selected, "https://example.com/v1")
    assert contract["operator_package"] == "env.chess"
    assert contract["stages"]["6"]["stage_source"]["module"] == "env.chess.candidate_generation"
    shutil.copytree(database, tmp_path / "data/dev/dev_databases/toy")
    dev_contract = build_contract(tmp_path, selected, "https://example.com/v1", db_mode="dev")
    assert "data/dev/dev_databases" in dev_contract["databases"]["toy"]["sqlite"]["path"].replace("\\", "/")


def test_generated_dev_normalizer_is_accepted_by_route_cache(tmp_path):
    from rl.reward import EX_FIRST_GLOBAL_STAGE_COST_V1, RewardConfig, RewardEvaluator
    from rl.stage_cost_normalization import load_global_stage_cost_normalizer
    from rl.trajectory_buffer import TrajectoryRecord
    from scripts._dev_runtime import DevTrajectoryCache
    from scripts.fit_dev_cost_normalizer import main as fit_normalizer

    record = TrajectoryRecord(
        question_id=1, actions=[0] * 6, reward=1.0, exec_res=1,
        total_cost=0.06, decision_stage_costs=[0.01] * 6,
    )
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([record.to_dict()]), encoding="utf-8")
    normalizer_path = tmp_path / "stage_cost_normalization.json"
    fit_normalizer(["--buffer", str(train_path), "--output", str(normalizer_path)])
    normalizer = load_global_stage_cost_normalizer(normalizer_path)
    evaluator = RewardEvaluator(RewardConfig(version=EX_FIRST_GLOBAL_STAGE_COST_V1), normalizer)
    outcome = evaluator.evaluate(exec_res=1, total_cost=record.total_cost,
                                 decision_stage_costs=record.decision_stage_costs)
    dev_path = tmp_path / "dev.json"
    dev_path.write_text(json.dumps([{**record.to_dict(), "metadata": evaluator.metadata(outcome)}]),
                        encoding="utf-8")
    cache = DevTrajectoryCache(dev_path, normalizer_path=str(normalizer_path))
    assert cache.get(qid=1, route_key="0-0-0-0-0-0") is not None
