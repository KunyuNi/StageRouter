"""Small, offline checks for the five-step submission contract."""

from __future__ import annotations

import json

import pytest
import torch

from RAG.rl.dqn_experiments import EXPERIMENTS, TrainerConfig
from RAG.rl.dqn_networks import ScalarStageFiLMDuelingDQN
from RAG.rl.dqn_objective import scalar_double_dqn_targets, scalar_mc_returns, scalar_rewards
from RAG.rl.stage_mapping import DECISION_TO_PHYSICAL
from RAG.rl.state_v1_schema import STATE_V1_DIM, STATE_V1_SCHEMA, legal_action_mask
from RAG.scripts.build_route_library import build
from RAG.scripts.collect_fixed_routes import _load_observed_route_plan


def test_anchor_observations_freeze_cost_pruned_route_plan(tmp_path):
    buffer = tmp_path / "anchors.json"
    manifest = tmp_path / "anchors_manifest.json"
    manifest.write_text(json.dumps({"status": "completed", "workflow_identity": "wf", "operator_version": 1}))
    f1 = {"all_t1": 0.8, "all_t2": 0.6, "all_t3": 0.5}
    rows = [
        {
            "question_id": qid,
            "route_name": name,
            "actions": list(actions),
            "workflow_identity": "wf",
            "operator_version": 1,
            "f1_score": f1[name],
            "total_cost": 0.004,
            "decision_stage_costs": [0.001, 0.0004, 0.002, 0.0006],
        }
        for qid in (1, 2)
        for name, actions in (("all_t1", (0, 0, 0, 0)), ("all_t2", (1, 1, 1, 1)), ("all_t3", (2, 2, 2, 2)))
    ]
    buffer.write_text(json.dumps(rows))
    plan = build(buffer, manifest)
    assert plan["anchor_qids"] == 2
    assert plan["api_calls"] == 0
    assert plan["high_cost_stages"] == [0, 2, 3]
    assert len(plan["routes"]) > 38
    assert any(route["name"].startswith("couple_") for route in plan["routes"])
    assert any(route["actions"][1] == 3 for route in plan["routes"])
    assert not any(route["name"].startswith("isolate_") and "_s2_" in route["name"] for route in plan["routes"])
    path = tmp_path / "route_plan.json"
    path.write_text(json.dumps(plan))
    profile = _load_observed_route_plan(path)
    assert profile.sha256 == plan["route_profile_sha256"]
    assert all(route.actions[3] != 3 for route in profile.routes)
    plan["routes"][3]["actions"][3] = 3
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="cost-pruning rule"):
        _load_observed_route_plan(path)


def test_only_scalar_dueling_td_mc_is_registered_and_rag_masks_are_native():
    assert set(EXPERIMENTS) == {"S1"}
    assert EXPERIMENTS["S1"].mc_enabled
    assert EXPERIMENTS["S1"].head_type == "dueling"
    assert DECISION_TO_PHYSICAL == (0, 2, 4, 5)
    assert [tuple(legal_action_mask(i)) for i in range(4)] == [
        (True, True, True, True),
        (True, True, True, True),
        (True, True, True, True),
        (True, True, True, False),
    ]
    assert TrainerConfig().total_steps == 7500
    assert TrainerConfig().mc_weight == 0.25


def test_scalar_reward_double_q_and_dueling_head_agree():
    costs = (0.001, 0.002, 0.0, 0.001)
    reward = scalar_rewards(0.75, costs, c_ref=0.004, lambda_cost=0.01)
    returns = scalar_mc_returns(0.75, costs, c_ref=0.004, lambda_cost=0.01)
    assert sum(reward) == pytest.approx(0.74)
    assert returns[0] == pytest.approx(0.74)
    target = scalar_double_dqn_targets(
        rewards=torch.tensor([reward[0]]),
        online_next_q=torch.tensor([[0.0, 2.0, 1.0, 100.0]]),
        target_next_q=torch.tensor([[0.0, 0.4, 0.9, 50.0]]),
        legal_mask=torch.tensor([[True, True, True, False]]),
        terminal=torch.tensor([False]),
    )
    assert target.item() == pytest.approx(reward[0] + 0.4)
    model = ScalarStageFiLMDuelingDQN(STATE_V1_SCHEMA, torch.zeros(4, 1024))
    q = model(torch.zeros(1, STATE_V1_DIM), torch.tensor([[True, True, True, False]]))
    assert q.shape == (1, 4)
    assert torch.isneginf(q[0, 3])
