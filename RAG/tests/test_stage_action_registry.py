import importlib

import pytest

from RAG.env._shared.tier_registry import TierRegistry


def load_environment():
    module = importlib.import_module("RAG.env.0_old")
    return importlib.reload(module)


def test_decision_stage_actions_match_the_approved_skip_policy():
    module = load_environment()

    assert module.T1_ACTION == 0
    assert module.T2_ACTION == 1
    assert module.T3_ACTION == 2
    assert module.SKIP_ACTION == 3
    assert module.StageActionRegistry.available_actions("question_to_query") == [0, 1, 2, 3]
    assert module.StageActionRegistry.available_actions("context_question_to_query") == [0, 1, 2, 3]
    assert module.StageActionRegistry.available_actions("rerank") == [0, 1, 2, 3]
    assert module.StageActionRegistry.available_actions("context_question_to_answer") == [0, 1, 2]


def test_skip_is_an_action_and_not_a_fake_llm_tier():
    module = load_environment()

    for stage in module.StageActionRegistry.registered_stages():
        assert TierRegistry.available_tiers(stage) == [1, 2, 3]
        assert TierRegistry.get(stage, 4) is None


def test_retrieval1_always_runs_after_stage1_even_when_rewrite_is_skipped():
    module = load_environment()

    for action in (module.T1_ACTION, module.T2_ACTION, module.T3_ACTION, module.SKIP_ACTION):
        assert module.StageActionRegistry.fixed_followups(
            "question_to_query", action
        ) == ["retrieval1"]


def test_retrieval2_runs_only_when_second_hop_is_not_skipped():
    module = load_environment()

    for action in (module.T1_ACTION, module.T2_ACTION, module.T3_ACTION):
        assert module.StageActionRegistry.fixed_followups(
            "context_question_to_query", action
        ) == ["retrieval2"]
    assert module.StageActionRegistry.fixed_followups(
        "context_question_to_query", module.SKIP_ACTION
    ) == []


def test_retrieval1_is_fixed_and_cannot_be_registered_as_a_decision_stage():
    module = load_environment()

    assert "retrieval1" in module.FIXED_OPERATORS
    assert "retrieval1" not in module.StageActionRegistry.registered_stages()
    with pytest.raises(KeyError, match="retrieval1"):
        module.StageActionRegistry.available_actions("retrieval1")


def test_stage4_rejects_skip_action():
    module = load_environment()

    with pytest.raises(ValueError, match="does not allow action 3"):
        module.StageActionRegistry.fixed_followups(
            "context_question_to_answer", module.SKIP_ACTION
        )
