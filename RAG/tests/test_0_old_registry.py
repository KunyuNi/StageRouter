import importlib

from RAG.env._shared.tier_registry import TierRegistry


LLM_STAGES = {
    "question_to_query",
    "context_question_to_query",
    "rerank",
    "context_question_to_answer",
}


def load_environment():
    TierRegistry.clear()
    module = importlib.import_module("RAG.env.0_old")
    return importlib.reload(module)


def test_four_llm_stages_register_all_tiers():
    load_environment()

    assert set(TierRegistry.registered_stages()) == LLM_STAGES
    for stage in LLM_STAGES:
        assert TierRegistry.available_tiers(stage) == [1, 2, 3]


def test_fixed_stages_are_available_but_not_tiered():
    module = load_environment()

    assert set(module.FIXED_OPERATORS) == {
        "retrieval1",
        "retrieval2",
        "qa_evaluation",
    }
    for stage in module.FIXED_OPERATORS:
        assert TierRegistry.available_tiers(stage) == []


def test_rag_environment_has_no_runtime_nl2sql_dependency():
    module = load_environment()

    registered = [
        TierRegistry.require(stage, tier)
        for stage in LLM_STAGES
        for tier in (1, 2, 3)
    ]
    assert module.__name__ == "RAG.env.0_old"
    assert all(not operator.__module__.startswith("nl2sql") for operator in registered)

