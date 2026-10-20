from RAG.env._shared.model_config import (
    EMBEDDING_MODEL,
    LLM_PRICING_DOLLAR_PER_1M,
    tier_model,
)
from RAG.env._shared.state_payload import StatePayload
from RAG.env._shared.tier_registry import TierRegistry


def test_model_configuration_is_canonical():
    assert tier_model(1) == "qwen-plus"
    assert tier_model(2) == "qwen3.5-flash"
    assert tier_model(3) == "qwen-flash"
    assert EMBEDDING_MODEL == "text-embedding-v3"
    assert set(LLM_PRICING_DOLLAR_PER_1M) == {
        "qwen-plus", "qwen3.5-flash", "qwen-flash"
    }


def test_tier_registry_exposes_registered_tiers():
    TierRegistry.clear()

    def operator(payload):
        return payload

    TierRegistry.register("rewrite", 2, operator)

    assert TierRegistry.get("rewrite", 2) is operator
    assert TierRegistry.has("rewrite", 2)
    assert TierRegistry.available_tiers("rewrite") == [2]
    assert TierRegistry.registered_stages() == ["rewrite"]
    assert TierRegistry.get("missing", 1) is None

    TierRegistry.clear()


def test_state_payload_round_trips_without_aliasing():
    original = StatePayload(
        question_id="q1",
        question="Who?",
        ground_truth="Alice",
        context=["a"],
        metadata={"source": "test"},
    )

    clone = original.snapshot()
    clone.context.append("b")
    clone.metadata["source"] = "changed"

    assert original.context == ["a"]
    assert original.metadata == {"source": "test"}
    assert StatePayload.from_json_dict(original.to_json_dict()) == original


def test_state_payload_records_stage_costs():
    payload = StatePayload(question="Who?")
    payload.cost_tracker.record_llm_call(
        stage="question_to_query",
        model="qwen-plus",
        input_tokens=10,
        output_tokens=3,
        latency_ms=12.5,
    )

    assert payload.cost_tracker.total_input_tokens == 10
    assert payload.cost_tracker.total_output_tokens == 3
    assert payload.cost_tracker.stage_costs[0]["stage"] == "question_to_query"
    assert "cost_dollar" in payload.cost_tracker.stage_costs[0]
    assert payload.cost_tracker.total_cost_dollar() == payload.cost_tracker.stage_costs[0]["cost_dollar"]
    assert payload.snapshot().cost_tracker.stage_costs == payload.cost_tracker.stage_costs


def test_incompatible_cached_stage_cost_is_rejected():
    payload = StatePayload(question="Who?").to_json_dict()
    payload["cost_tracker"]["stage_costs"] = [{"stage": "question_to_query", "amount": 0.01}]
    import pytest
    with pytest.raises(ValueError, match="cost_dollar"):
        StatePayload.from_json_dict(payload)
