import importlib

import pytest

from RAG.env._shared.llm_adapter import LLMResult
from RAG.env._shared.retriever import RetrievalItem, RetrievalResponse
from RAG.env._shared.state_payload import StatePayload
from RAG.env._shared.tier_registry import TierRegistry


class RecordingAdapter:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, model, prompt, **options):
        self.calls.append({"model": model, "prompt": prompt, "options": options})
        return LLMResult(
            text=self.response,
            input_tokens=11,
            output_tokens=4,
            latency_ms=2.5,
        )


class RecordingRetriever:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def retrieve(self, query, top_k=3, embedding_model=""):
        self.calls.append((query, top_k, embedding_model))
        value = next(self.responses)
        if isinstance(value, RetrievalResponse):
            return value
        return RetrievalResponse(
            query=query,
            results=[
                RetrievalItem(
                    chunk_id=f"fake:{text}",
                    text=text,
                    score=1.0,
                    rank=index,
                    metadata={},
                )
                for index, text in enumerate(value, start=1)
            ],
        )


def load_environment():
    module = importlib.import_module("RAG.env.0_old")
    return importlib.reload(module)


@pytest.mark.parametrize(
    ("stage", "payload", "response"),
    [
        ("question_to_query", StatePayload(question="Who won?"), "search query: winner"),
        (
            "context_question_to_query",
            StatePayload(question="Who won?", context=["A race happened"]),
            "search query: race winner",
        ),
        (
            "rerank",
            StatePayload(question="Who won?", context=["Alice won"]),
            '{"score": 95, "justification": "direct"}',
        ),
        (
            "context_question_to_answer",
            StatePayload(question="Who won?", context=["Alice won"]),
            "answer: Alice",
        ),
    ],
)
@pytest.mark.parametrize(
    ("tier", "expected_model"),
    [(1, "qwen-plus"), (2, "qwen3.5-flash"), (3, "qwen-flash")],
)
def test_each_llm_stage_routes_every_tier(stage, payload, response, tier, expected_model):
    load_environment()
    adapter = RecordingAdapter(response)

    result = TierRegistry.require(stage, tier)(payload.snapshot(), adapter=adapter)

    assert result is not payload
    assert [call["model"] for call in adapter.calls] == [expected_model]
    assert result.cost_tracker.stage_costs[-1]["stage"] == stage
    assert result.cost_tracker.stage_costs[-1]["model"] == expected_model


def test_retrieval1_falls_back_to_question_and_uses_fixed_embedding_model():
    module = load_environment()
    retriever = RecordingRetriever([["first evidence"]])
    payload = StatePayload(question="Who won?", query_1="")

    result = module.FIXED_OPERATORS["retrieval1"](
        payload, retriever=retriever, top_k=5
    )

    assert retriever.calls == [("Who won?", 5, "text-embedding-v3")]
    assert result.context == ["first evidence"]


def test_retrieval2_merges_and_stably_deduplicates_context():
    module = load_environment()
    retriever = RecordingRetriever([["shared", "new", "new"]])
    payload = StatePayload(
        question="Who won?",
        query_2="follow up",
        context=["old", "shared"],
    )

    result = module.FIXED_OPERATORS["retrieval2"](payload, retriever=retriever)

    assert retriever.calls == [("follow up", 3, "text-embedding-v3")]
    assert result.context == ["old", "shared", "new"]


def test_context_query_falls_back_when_llm_returns_empty_text():
    load_environment()
    adapter = RecordingAdapter("")
    payload = StatePayload(
        question="Who won?", query_1="race winner", context=["A race happened"]
    )

    result = TierRegistry.require("context_question_to_query", 3)(
        payload, adapter=adapter
    )

    assert result.query_2 == "race winner"


def test_retrieval_failure_is_not_silently_converted_to_empty_context():
    module = load_environment()

    class FailingRetriever:
        def retrieve(self, *args, **kwargs):
            raise ConnectionError("offline")

    with pytest.raises(ConnectionError, match="offline"):
        module.FIXED_OPERATORS["retrieval1"](
            StatePayload(question="Who won?"), retriever=FailingRetriever()
        )


def test_qa_evaluation_is_deterministic_token_f1():
    module = load_environment()
    payload = StatePayload(
        question="Who won?",
        answer="The Alice!",
        ground_truth="Alice",
    )

    result = module.FIXED_OPERATORS["qa_evaluation"](payload)

    assert result.evaluation == {"f1_score": 1.0, "answer": "The Alice!"}


def test_required_inputs_fail_explicitly():
    module = load_environment()

    with pytest.raises(ValueError, match="question"):
        module.FIXED_OPERATORS["retrieval1"](StatePayload())
    with pytest.raises(ValueError, match="ground_truth"):
        module.FIXED_OPERATORS["qa_evaluation"](
            StatePayload(question="Who won?", answer="Alice")
        )
