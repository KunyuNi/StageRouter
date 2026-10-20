import importlib

from RAG.env._shared.llm_adapter import LLMResult
from RAG.env._shared.state_payload import StatePayload
from RAG.env._shared.tier_registry import TierRegistry


class WorkflowLLM:
    def __init__(self):
        self.models = []

    def call(self, model, prompt, **options):
        self.models.append(model)
        if "careful QA reranker" in prompt:
            score = 95 if "Alice won" in prompt else 20
            text = f'{{"score": {score}, "justification": "evidence"}}'
        elif "follow-up search query" in prompt:
            text = "search query: race winner details"
        elif "Answer the question" in prompt:
            text = "answer: Alice"
        else:
            text = "search query: race winner"
        return LLMResult(text=text, input_tokens=5, output_tokens=2, latency_ms=1)


class WorkflowRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=3, embedding_model=""):
        self.calls.append((query, embedding_model))
        if len(self.calls) == 1:
            return ["A race was held", "shared"]
        return ["shared", "Alice won the race"]


def test_baseline_workflow_runs_with_injected_services():
    module = importlib.reload(importlib.import_module("RAG.env.0_old"))
    llm = WorkflowLLM()
    retriever = WorkflowRetriever()
    payload = StatePayload(question="Who won?", ground_truth="Alice")

    payload = TierRegistry.require("question_to_query", 2)(payload, adapter=llm)
    payload = module.FIXED_OPERATORS["retrieval1"](payload, retriever=retriever)
    payload = TierRegistry.require("context_question_to_query", 2)(
        payload, adapter=llm
    )
    payload = module.FIXED_OPERATORS["retrieval2"](payload, retriever=retriever)
    payload = TierRegistry.require("rerank", 2)(payload, adapter=llm)
    payload = TierRegistry.require("context_question_to_answer", 2)(
        payload, adapter=llm
    )
    payload = module.FIXED_OPERATORS["qa_evaluation"](payload)

    assert payload.answer == "Alice"
    assert payload.evaluation == {"f1_score": 1.0, "answer": "Alice"}
    assert payload.context[0] == "Alice won the race"
    assert [model for model in llm.models] == ["qwen3.5-flash"] * 6
    assert retriever.calls == [
        ("race winner", "text-embedding-v3"),
        ("race winner details", "text-embedding-v3"),
    ]
