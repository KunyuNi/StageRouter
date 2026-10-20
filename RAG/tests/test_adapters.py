from RAG.env._shared.llm_adapter import (
    LLMResult,
    QwenAdapter,
    _request_options,
    get_adapter,
    set_adapter,
)
from RAG.env._shared.retriever import (
    HttpRetriever,
    RetrievalItem,
    RetrievalResponse,
    get_retriever,
    set_retriever,
)
from RAG.retrieval.embedding import TrackedOpenAIEmbedding


class FakeAdapter:
    def call(self, model, prompt, **options):
        return LLMResult(text="ok", input_tokens=1, output_tokens=1)


class FakeRetriever:
    def retrieve(self, query, top_k=3, embedding_model="text-embedding-v3"):
        return RetrievalResponse(
            query=query,
            results=[
                RetrievalItem(
                    chunk_id="fake:0",
                    text=query,
                    score=1.0,
                    rank=1,
                    metadata={},
                )
            ],
        )


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        import json

        return json.dumps(self.payload).encode("utf-8")


def test_adapter_can_be_replaced_without_network():
    previous = get_adapter()
    fake = FakeAdapter()
    try:
        set_adapter(fake)
        assert get_adapter() is fake
    finally:
        set_adapter(previous)


def test_retriever_can_be_replaced_without_network():
    previous = get_retriever()
    fake = FakeRetriever()
    try:
        set_retriever(fake)
        assert get_retriever() is fake
    finally:
        set_retriever(previous)


def test_http_retriever_request_uses_fixed_embedding_model():
    retriever = HttpRetriever(base_url="http://example.test/")

    request = retriever.build_request("query", top_k=3)

    assert retriever.base_url == "http://example.test"
    assert request == {
        "query": "query",
        "top_k": 3,
        "embedding_model": "text-embedding-v3",
    }


def test_http_retriever_preserves_score_rank_metadata_and_embedding_usage(monkeypatch):
    payload = {
        "query": "rewritten query",
        "index_identity": "idx-1",
        "embedding_usage": {"input_tokens": 4, "latency_ms": 12.5},
        "results": [
            {
                "chunk_id": "doc-0:chunk-1",
                "text": "evidence text",
                "score": 0.75,
                "rank": 1,
                "metadata": {"title": "Title", "source": "Source"},
            }
        ],
    }
    monkeypatch.setattr(
        "RAG.env._shared.retriever.urlopen",
        lambda request, timeout: FakeHttpResponse(payload),
    )
    response = HttpRetriever("http://retriever.test").retrieve(
        "rewritten query", 3
    )
    assert response.index_identity == "idx-1"
    assert response.embedding_input_tokens == 4
    assert response.embedding_latency_ms == 12.5
    assert response.results[0].chunk_id == "doc-0:chunk-1"
    assert response.results[0].score == 0.75
    assert response.results[0].rank == 1
    assert response.results[0].metadata["title"] == "Title"


def test_qwen_adapter_constructor_has_no_network_side_effects():
    adapter = QwenAdapter(
        api_key="test-key",
        base_url="https://example.test/v1",
    )

    assert adapter.api_key == "test-key"
    assert adapter.base_url == "https://example.test/v1"


def test_llm_and_embedding_clients_default_to_one_sdk_retry():
    adapter = QwenAdapter(api_key="test-key")
    embedding = TrackedOpenAIEmbedding(api_key="test-key")

    assert adapter.sdk_max_retries == 1
    assert embedding.sdk_max_retries == 1


def test_qwen_adapter_uses_nl2sql_compatible_environment(monkeypatch):
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "shared-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://shared.example.test/v1/")

    adapter = QwenAdapter()

    assert adapter.api_key == "shared-key"
    assert adapter.base_url == "https://shared.example.test/v1"


def test_qwen_adapter_loads_repository_root_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=file-key\nLLM_BASE_URL=https://file.example.test/v1\n",
        encoding="utf-8",
    )
    for name in (
        "RAG_LLM_API_KEY",
        "RAG_LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_ENV_FILE", str(tmp_path / "ignored.env"))
    monkeypatch.setattr(
        "RAG.env._shared.llm_adapter.__file__",
        str(tmp_path / "RAG" / "env" / "_shared" / "llm_adapter.py"),
    )

    adapter = QwenAdapter()

    assert adapter.api_key == "file-key"
    assert adapter.base_url == "https://file.example.test/v1"


def test_embedding_adapter_loads_repository_root_env_file(monkeypatch, tmp_path):
    from RAG.retrieval.embedding import _load_runtime_env

    (tmp_path / ".env").write_text("LLM_API_KEY=embedding-test-key\n", encoding="utf-8")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("RAG_ENV_FILE", str(tmp_path / "ignored.env"))
    monkeypatch.setattr(
        "RAG.retrieval.embedding.__file__",
        str(tmp_path / "RAG" / "retrieval" / "embedding.py"),
    )

    _load_runtime_env()

    import os

    assert os.getenv("LLM_API_KEY") == "embedding-test-key"

