from RAG.retrieval import service


def test_embedding_rate_limit_is_exposed_as_http_429():
    error = RuntimeError("limit_requests from text-embedding-v3")

    assert service._provider_http_status(error) == 429
    assert service._provider_http_status(RuntimeError("provider crashed")) is None
