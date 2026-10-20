"""Check the decoding settings passed through the workflow's LLM adapter."""

from env.llm import models, utils


def test_prompt_configuration_reaches_model_constructor(monkeypatch):
    seen = {}

    def make_model(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setitem(
        models.ENGINE_CONFIGS,
        "qwen-flash",
        {"constructor": make_model, "params": {"model": "qwen-flash"}},
    )
    monkeypatch.setattr(utils, "get_prompt", lambda *_args, **_kwargs: "prompt")
    monkeypatch.setattr(utils, "get_parser", lambda *_args: None)
    monkeypatch.setenv("NL2SQL_SDK_MAX_RETRIES", "0")

    prompt, _engine, _parser = utils.get_prompt_engine_parser(
        "keyword_extraction",
        {"model": "qwen-flash", "temperature": 0.0},
    )
    assert prompt == "prompt"
    assert seen["temperature"] == 0.0
    assert "max_tokens" not in seen
    assert seen["max_retries"] == 0
