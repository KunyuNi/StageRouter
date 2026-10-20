from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from _shared.model_config import all_configured_llm_models
except ModuleNotFoundError:
    try:
        from env._shared.model_config import all_configured_llm_models
    except ModuleNotFoundError:
        from nl2sql.env._shared.model_config import all_configured_llm_models

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

_API_KEY = os.getenv("LLM_API_KEY")
_BASE_URL = os.getenv("LLM_BASE_URL")


def _chat_config(model_key: str) -> Dict[str, Any]:
    return {
        "constructor": ChatOpenAI,
        "params": {
            "model": model_key,
            "api_key": _API_KEY,
            "base_url": _BASE_URL,
            "max_retries": 5,
            "timeout": 120,
        },
    }


ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    model_key: _chat_config(model_key)
    for model_key in all_configured_llm_models()
}
