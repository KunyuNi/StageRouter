"""Central model selection and pricing for nl2sql.

Change model choices here instead of editing runner scripts or operators.
"""

from __future__ import annotations

from typing import Dict

TIER_LLM_MODELS: Dict[int, str] = {
    1: "qwen-plus",
    2: "qwen3.5-flash",
    3: "qwen-flash",
}

DEFAULT_LLM_MODEL = TIER_LLM_MODELS[1]
EMBEDDING_MODEL = "text-embedding-v3"

LLM_PRICING_DOLLAR_PER_1M: Dict[str, Dict[str, float]] = {
    "qwen-plus": {"input": 0.80, "output": 2.00},
    "qwen3.5-flash": {"input": 0.20, "output": 2.00},
    "qwen-flash": {"input": 0.15, "output": 1.50},
}

EMBEDDING_PRICING_DOLLAR_PER_1M: Dict[str, float] = {
    EMBEDDING_MODEL: 0.5,
}


def tier_model(tier: int) -> str:
    """Return the canonical LLM model name for a 1-indexed tier."""
    return TIER_LLM_MODELS[int(tier)]


def all_configured_llm_models() -> list[str]:
    """Return the canonical LLM models configured for T1/T2/T3."""
    return list(dict.fromkeys(TIER_LLM_MODELS.values()))
