"""Canonical model choices for the independent RAG environment."""

from __future__ import annotations

from typing import Dict

TIER_LLM_MODELS: Dict[int, str] = {
    1: "qwen-plus",
    2: "qwen3.5-flash",
    3: "qwen-flash",
}

DEFAULT_LLM_MODEL = TIER_LLM_MODELS[1]
EMBEDDING_MODEL = "text-embedding-v3"

# Embeddings are a fixed part of the RAG retrieval workflow (they are not a
# selectable tier action), but their input-token cost still belongs in the
# route-level accounting.  Keep this beside the LLM price table so all
# operators use one canonical value.
EMBEDDING_PRICING_DOLLAR_PER_1M: Dict[str, float] = {
    EMBEDDING_MODEL: 0.50,
}

LLM_PRICING_DOLLAR_PER_1M: Dict[str, Dict[str, float]] = {
    "qwen-plus": {"input": 0.80, "output": 2.00},
    "qwen3.5-flash": {"input": 0.20, "output": 2.00},
    "qwen-flash": {"input": 0.15, "output": 1.50},
}


def tier_model(tier: int) -> str:
    """Return the configured physical LLM model for a 1-indexed tier."""
    try:
        return TIER_LLM_MODELS[int(tier)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported RAG LLM tier: {tier!r}") from exc
