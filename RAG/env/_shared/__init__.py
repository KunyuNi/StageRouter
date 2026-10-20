"""Shared contracts for RAG environments."""

from .model_config import (
    EMBEDDING_MODEL,
    EMBEDDING_PRICING_DOLLAR_PER_1M,
    TIER_LLM_MODELS,
    tier_model,
)
from .llm_adapter import LLMAdapter, LLMResult, get_adapter, set_adapter
from .retriever import (
    HttpRetriever,
    RetrievalItem,
    RetrievalResponse,
    Retriever,
    get_retriever,
    set_retriever,
)
from .stage_action_registry import (
    SKIP_ACTION,
    T1_ACTION,
    T2_ACTION,
    T3_ACTION,
    StageActionRegistry,
)
from .state_payload import CostTracker, StatePayload
from .tier_registry import TierRegistry

__all__ = [
    "CostTracker",
    "EMBEDDING_MODEL",
    "EMBEDDING_PRICING_DOLLAR_PER_1M",
    "HttpRetriever",
    "LLMAdapter",
    "LLMResult",
    "Retriever",
    "RetrievalItem",
    "RetrievalResponse",
    "SKIP_ACTION",
    "StatePayload",
    "StageActionRegistry",
    "T1_ACTION",
    "T2_ACTION",
    "T3_ACTION",
    "TIER_LLM_MODELS",
    "TierRegistry",
    "get_adapter",
    "get_retriever",
    "set_adapter",
    "set_retriever",
    "tier_model",
]
