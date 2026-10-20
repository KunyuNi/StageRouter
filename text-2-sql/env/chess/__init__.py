"""
0_old — 旧 pipeline_functions/ 算子 + StatePayload/CostTracker 新架构

每个阶段只有一个函数（单次高质量 LLM 调用），不做 T1/T2/T3 分层。
作为 ARTIST 2.0 的 baseline 版本。

共享基础设施从 new/ 导入:
  - StatePayload, CostTracker (统一 I/O + 成本追踪)
  - QwenAdapter, llm_chain, track_cost (LLM 调用 + 装饰器)
  - TierRegistry (统一注册调度, 可选)
"""

import inspect
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_FUNCTIONS_ROOT = os.path.dirname(_SCRIPT_DIR)
for p in [_PIPELINE_FUNCTIONS_ROOT, os.path.dirname(_PIPELINE_FUNCTIONS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from _shared.state_payload import StatePayload, CostTracker, QWEN_PRICING
from _shared.llm_adapter import get_adapter, track_cost, start_cost_context, end_cost_context
from _shared.model_config import TIER_LLM_MODELS

# Import adapted operators
from .keyword_extraction import keyword_extraction_tier1
from .entity_retrieval import entity_retrieval_tier1
from .context_retrieval import context_retrieval_tier1
from .column_filtering import column_filtering_tier1
from .table_selection import table_selection_tier1
from .column_selection import column_selection_tier1
from .candidate_generation import candidate_generation_tier1
from .revision import revision_tier1
from .evaluation import evaluation_tier1

stage_functions = {
    "keyword_extraction": keyword_extraction_tier1,
    "entity_retrieval": entity_retrieval_tier1,
    "context_retrieval": context_retrieval_tier1,
    "column_filtering": column_filtering_tier1,
    "table_selection": table_selection_tier1,
    "column_selection": column_selection_tier1,
    "candidate_generation": candidate_generation_tier1,
    "revision": revision_tier1,
    "evaluation": evaluation_tier1,
}


def _fixed_model(func, model_name):
    parameters = inspect.signature(func).parameters

    def wrapped(payload, **kwargs):
        call_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        if "model" in parameters:
            call_kwargs["model"] = model_name
        return func(payload, **call_kwargs)

    wrapped.__name__ = f"{func.__name__}_{model_name.replace('-', '_')}"
    wrapped.__signature__ = inspect.signature(func)
    wrapped.__new2_source__ = func.__module__
    wrapped.__new2_model__ = model_name
    return wrapped


from env._shared.tier_registry import TierRegistry

for _stage_name, _function in stage_functions.items():
    for _tier, _model_name in TIER_LLM_MODELS.items():
        TierRegistry.register(_stage_name, _tier, _fixed_model(_function, _model_name))

__all__ = [
    "StatePayload", "CostTracker", "QWEN_PRICING",
    "get_adapter", "track_cost",
    "keyword_extraction_tier1", "entity_retrieval_tier1",
    "context_retrieval_tier1", "column_filtering_tier1",
    "table_selection_tier1", "column_selection_tier1",
    "candidate_generation_tier1", "revision_tier1",
    "evaluation_tier1",
]


