"""
0_old — Keyword Extraction (旧代码适配 StatePayload)

从旧 pipeline_functions/keyword_extraction.py 提取核心 LLM 调用逻辑，
输入/输出改用 StatePayload 传递。
"""

import logging
from typing import Any, Dict

# 旧代码依赖
from llm.utils import get_prompt_engine_parser
from llm.models import llm_chain
from _shared.model_config import DEFAULT_LLM_MODEL


def keyword_extraction_tier1(payload: "StatePayload", model: str = DEFAULT_LLM_MODEL) -> "StatePayload":
    """
    S1 Keyword Extraction / 意图提炼

    单次 LLM 调用 + 旧调优 prompt 模板。
    Model is supplied by the caller or _shared.model_config.
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting keyword extraction (model=%s)", model)

    llm_config = {
        "model": model,
        "temperature": 0.0,
    }
    request_kwargs = {
        "HINT": payload.evidence or "",
        "QUESTION": payload.question or "",
    }

    prompt, engine, parser = get_prompt_engine_parser(
        node_name="keyword_extraction",
        llm_config=llm_config,
    )

    output, token_usage = llm_chain(
        prompt=prompt,
        engine=engine,
        parser=parser,
        request_kwargs=request_kwargs,
    )

    payload.keywords = output if isinstance(output, list) else []

    # 调试: 看 token_usage 实际格式
    logging.info("[0_old] token_usage type=%s, keys=%s, value=%s",
                 type(token_usage).__name__,
                 list(token_usage.keys()) if isinstance(token_usage, dict) else
                 (len(token_usage) if isinstance(token_usage, (tuple, list)) else "?"),
                 str(token_usage)[:200])

    # 手动记录成本 (旧 llm_chain 不走 QwenAdapter)
    payload.cost_tracker.record_llm_call(
        stage="keyword_extraction",
        model=token_usage.get("model_name", model) if isinstance(token_usage, dict) else model,
        input_tokens=token_usage.get("prompt_tokens", 0) if isinstance(token_usage, dict) else (token_usage[0] if isinstance(token_usage, tuple) else 0),
        output_tokens=token_usage.get("completion_tokens", 0) if isinstance(token_usage, dict) else (token_usage[1] if isinstance(token_usage, tuple) else 0),
        latency_ms=0.0,
    )

    logging.info("[0_old] Keyword extraction done: %d keywords", len(payload.keywords))
    return payload


