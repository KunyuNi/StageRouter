"""
0_old — Candidate Generation (旧代码适配 StatePayload)

从旧 pipeline_functions/candidate_generation.py 提取核心逻辑:
  富 schema 字符串 + LLM 生成 SQL。
整个 Pipeline 最关键、最昂贵的阶段。
"""

import logging
from typing import Any

from runner.database_manager import DatabaseManager
from llm.utils import get_prompt_engine_parser
from llm.models import llm_chain
from _shared.model_config import DEFAULT_LLM_MODEL


def candidate_generation_tier1(payload: "StatePayload", db_manager=None,
                                use_mcts: bool = True,
                                model: str = DEFAULT_LLM_MODEL) -> "StatePayload":
    """
    S7 Candidate Generation / SQL 代码生成

    富 schema 字符串 + 单次高质量 LLM 调用。
    use_mcts 参数保持接口兼容 (0_old 不使用 MCTS)。
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting candidate generation")

    db_id = payload.db_id or ""
    question = payload.question or ""
    evidence = payload.evidence or ""
    tentative_schema = payload.tentative_schema or {}
    similar_values = payload.similar_values or {}
    schema_with_descriptions = payload.schema_with_descriptions or {}

    if not db_id or not tentative_schema:
        logging.warning("[0_old] Candidate generation skipped (no db_id or schema)")
        payload.final_sql = ""
        return payload

    schema_string = DatabaseManager(db_id=db_id).get_database_schema_string(
        tentative_schema,
        similar_values,
        schema_with_descriptions,
        include_value_description=True,
    )

    llm_config = {
        "model": model,
        "temperature": 0.0,
    }
    prompt, engine, parser = get_prompt_engine_parser(
        node_name="candidate_generation",
        llm_config=llm_config,
        schema_string=schema_string,
    )

    request_kwargs = {
        "HINT": evidence,
        "QUESTION": question,
    }

    output, token_usage = llm_chain(
        prompt=prompt,
        engine=engine,
        parser=parser,
        request_kwargs=request_kwargs,
    )

    sql = output.get("SQL", "") if isinstance(output, dict) else str(output)
    payload.final_sql = sql
    payload.candidates = [{"sql": sql, "strategy": "old_t1", "confidence": 0.85}]

    # 记录成本
    payload.cost_tracker.record_llm_call(
        stage="candidate_generation",
        model=token_usage.get("model_name", model) if isinstance(token_usage, dict) else model,
        input_tokens=token_usage.get("prompt_tokens", 0) if isinstance(token_usage, dict) else 0,
        output_tokens=token_usage.get("completion_tokens", 0) if isinstance(token_usage, dict) else 0,
        latency_ms=0.0,
    )

    logging.info("[0_old] Candidate generation done: SQL len=%d", len(sql))
    return payload


