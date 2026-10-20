"""
0_old — Column Filtering (旧代码适配 StatePayload)

从旧 pipeline_functions/column_filtering.py 提取核心逻辑:
  逐列 LLM 二分类 + FK 连通性补充。
"""

import logging
import asyncio
from typing import Dict, Any

from runner.database_manager import DatabaseManager
from llm.utils import get_prompt_engine_parser
from llm.models import async_llm_chain_call
from _shared.model_config import DEFAULT_LLM_MODEL


def column_filtering_tier1(payload: "StatePayload", db_manager=None,
                              model: str = DEFAULT_LLM_MODEL) -> "StatePayload":
    """
    S4 Column Filtering / 列级粗筛

    对每列做 LLM 二分类 (is_column_information_relevant).
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting column filtering")

    db_id = payload.db_id or ""
    question = payload.question or ""
    evidence = payload.evidence or ""

    if not db_id:
        logging.warning("[0_old] Column filtering skipped (no db_id)")
        return payload

    # 获取列 profiles (含值描述、key 信息)
    schema_with_examples = payload.similar_values or {}
    column_profiles = DatabaseManager(db_id=db_id).get_column_profiles(
        schema_with_examples=schema_with_examples,
        use_value_description=True,
        with_keys=True,
        with_references=True,
    )

    list_of_kwargs = []
    for table_name, columns in column_profiles.items():
        for column_name, column_profile in columns.items():
            kwargs = {
                "QUESTION": question,
                "HINT": evidence,
                "COLUMN_PROFILE": column_profile,
            }
            list_of_kwargs.append(kwargs)

    llm_config = {
        "model": model,
        "temperature": 0.0,
    }
    prompt, engine, parser = get_prompt_engine_parser(
        node_name="column_filtering",
        llm_config=llm_config,
    )

    try:
        response, token_usage = asyncio.run(async_llm_chain_call(
            prompt=prompt,
            engine=engine,
            parser=parser,
            request_list=list_of_kwargs,
            step="column_filtering",
            sampling_count=1,
        ))
    except RuntimeError:
        # 如果在已有 event loop 的环境 (如 Jupyter/某些测试框架) 中运行
        import concurrent.futures
        loop = asyncio.new_event_loop()
        try:
            response, token_usage = loop.run_until_complete(
                async_llm_chain_call(
                    prompt=prompt, engine=engine, parser=parser,
                    request_list=list_of_kwargs, step="column_filtering",
                    sampling_count=1,
                )
            )
        finally:
            loop.close()

    # 解析每列的二分类结果
    choices = []
    for col_choice in response:
        if isinstance(col_choice, list) and len(col_choice) > 0:
            choice_val = col_choice[0].get("is_column_information_relevant", "no")
        elif isinstance(col_choice, dict):
            choice_val = col_choice.get("is_column_information_relevant", "no")
        else:
            choice_val = "no"
        choices.append(str(choice_val).strip().lower())

    index = 0
    tentative_schema = {}
    for table_name, columns in column_profiles.items():
        tentative_schema[table_name] = []
        for column_name, column_profile in columns.items():
            try:
                chosen = choices[index]
                if chosen in ("yes", "true", "1"):
                    tentative_schema[table_name].append(column_name)
            except (IndexError, Exception):
                pass
            index += 1

    # FK 连通性增强 (from old code)
    try:
        tentative_schema = DatabaseManager(db_id=db_id).add_connections_to_tentative_schema(
            tentative_schema)
    except Exception:
        pass

    payload.tentative_schema = tentative_schema

    # 记录成本
    payload.cost_tracker.record_llm_call(
        stage="column_filtering",
        model=token_usage.get("model_name", model) if isinstance(token_usage, dict) else model,
        input_tokens=token_usage.get("prompt_tokens", 0) if isinstance(token_usage, dict) else 0,
        output_tokens=token_usage.get("completion_tokens", 0) if isinstance(token_usage, dict) else 0,
        latency_ms=0.0,
    )

    logging.info("[0_old] Column filtering done: %d tables, %d columns",
                 len(tentative_schema), sum(len(c) for c in tentative_schema.values()))
    return payload


