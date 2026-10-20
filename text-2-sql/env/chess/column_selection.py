"""
0_old — Column Selection (旧代码适配 StatePayload)

从旧 pipeline_functions/column_selection.py 提取核心逻辑:
  富 schema 字符串 + LLM 选择列 + 聚合去重。
"""

import logging
from typing import Dict, List, Any

from runner.database_manager import DatabaseManager
from llm.utils import get_prompt_engine_parser
from llm.models import llm_chain
from _shared.model_config import DEFAULT_LLM_MODEL


def column_selection_tier1(payload: "StatePayload", db_manager=None,
                            model: str = DEFAULT_LLM_MODEL) -> "StatePayload":
    """
    S6 Column Selection / 确认 SELECT 字段

    富 schema 字符串 + LLM 选列 + 大小写去重聚合。
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting column selection")

    db_id = payload.db_id or ""
    question = payload.question or ""
    evidence = payload.evidence or ""
    tentative_schema = payload.tentative_schema or {}
    similar_values = payload.similar_values or {}
    schema_with_descriptions = payload.schema_with_descriptions or {}

    if not db_id or not tentative_schema:
        logging.warning("[0_old] Column selection skipped (no db_id or schema)")
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
        node_name="column_selection",
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

    selected_tables = list(tentative_schema.keys())
    column_names = _aggregate_columns([output], selected_tables)

    payload.tentative_schema = column_names

    # 记录成本
    payload.cost_tracker.record_llm_call(
        stage="column_selection",
        model=token_usage.get("model_name", model) if isinstance(token_usage, dict) else model,
        input_tokens=token_usage.get("prompt_tokens", 0) if isinstance(token_usage, dict) else 0,
        output_tokens=token_usage.get("completion_tokens", 0) if isinstance(token_usage, dict) else 0,
        latency_ms=0.0,
    )

    logging.info("[0_old] Column selection done: %d columns",
                 sum(len(c) for c in column_names.values()))
    return payload


def _aggregate_columns(columns_dicts: List[Dict], selected_tables: List[str]) -> Dict[str, List[str]]:
    columns = {}
    for column_dict in columns_dicts:
        for key, value in column_dict.items():
            if key == "chain_of_thought_reasoning":
                continue
            table_name = key.strip("`")
            column_names = value
            if table_name.lower() in [t.lower() for t in selected_tables]:
                for column_name in column_names:
                    column_name = column_name.strip("`")
                    columns.setdefault(table_name, [])
                    if column_name.lower() not in [c.lower() for c in columns[table_name]]:
                        columns[table_name].append(column_name)
    return columns


