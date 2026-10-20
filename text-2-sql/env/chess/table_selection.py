"""
0_old — Table Selection (旧代码适配 StatePayload)

从旧 pipeline_functions/table_selection.py 提取核心逻辑:
  get_database_schema_string() 富 schema 字符串 + LLM 选择表 + FK 补充。
"""

import logging
from typing import Dict, List, Any

from runner.database_manager import DatabaseManager
from llm.utils import get_prompt_engine_parser
from llm.models import llm_chain
from _shared.model_config import DEFAULT_LLM_MODEL


def table_selection_tier1(payload: "StatePayload", db_manager=None,
                           model: str = DEFAULT_LLM_MODEL) -> "StatePayload":
    """
    S5 Table Selection / 寻找目标基表

    富 schema 字符串 + LLM 选择 + FK 连通性增强。
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting table selection")

    db_id = payload.db_id or ""
    question = payload.question or ""
    evidence = payload.evidence or ""
    tentative_schema = payload.tentative_schema or {}
    similar_values = payload.similar_values or {}
    schema_with_descriptions = payload.schema_with_descriptions or {}

    if not db_id or not tentative_schema:
        logging.warning("[0_old] Table selection skipped (no db_id or schema)")
        return payload

    # 构建富 schema 字符串 (旧代码核心能力)
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
        node_name="table_selection",
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

    # 聚合表名
    aggregated = _aggregate_tables([output])
    table_names = aggregated.get("table_names", [])

    # 裁剪 tentative_schema 到选中表
    pruned_schema = {
        t: tentative_schema.get(t, []) for t in table_names
    }

    # FK 连通性补充
    try:
        pruned_schema = DatabaseManager(db_id=db_id).add_connections_to_tentative_schema(
            pruned_schema)
    except Exception:
        pass

    payload.tentative_schema = pruned_schema
    payload.selected_tables = table_names

    # 记录成本
    payload.cost_tracker.record_llm_call(
        stage="table_selection",
        model=token_usage.get("model_name", model) if isinstance(token_usage, dict) else model,
        input_tokens=token_usage.get("prompt_tokens", 0) if isinstance(token_usage, dict) else 0,
        output_tokens=token_usage.get("completion_tokens", 0) if isinstance(token_usage, dict) else 0,
        latency_ms=0.0,
    )

    logging.info("[0_old] Table selection done: %d tables selected", len(table_names))
    return payload


def _aggregate_tables(tables_dicts: List[Dict]) -> Dict:
    tables = []
    chain_of_thoughts = []
    for table_dict in tables_dicts:
        chain_of_thoughts.append(table_dict.get("chain_of_thought_reasoning", ""))
        for table in table_dict.get("table_names", []):
            if table.lower() not in [t.lower() for t in tables]:
                tables.append(table)
    return {
        "table_names": tables,
        "chain_of_thought_reasoning": "\n----\n".join(chain_of_thoughts),
    }


