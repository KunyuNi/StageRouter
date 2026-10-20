"""
0_old — Revision (旧代码适配 StatePayload)

从旧 pipeline_functions/revision.py 提取核心逻辑:
  执行预测 SQL → 发现错误 → LLM 修正 SQL。
"""

import difflib
import logging
from typing import Dict, List, Tuple

from runner.database_manager import DatabaseManager
from database_utils.schema import DatabaseSchema
from llm.utils import get_prompt_engine_parser
from llm.models import llm_chain
from _shared.model_config import DEFAULT_LLM_MODEL


def revision_tier1(payload: "StatePayload", db_manager=None,
                    model: str = DEFAULT_LLM_MODEL) -> "StatePayload":
    """
    S8 Revision / 语法自修护栏

    执行预测 SQL → 找出错误实体 → LLM 修正 SQL。
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting SQL revision")

    db_id = payload.db_id or ""
    final_sql = payload.final_sql or ""
    similar_values = payload.similar_values or {}
    schema_with_descriptions = payload.schema_with_descriptions or {}

    if not final_sql or not db_id:
        logging.warning("[0_old] Revision skipped (no SQL or db_id)")
        return payload

    # 完整 schema (用于修正)
    try:
        complete_schema = DatabaseManager(db_id=db_id).get_db_schema()
    except Exception:
        complete_schema = {}

    schema_string = DatabaseManager(db_id=db_id).get_database_schema_string(
        complete_schema,
        similar_values,
        schema_with_descriptions,
        include_value_description=True,
    )

    llm_config = {
        "model": model,
        "temperature": 0.0,
    }
    prompt, engine, parser = get_prompt_engine_parser(
        node_name="revision",
        llm_config=llm_config,
        schema_string=schema_string,
    )

    # 执行预测 SQL 获取错误信息
    try:
        query_result = DatabaseManager(db_id=db_id).validate_sql_query(
            sql=final_sql, meta_time_out=10
        ).get("RESULT", "")
    except Exception as e:
        query_result = str(e)

    # 发现错误实体
    try:
        missing_entities = _find_wrong_entities(final_sql, similar_values, db_id)
    except Exception:
        missing_entities = {}

    request_kwargs = {
        "SQL": final_sql,
        "QUESTION": payload.question or "",
        "MISSING_ENTITIES": missing_entities,
        "EVIDENCE": payload.evidence or "",
        "QUERY_RESULT": query_result,
    }

    output, token_usage = llm_chain(
        prompt=prompt,
        engine=engine,
        parser=parser,
        request_kwargs=request_kwargs,
    )

    revised_sql = output.get("revised_SQL", final_sql) if isinstance(output, dict) else final_sql
    payload.final_sql = revised_sql
    payload.revision_count = 1

    # 记录成本
    payload.cost_tracker.record_llm_call(
        stage="revision",
        model=token_usage.get("model_name", model) if isinstance(token_usage, dict) else model,
        input_tokens=token_usage.get("prompt_tokens", 0) if isinstance(token_usage, dict) else 0,
        output_tokens=token_usage.get("completion_tokens", 0) if isinstance(token_usage, dict) else 0,
        latency_ms=0.0,
    )

    logging.info("[0_old] Revision done")
    return payload


def _find_wrong_entities(sql: str, similar_values: Dict, db_id: str,
                         similarity_threshold: float = 0.4) -> str:
    wrong_entities = ""
    try:
        used_entities = DatabaseManager(db_id=db_id).get_sql_condition_literals(sql)
        similar_values_schema = DatabaseSchema.from_schema_dict_with_examples(similar_values)
        for table_name, column_info in used_entities.items():
            for column_name, column_values in column_info.items():
                target_info = similar_values_schema.get_column_info(table_name, column_name)
                if not target_info:
                    continue
                for value in column_values:
                    if value not in target_info.examples:
                        most_similar, similarity = _find_most_similar(value, target_info.examples)
                        if similarity > similarity_threshold:
                            wrong_entities += (
                                f"Column {column_name} in table {table_name} does not contain "
                                f"the value '{value}'. The correct value is '{most_similar}'.\n"
                            )
    except Exception:
        pass
    return wrong_entities


def _find_most_similar(target_value: str, candidates: List[str]) -> Tuple[str, float]:
    if not candidates:
        return "", 0.0
    best = max(candidates, key=lambda v: difflib.SequenceMatcher(None, v, target_value).ratio())
    return best, difflib.SequenceMatcher(None, best, target_value).ratio()


