"""
0_old — Context Retrieval (旧代码适配 StatePayload)

从旧 pipeline_functions/context_retrieval.py 提取核心逻辑:
  对每个 keyword 查询 Chroma 向量数据库，收集列/表描述。
纯非 LLM 阶段 (向量检索 + 去重聚合)。
"""

import logging
from typing import Dict, List, Any
from runner.database_manager import DatabaseManager


def context_retrieval_tier1(payload: "StatePayload", db_manager=None) -> "StatePayload":
    """
    S3 Context Retrieval / 查数据库注释

    关键词 → Chroma 向量检索 × evidence/question query → 去重聚合。
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting context retrieval")

    db_id = payload.db_id or ""
    keywords = payload.keywords
    question = payload.question or ""
    evidence = payload.evidence or ""

    if not keywords or not db_id:
        logging.warning("[0_old] Context retrieval skipped (no keywords or db_id)")
        return payload

    top_k = 5  # 默认值

    retrieved = _find_most_similar_columns_impl(
        db_id=db_id,
        question=question,
        evidence=evidence,
        keywords=keywords,
        top_k=top_k,
    )

    schema_with_descriptions = _format_retrieved_descriptions(retrieved)
    payload.schema_with_descriptions = schema_with_descriptions

    logging.info("[0_old] Context retrieval done: %d tables",
                 len(schema_with_descriptions))
    return payload


# === 以下函数从旧 context_retrieval.py 提取 ===

def _find_most_similar_columns_impl(
    db_id: str, question: str, evidence: str,
    keywords: List, top_k: int
) -> Dict:
    tables_with_descriptions = {}
    for keyword in keywords:
        kw_val = keyword.get("value", keyword) if isinstance(keyword, dict) else keyword
        question_based_query = f"{question} {kw_val}"
        evidence_based_query = f"{evidence} {kw_val}"

        retrieved_q = DatabaseManager(db_id=db_id).query_vector_db(
            question_based_query, top_k=top_k)
        retrieved_e = DatabaseManager(db_id=db_id).query_vector_db(
            evidence_based_query, top_k=top_k)

        tables_with_descriptions = _add_description(
            tables_with_descriptions, retrieved_q)
        tables_with_descriptions = _add_description(
            tables_with_descriptions, retrieved_e)
    return tables_with_descriptions


def _add_description(tables_with_descriptions, retrieved_descriptions):
    for table_name, column_descriptions in retrieved_descriptions.items():
        if table_name not in tables_with_descriptions:
            tables_with_descriptions[table_name] = {}
        for column_name, desc in column_descriptions.items():
            if (column_name not in tables_with_descriptions[table_name] or
                    desc.get("score", 0) > tables_with_descriptions[table_name].get(column_name, {}).get("score", 0)):
                tables_with_descriptions[table_name][column_name] = desc
    return tables_with_descriptions


def _format_retrieved_descriptions(retrieved_columns):
    for column_descriptions in retrieved_columns.values():
        for column_info in column_descriptions.values():
            column_info.pop("score", None)
    return retrieved_columns


