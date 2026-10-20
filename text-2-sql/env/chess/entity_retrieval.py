"""
0_old — Entity Retrieval (旧代码适配 StatePayload)

从旧 pipeline_functions/entity_retrieval.py 提取核心逻辑:
  LSH recall + 编辑距离 + embedding 相似度三度量融合。
纯非 LLM 阶段，但调用了远程 embedding API (SafeOpenAIEmbeddings)。
"""

import logging
import difflib
import numpy as np
import os
import concurrent.futures
from typing import Dict, List, Any

from runner.database_manager import DatabaseManager

try:
    from llm.models import SafeOpenAIEmbeddings
    from _shared.model_config import EMBEDDING_MODEL

    EMBEDDING_FUNCTION = SafeOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
except Exception:
    EMBEDDING_FUNCTION = None


def entity_retrieval_tier1(payload: "StatePayload", db_manager=None) -> "StatePayload":
    """
    S2 Entity Retrieval / 对齐数据库值

    LSH + 编辑距离 + embedding 三度量融合。
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting entity retrieval")

    db_id = payload.db_id or ""
    keywords = payload.keywords

    if not keywords or not db_id:
        logging.warning("[0_old] Entity retrieval skipped (no keywords or db_id)")
        return payload

    similar_values = _get_similar_entities(db_id, keywords)
    payload.similar_values = similar_values

    logging.info("[0_old] Entity retrieval done: %d tables with matches",
                 len(similar_values))
    return payload


# === 以下函数从旧 entity_retrieval.py 直接复制 ===

def _get_similar_entities(db_id: str, keywords: List[str]) -> Dict:
    selected_values = {}

    def _get_similar_values_target_string(target_string: str):
        unique_similar_values = DatabaseManager(db_id=db_id).query_lsh(
            keyword=target_string, signature_size=100, top_n=10)
        return target_string, {
            table_name: {
                column_name: _get_similar_values_impl(target_string, values)
                for column_name, values in column_values.items()
            }
            for table_name, column_values in unique_similar_values.items()
        }

    for keyword in keywords:
        keyword = keyword.get("value", keyword) if isinstance(keyword, dict) else keyword
        keyword = str(keyword).strip()
        to_search_values = [keyword]
        if (" " in keyword) and ("=" not in keyword):
            parts = keyword.split()
            to_search_values.extend(parts)

        to_search_values.sort(key=len, reverse=True)
        hint_column, hint_value = _column_value(keyword)
        if hint_value:
            to_search_values = [hint_value] + to_search_values

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_get_similar_values_target_string, ts): ts
                for ts in to_search_values
            }
            for future in concurrent.futures.as_completed(futures):
                target_string, similar_vals = future.result()
                for table_name, column_values in similar_vals.items():
                    for column_name, entities in column_values.items():
                        if entities:
                            selected_values.setdefault(table_name, {}).setdefault(
                                column_name, []).extend([
                                    (ts, value, edit_distance, embedding)
                                    for ts, value, edit_distance, embedding in entities
                                ])

    for table_name, column_values in selected_values.items():
        for column_name, values in column_values.items():
            if not values:
                continue
            max_edit_distance = max(values, key=lambda x: x[2])[2]
            selected_values[table_name][column_name] = list(set(
                value for _, value, edit_distance, _ in values
                if edit_distance == max_edit_distance
            ))
    return selected_values


def _get_similar_values_impl(target_string: str, values: List[str]):
    edit_distance_threshold = 0.3
    top_k_edit_distance = 5
    embedding_similarity_threshold = 0.6
    top_k_embedding = 1

    edit_distance_similar_values = [
        (value, difflib.SequenceMatcher(None, value.lower(), target_string.lower()).ratio())
        for value in values
        if difflib.SequenceMatcher(None, value.lower(), target_string.lower()).ratio() >= edit_distance_threshold
    ]
    edit_distance_similar_values.sort(key=lambda x: x[1], reverse=True)
    edit_distance_similar_values = edit_distance_similar_values[:top_k_edit_distance]

    if not edit_distance_similar_values:
        return []

    if EMBEDDING_FUNCTION is not None:
        try:
            target_emb = EMBEDDING_FUNCTION.embed_query(target_string)
            value_texts = [v for v, _ in edit_distance_similar_values]
            all_embs = EMBEDDING_FUNCTION.embed_documents(value_texts)
            similarities = [float(np.dot(target_emb, emb)) for emb in all_embs]
        except Exception:
            similarities = [0.0] * len(edit_distance_similar_values)
    else:
        similarities = [0.0] * len(edit_distance_similar_values)

    embedding_similar_values = [
        (target_string, edit_distance_similar_values[i][0],
         edit_distance_similar_values[i][1], similarities[i])
        for i in range(len(edit_distance_similar_values))
        if similarities[i] >= embedding_similarity_threshold
    ]
    embedding_similar_values.sort(key=lambda x: x[2], reverse=True)
    return embedding_similar_values[:top_k_embedding]


def _column_value(string: str):
    if "=" in string:
        left_equal = string.find("=")
        first_part = string[:left_equal].strip()
        second_part = string[left_equal + 1:].strip() if len(string) > left_equal + 1 else None
        return first_part, second_part
    return None, None


