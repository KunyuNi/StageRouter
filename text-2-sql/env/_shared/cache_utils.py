"""
冷启动缓存工具 — 嵌入测试脚本, 跑测试的同时填充 TransitionCache

设计原则:
  1. 每个算子包使用自己的缓存目录 (避免不同版本的算子输出混在一起)
  2. 零额外 LLM 调用: cache.store() 只是 SQLite INSERT (~0.5ms)
  3. 选完最优版本后, 手动 cp 到共享位置激活训练

使用:
  from _shared.cache_utils import init_caches, cache_stage
  cache = init_caches(base_dir=os.path.dirname(__file__))
  ...
  cache_stage(cache, qid, stage_idx, tier, before, after)

目录结构:
  env/chess/cache/transitions.db    <- Chess operator cache
  env/chess/cache/transitions.db          ← 0_old 独立
  env/1_new/cache/transitions.db          ← 1_new 独立
  env/2_new_combine/cache/transitions.db  ← 2_new_combine 独立

选完版本后手动激活:
  cp env/<chosen>/cache/transitions.db → nl2sql/cache/transitions.db
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 训练时使用的共享路径 (由用户手动激活)
SHARED_TRANSITION_DB = "cache/transitions.db"


def init_caches(
    base_dir: str,
    transition_db: Optional[str] = None,
):
    """
    初始化当前算子包的 TransitionCache.

    Args:
        base_dir: 算子包根目录 (通常传 os.path.dirname(__file__) 或 _SCRIPT_DIR)
        transition_db: 自定义 DB 路径 (默认 base_dir/cache/transitions.db)

    Returns:
        TransitionCache instance
    """
    from rl.transition_cache import TransitionCache

    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    db_path = transition_db or os.path.join(cache_dir, "transitions.db")

    cache = TransitionCache(db_path)

    logger.info("Cold-start cache: transitions=%s (%d entries)",
                db_path, cache.count())
    return cache


def cache_stage(
    cache,                             # TransitionCache
    question_id: int,
    stage_idx: int,
    tier: int,                         # 0-indexed (0=T1, 1=T2, 2=T3)
    before_payload,                    # StatePayload (执行前)
    after_payload,                     # StatePayload (执行后)
    operator_version: int = 1,
    source: str = "cold_start",
):
    """
    缓存单个 stage 的算子输入→输出.

    嵌入位置: 每个 pipeline stage 调用后立即调用.
    零额外 LLM: 算子已经执行过了，这里只存结果.

    注意: tier 是 0-indexed (0=T1, 1=T2, 2=T3).
    如需跳过特定 tier 的缓存 (如 non-LLM T3), 调用方自行判断不调此函数.
    """

    from rl.transition_cache import TransitionKey, compute_input_hash, compute_semantic_hash

    try:
        input_hash = compute_input_hash(before_payload.to_json_dict(), stage_idx, tier)
    except ValueError:
        return  # S9 或未知 stage

    key = TransitionKey(
        question_id=question_id,
        stage_idx=stage_idx,
        tier=tier,
        input_payload_hash=input_hash,
        operator_version=operator_version,
    )

    after_dict = after_payload.to_json_dict()
    semantic_hash = compute_semantic_hash(after_dict, stage_idx)
    cache.store(key, after_dict, source=source,
                semantic_hash=semantic_hash, stored_hash=input_hash)
