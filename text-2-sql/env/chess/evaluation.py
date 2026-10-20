"""
0_old — Evaluation (旧代码适配 StatePayload)

从旧 pipeline_functions/evaluation.py 提取核心逻辑:
  执行 predicted_sql vs gold_sql 对比。
固定执行节点，不参与 Tier 路由。
"""

import errno
import logging
from typing import Any

from runner.database_manager import DatabaseManager


def evaluation_tier1(payload: "StatePayload", db_manager=None) -> "StatePayload":
    """
    S9 Evaluation / 沙盒评估

    执行 predicted_sql 与 gold_sql 的结果对比。
    """
    from _shared.state_payload import StatePayload

    logging.info("[0_old] Starting evaluation")

    predicted_sql = payload.final_sql or ""
    gold_sql = payload.gold_sql or ""
    db_id = payload.db_id or ""

    if not predicted_sql or not gold_sql:
        payload.evaluation = {
            "exec_res": 0,
            "exec_err": "missing SQL (predicted or gold)",
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql,
        }
        logging.warning("[0_old] Evaluation skipped (missing SQL)")
        return payload

    try:
        result = DatabaseManager(db_id=db_id).compare_sqls(
            predicted_sql=predicted_sql,
            ground_truth_sql=gold_sql,
            meta_time_out=120,
        )
        payload.evaluation = {
            "exec_res": result.get("exec_res", 0),
            "exec_err": result.get("exec_err", "unknown"),
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql,
        }
    except Exception as e:
        if isinstance(e, OSError) and e.errno in {errno.EMFILE, errno.ENFILE}:
            raise
        if "unable to open database file" in str(e).lower():
            raise
        payload.evaluation = {
            "exec_res": 0,
            "exec_err": str(e),
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql,
        }

    logging.info("[0_old] Evaluation done: EX=%s",
                 payload.evaluation.get("exec_res"))
    return payload


