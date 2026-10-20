from __future__ import annotations

import json

from RAG.rl.route_result_cache import RouteResultCache


def _record(qid: int) -> dict:
    return {
        "split": "train",
        "qid": qid,
        "question_id": qid,
        "route_key": "2-2-2-2",
        "actions": [2, 2, 2, 2],
        "operator_version": 1,
        "workflow_identity": "workflow-a",
        "status": "ok",
    }


def test_append_many_persists_every_record_and_reloads_index(tmp_path):
    path = tmp_path / "results.jsonl"
    cache = RouteResultCache(path)

    cache.append_many([_record(11), _record(12), _record(13)])

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["qid"] for row in lines] == [11, 12, 13]
    reloaded = RouteResultCache(path)
    for qid in (11, 12, 13):
        assert reloaded.get(
            split="train",
            qid=qid,
            route_key="2-2-2-2",
            operator_version=1,
            workflow_identity="workflow-a",
        )["status"] == "ok"
