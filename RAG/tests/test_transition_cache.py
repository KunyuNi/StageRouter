from __future__ import annotations

import pytest

from RAG.rl.transition_cache import TransitionCache, TransitionKey


def test_transition_cache_round_trip_is_exact_and_workflow_guarded(tmp_path):
    cache = TransitionCache(tmp_path / "new1.db", workflow_identity="workflow-a")
    key = TransitionKey(7, 1, 0, "input-a", 1)
    payload = {
        "question_id": 7,
        "context": ["evidence"],
        "cost_tracker": {"stage_costs": []},
    }
    cache.store(key, payload, source="fixed_route")
    assert cache.lookup(key) == payload
    assert cache.lookup(TransitionKey(7, 1, 0, "input-b", 1)) is None
    assert cache.integrity_check() == "ok"
    cache.close()
    with pytest.raises(ValueError, match="workflow identity"):
        TransitionCache(tmp_path / "new1.db", workflow_identity="workflow-b")


def test_readonly_cache_can_replay_but_cannot_store(tmp_path):
    path = tmp_path / "new1.db"
    key = TransitionKey(7, 1, 0, "input-a", 1)
    payload = {"question_id": 7, "cost_tracker": {"stage_costs": []}}
    writable = TransitionCache(path, workflow_identity="workflow-a")
    writable.store(key, payload)
    writable.close()

    readonly = TransitionCache(path, workflow_identity="workflow-a", readonly=True)
    assert readonly.lookup(key) == payload
    with pytest.raises(RuntimeError, match="read-only"):
        readonly.store(key, payload)
    readonly.close()
