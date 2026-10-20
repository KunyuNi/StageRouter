from __future__ import annotations

from RAG.rl.dqn_replay import DeterministicRouteSchedule


def _routes():
    return {
        qid: tuple(f"route_{route}" for route in range(8))
        for qid in range(32)
    }


def test_schedule_is_deterministic_balanced_and_exactly_resumable():
    first = DeterministicRouteSchedule(_routes(), seed=1145)
    second = DeterministicRouteSchedule(_routes(), seed=1145)

    batch_a = first.next_batch()
    batch_b = second.next_batch()
    assert batch_a == batch_b
    assert len(batch_a) == 64
    assert len({qid for qid, _route in batch_a}) == 16
    for qid in {qid for qid, _route in batch_a}:
        assert len({route for row_qid, route in batch_a if row_qid == qid}) == 4

    state = first.state_dict()
    expected_next = first.next_batch()
    restored = DeterministicRouteSchedule(_routes(), seed=999)
    restored.load_state_dict(state)
    assert restored.next_batch() == expected_next
    assert restored.identity == first.identity
