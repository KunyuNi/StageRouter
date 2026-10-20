from __future__ import annotations

import numpy as np

from RAG.rl.dqn_replay import DeterministicRouteSchedule, OfflineTransition
from RAG.rl.prepared_artifacts import PreparedReplayDataset, PreparedReplayStore
from RAG.rl.state_v1_encoder import EncodedState
from RAG.rl.state_v1_schema import STATE_V1_DIM


def _state(value: float, decision_idx: int) -> EncodedState:
    vector = np.full(STATE_V1_DIM, value, dtype=np.float32)
    legal = np.asarray([1, 1, 1, decision_idx < 3], dtype=np.bool_)
    return EncodedState(
        vector,
        legal,
        f"logical-{value}-{decision_idx}",
        f"tensor-{value}-{decision_idx}",
    )


def _trajectory(qid: int, route: str, offset: float):
    states = [_state(offset + index, index) for index in range(4)]
    return tuple(
        OfflineTransition(
            qid=qid,
            route_key=route,
            trajectory_identity=route,
            decision_idx=index,
            action=index % 3,
            state=states[index],
            next_state=states[index + 1] if index < 3 else None,
            reward_scalar=0.1,
            reward_f1=0.2,
            reward_cost=0.3,
            mc_scalar=0.4,
            mc_f1=0.5,
            mc_cost=0.6,
            terminal=index == 3,
            final_f1=0.7,
            remaining_cost=0.001,
            actions=(0, 1, 2, 0),
        )
        for index in range(4)
    )


def test_prepared_replay_store_deduplicates_states_and_loads_schedule_batches(tmp_path):
    path = tmp_path / "offline_replay.db"
    with PreparedReplayStore(path, identity="prepared-v1") as store:
        for qid in range(1, 3):
            for route_idx in range(4):
                store.put_trajectory(
                    _trajectory(qid, f"route-{route_idx}", float(qid * 10 + route_idx))
                )
        store.commit()
        assert store.trajectory_count() == 8
        assert store.transition_count() == 32

    routes = {qid: [f"route-{index}" for index in range(4)] for qid in range(1, 3)}
    schedule = DeterministicRouteSchedule(
        routes, seed=1145, qids_per_step=2, routes_per_qid=4
    )
    with PreparedReplayStore(path, identity="prepared-v1", readonly=True) as store:
        replay = PreparedReplayDataset(store, schedule)
        rows = replay.next_batch()
        assert len(rows) == 32
        assert len({(row.qid, row.route_key) for row in rows}) == 8
        assert all(row.state.vector.dtype == np.float32 for row in rows)
        assert (
            replay.trajectories[(1, "route-0")][0].state.tensor_state_hash
            == "tensor-10.0-0"
        )


def test_prepared_store_rejects_identity_mismatch(tmp_path):
    path = tmp_path / "offline_replay.db"
    with PreparedReplayStore(path, identity="one"):
        pass
    try:
        PreparedReplayStore(path, identity="two", readonly=True)
    except ValueError as exc:
        assert "identity mismatch" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("identity mismatch was accepted")
