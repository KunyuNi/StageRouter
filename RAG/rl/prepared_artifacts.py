"""Persistent, lazy-loading artifacts for the offline RAG replay buffer."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dqn_replay import DeterministicRouteSchedule, OfflineTransition
from .state_v1_encoder import EncodedState
from .state_v1_schema import STATE_V1_DIM


class PreparedReplayStore:
    """SQLite store that deduplicates State tensors and lazily restores routes."""

    def __init__(
        self,
        path: str | Path,
        *,
        identity: str,
        readonly: bool = False,
    ) -> None:
        self.path = Path(path)
        self.identity = str(identity)
        self.readonly = bool(readonly)
        if not self.identity:
            raise ValueError("prepared replay identity is required")
        self._lock = threading.RLock()
        self._closed = False
        if readonly:
            resolved = self.path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"prepared replay not found: {resolved}")
            self._connection = sqlite3.connect(
                resolved.as_uri() + "?mode=ro", uri=True, check_same_thread=False
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                str(self.path), timeout=60.0, check_same_thread=False
            )
        self._connection.row_factory = sqlite3.Row
        if readonly:
            self._validate_identity()
        else:
            self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifact_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS states (
                    tensor_state_hash TEXT PRIMARY KEY,
                    logical_state_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    dtype TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    legal_mask BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    qid INTEGER NOT NULL,
                    route_key TEXT NOT NULL,
                    trajectory_identity TEXT NOT NULL,
                    decision_idx INTEGER NOT NULL,
                    action INTEGER NOT NULL,
                    state_hash TEXT NOT NULL,
                    next_state_hash TEXT,
                    reward_scalar REAL NOT NULL,
                    reward_f1 REAL NOT NULL,
                    reward_cost REAL NOT NULL,
                    mc_scalar REAL NOT NULL,
                    mc_f1 REAL NOT NULL,
                    mc_cost REAL NOT NULL,
                    terminal INTEGER NOT NULL,
                    final_f1 REAL NOT NULL,
                    remaining_cost REAL NOT NULL,
                    actions_json TEXT NOT NULL,
                    PRIMARY KEY (qid, route_key, decision_idx),
                    FOREIGN KEY (state_hash) REFERENCES states(tensor_state_hash),
                    FOREIGN KEY (next_state_hash) REFERENCES states(tensor_state_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_prepared_routes
                    ON transitions(qid, route_key, decision_idx);
                """
            )
            row = self._connection.execute(
                "SELECT value FROM artifact_meta WHERE key='identity'"
            ).fetchone()
            existing = str(row["value"]) if row else ""
            if existing and existing != self.identity:
                raise ValueError(
                    f"prepared replay identity mismatch: store={existing}, requested={self.identity}"
                )
            if not existing:
                self._connection.execute(
                    "INSERT INTO artifact_meta(key,value) VALUES('identity',?)",
                    (self.identity,),
                )
            self._connection.commit()

    def _validate_identity(self) -> None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM artifact_meta WHERE key='identity'"
            ).fetchone()
            existing = str(row["value"]) if row else ""
            if existing != self.identity:
                raise ValueError(
                    f"prepared replay identity mismatch: store={existing}, requested={self.identity}"
                )
            self._connection.execute("SELECT 1 FROM transitions LIMIT 1").fetchone()

    @staticmethod
    def _state_payload(state: EncodedState) -> tuple[Any, ...]:
        vector = np.asarray(state.vector, dtype=np.float32).reshape(-1)
        legal = np.asarray(state.legal_action_mask, dtype=np.bool_).reshape(-1)
        if vector.shape != (STATE_V1_DIM,) or legal.shape != (4,):
            raise ValueError("prepared State V1 has an invalid shape")
        if not np.isfinite(vector).all():
            raise ValueError("prepared State V1 contains non-finite values")
        return (
            str(state.tensor_state_hash),
            str(state.logical_state_hash),
            STATE_V1_DIM,
            "float32",
            np.asarray(vector, dtype="<f4").tobytes(order="C"),
            legal.astype(np.uint8).tobytes(order="C"),
        )

    def _put_state(self, state: EncodedState) -> None:
        payload = self._state_payload(state)
        self._connection.execute(
            """
            INSERT OR IGNORE INTO states
                (tensor_state_hash,logical_state_hash,dimension,dtype,vector,legal_mask)
            VALUES (?,?,?,?,?,?)
            """,
            payload,
        )
        row = self._connection.execute(
            """SELECT logical_state_hash,dimension,dtype,vector,legal_mask
               FROM states WHERE tensor_state_hash=?""",
            (payload[0],),
        ).fetchone()
        if row is None or tuple(row) != payload[1:]:
            raise ValueError("tensor hash collision or inconsistent prepared state")

    def put_trajectory(self, transitions: Sequence[OfflineTransition]) -> None:
        if self.readonly:
            raise RuntimeError("prepared replay store is read-only")
        rows = tuple(transitions)
        if len(rows) != 4 or tuple(row.decision_idx for row in rows) != (0, 1, 2, 3):
            raise ValueError("prepared trajectory must contain decisions 0..3")
        qid = int(rows[0].qid)
        route = str(rows[0].route_key)
        if any(int(row.qid) != qid or str(row.route_key) != route for row in rows):
            raise ValueError("prepared trajectory identity is inconsistent")
        with self._lock:
            for row in rows:
                self._put_state(row.state)
                if row.next_state is not None:
                    self._put_state(row.next_state)
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO transitions
                        (qid,route_key,trajectory_identity,decision_idx,action,
                         state_hash,next_state_hash,reward_scalar,reward_f1,reward_cost,
                         mc_scalar,mc_f1,mc_cost,terminal,final_f1,remaining_cost,
                         actions_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        qid,
                        route,
                        str(row.trajectory_identity),
                        int(row.decision_idx),
                        int(row.action),
                        str(row.state.tensor_state_hash),
                        None if row.next_state is None else str(row.next_state.tensor_state_hash),
                        float(row.reward_scalar),
                        float(row.reward_f1),
                        float(row.reward_cost),
                        float(row.mc_scalar),
                        float(row.mc_f1),
                        float(row.mc_cost),
                        int(bool(row.terminal)),
                        float(row.final_f1),
                        float(row.remaining_cost),
                        json.dumps(list(row.actions), separators=(",", ":")),
                    ),
                )

    def _load_state(self, tensor_hash: str) -> EncodedState:
        row = self._connection.execute(
            """SELECT logical_state_hash,dimension,dtype,vector,legal_mask
               FROM states WHERE tensor_state_hash=?""",
            (str(tensor_hash),),
        ).fetchone()
        if row is None:
            raise KeyError(f"prepared state is missing: {tensor_hash}")
        if int(row["dimension"]) != STATE_V1_DIM or str(row["dtype"]) != "float32":
            raise ValueError("prepared state schema mismatch")
        vector = np.frombuffer(row["vector"], dtype="<f4").copy()
        legal = np.frombuffer(row["legal_mask"], dtype=np.uint8).astype(np.bool_)
        if vector.shape != (STATE_V1_DIM,) or legal.shape != (4,):
            raise ValueError("prepared state payload is corrupt")
        return EncodedState(
            vector=vector,
            legal_action_mask=legal,
            logical_state_hash=str(row["logical_state_hash"]),
            tensor_state_hash=str(tensor_hash),
        )

    def load_trajectory(self, qid: int, route_key: str) -> tuple[OfflineTransition, ...]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM transitions
                   WHERE qid=? AND route_key=? ORDER BY decision_idx""",
                (int(qid), str(route_key)),
            ).fetchall()
            if len(rows) != 4:
                raise KeyError(f"complete prepared trajectory is missing: {(qid, route_key)}")
            answer: list[OfflineTransition] = []
            for row in rows:
                state = self._load_state(str(row["state_hash"]))
                next_hash = row["next_state_hash"]
                next_state = None if next_hash is None else self._load_state(str(next_hash))
                actions = tuple(int(value) for value in json.loads(row["actions_json"]))
                answer.append(
                    OfflineTransition(
                        qid=int(row["qid"]),
                        route_key=str(row["route_key"]),
                        trajectory_identity=str(row["trajectory_identity"]),
                        decision_idx=int(row["decision_idx"]),
                        action=int(row["action"]),
                        state=state,
                        next_state=next_state,
                        reward_scalar=float(row["reward_scalar"]),
                        reward_f1=float(row["reward_f1"]),
                        reward_cost=float(row["reward_cost"]),
                        mc_scalar=float(row["mc_scalar"]),
                        mc_f1=float(row["mc_f1"]),
                        mc_cost=float(row["mc_cost"]),
                        terminal=bool(row["terminal"]),
                        final_f1=float(row["final_f1"]),
                        remaining_cost=float(row["remaining_cost"]),
                        actions=actions,  # type: ignore[arg-type]
                    )
                )
            return tuple(answer)

    def routes_by_qid(self) -> dict[int, tuple[str, ...]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT qid,route_key,COUNT(*) AS n FROM transitions
                   GROUP BY qid,route_key HAVING n=4 ORDER BY qid,route_key"""
            ).fetchall()
        result: dict[int, list[str]] = {}
        for row in rows:
            result.setdefault(int(row["qid"]), []).append(str(row["route_key"]))
        return {qid: tuple(routes) for qid, routes in result.items()}

    def trajectory_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """SELECT COUNT(*) AS n FROM
                   (SELECT 1 FROM transitions GROUP BY qid,route_key HAVING COUNT(*)=4)"""
            ).fetchone()
            return int(row["n"])

    def transition_count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM transitions").fetchone()
            return int(row["n"])

    def state_count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM states").fetchone()
            return int(row["n"])

    def commit(self) -> None:
        if self.readonly:
            return
        with self._lock:
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                if not self.readonly:
                    self._connection.commit()
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "PreparedReplayStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class _LazyTrajectoryMapping(Mapping[tuple[int, str], tuple[OfflineTransition, ...]]):
    def __init__(self, store: PreparedReplayStore, routes_by_qid: Mapping[int, Sequence[str]]):
        self.store = store
        self._keys = tuple(
            (int(qid), str(route))
            for qid, routes in sorted(routes_by_qid.items())
            for route in routes
        )
        self._key_set = set(self._keys)

    def __getitem__(self, key: tuple[int, str]) -> tuple[OfflineTransition, ...]:
        canonical = (int(key[0]), str(key[1]))
        if canonical not in self._key_set:
            raise KeyError(canonical)
        return self.store.load_trajectory(*canonical)

    def __iter__(self) -> Iterator[tuple[int, str]]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


class PreparedReplayDataset:
    """Schedule-compatible lazy replay used by the formal trainer."""

    def __init__(
        self,
        store: PreparedReplayStore,
        schedule: DeterministicRouteSchedule,
    ) -> None:
        self.store = store
        self.schedule = schedule
        self.trajectories = _LazyTrajectoryMapping(store, schedule.routes_by_qid)
        self.last_batch_identity = ""

    def next_batch(self) -> tuple[OfflineTransition, ...]:
        selected = self.schedule.next_batch()
        rows = tuple(
            transition
            for identity in selected
            for transition in self.trajectories[identity]
        )
        self.last_batch_identity = hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return rows

    def state_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.state_dict(),
            "last_batch_identity": self.last_batch_identity,
            "store_identity": self.store.identity,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if str(state.get("store_identity")) != self.store.identity:
            raise ValueError("prepared replay store identity mismatch")
        self.schedule.load_state_dict(state["schedule"])
        self.last_batch_identity = str(state.get("last_batch_identity", ""))


__all__ = ["PreparedReplayDataset", "PreparedReplayStore"]
