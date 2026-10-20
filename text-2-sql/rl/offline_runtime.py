"""Read-only transition access and network isolation for offline preparation."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sqlite3
import sys
import threading
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class OfflineNetworkError(RuntimeError):
    pass


class _BlockedApiImportLoader:
    def __init__(self, fullname: str) -> None:
        self.fullname = fullname

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        raise OfflineNetworkError(f"network API import blocked during offline execution: {self.fullname}")


class _BlockedApiImportFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {"openai", "dashscope"}:
            return importlib.util.spec_from_loader(
                fullname, _BlockedApiImportLoader(fullname), is_package="." not in fullname
            )
        return None


def _blocked_network_call(*args, **kwargs):
    raise OfflineNetworkError("network access blocked during offline execution")


@contextmanager
def offline_execution_guard():
    """Block API imports and network calls during offline preparation/training."""
    env_names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    missing = object()
    old_env = {name: os.environ.get(name, missing) for name in env_names}
    patched: list[tuple[Any, str, Any]] = []
    finder = _BlockedApiImportFinder()

    def patch(owner: Any, name: str) -> None:
        if owner is None or not hasattr(owner, name):
            return
        original = getattr(owner, name)
        try:
            setattr(owner, name, _blocked_network_call)
        except (AttributeError, TypeError):
            return
        patched.append((owner, name, original))

    try:
        for name in env_names:
            os.environ[name] = "1"
        patch(socket, "create_connection")
        patch(socket.socket, "connect")
        patch(socket.socket, "connect_ex")
        patch(socket.socket, "sendto")
        patch(urllib.request, "urlopen")
        requests_sessions = sys.modules.get("requests.sessions")
        patch(getattr(requests_sessions, "Session", None), "request")
        httpx = sys.modules.get("httpx")
        patch(getattr(httpx, "Client", None), "send")
        patch(getattr(httpx, "AsyncClient", None), "send")
        openai = sys.modules.get("openai")
        for name in ("OpenAI", "AsyncOpenAI", "Client", "AsyncClient"):
            patch(openai, name)
        dashscope = sys.modules.get("dashscope")
        for name in ("Generation", "TextEmbedding", "MultiModalConversation", "Application"):
            patch(getattr(dashscope, name, None), "call")
        sys.meta_path.insert(0, finder)
        yield
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)
        for name, value in old_env.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)


class ReadOnlyTransitionCache:
    """TransitionCache-compatible SQLite reader that never initializes or writes."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(Path(db_path).resolve())
        if not Path(self.db_path).is_file():
            raise FileNotFoundError(f"transition DB not found: {self.db_path}")
        self._conn_lock = threading.RLock()
        self._conns: dict[int, sqlite3.Connection] = {}

    def _get_conn(self) -> sqlite3.Connection:
        thread_id = threading.get_ident()
        with self._conn_lock:
            if thread_id not in self._conns:
                uri = f"{Path(self.db_path).as_uri()}?mode=ro&immutable=1"
                conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._conns[thread_id] = conn
            return self._conns[thread_id]

    def lookup(self, key: Any, stored_hash: str | None = None) -> Any:
        columns = "output_payload, stored_hash" if stored_hash else "output_payload"
        row = self._get_conn().execute(
            f"""SELECT {columns} FROM transitions
                WHERE question_id=? AND stage_idx=? AND tier=?
                  AND input_hash=? AND operator_version=?""",
            (
                int(key.question_id), int(key.stage_idx), int(key.tier),
                str(key.input_payload_hash), int(key.operator_version),
            ),
        ).fetchone()
        if row is None or (stored_hash and row["stored_hash"] != stored_hash):
            return None
        return json.loads(row["output_payload"])

    def lookup_semantic(
        self, question_id: int, stage_idx: int, tier: int,
        operator_version: int, semantic_hash: str,
    ) -> Any:
        row = self._get_conn().execute(
            """SELECT output_payload FROM transitions
               WHERE question_id=? AND stage_idx=? AND tier=?
                 AND operator_version=? AND semantic_hash=?
                 AND source='cold_start' LIMIT 1""",
            (int(question_id), int(stage_idx), int(tier), int(operator_version), str(semantic_hash)),
        ).fetchone()
        return None if row is None else json.loads(row["output_payload"])

    def get_version(self, stage_idx: int, tier: int) -> int:
        row = self._get_conn().execute(
            "SELECT MAX(operator_version) FROM transitions WHERE stage_idx=? AND tier=?",
            (int(stage_idx), int(tier)),
        ).fetchone()
        return max(int(row[0] or 0), 1)

    def close(self) -> None:
        with self._conn_lock:
            for connection in self._conns.values():
                connection.close()
            self._conns.clear()
