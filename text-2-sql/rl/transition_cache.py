"""
TransitionCache — 算子级 I/O KV 缓存 (SQLite)

核心设计:
  - Key: (question_id, stage_idx, tier, input_payload_hash, operator_version)
  - Value: output_payload_json (算子执行后的 StatePayload)
  - 两步查找: L1 exact hash → L2 semantic prefix match
  - 跨 tier 语义等价复用 (同 qid+stage, 不同 tier, 语义等价 input → 复用 output)
  - operator_version: 改 prompt → bump version → 旧条目过滤

Usage:
  from rl.transition_cache import TransitionCache, TransitionKey, compute_input_hash
  cache = TransitionCache("cache/transitions_train.db")
  key = TransitionKey(question_id=qid, stage_idx=0, tier=0, input_payload_hash=h)
  cached = cache.lookup(key)
  if cached is None:
      payload = operator(payload, ...)
      cache.store(key, payload)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

STAGE_NAMES = [
    "keyword_extraction",      # S1, stage_idx=0
    "entity_retrieval",        # S2, stage_idx=1
    "context_retrieval",       # S3, stage_idx=2
    "column_filtering",        # S4, stage_idx=3
    "table_selection",         # S5, stage_idx=4
    "column_selection",        # S6, stage_idx=5
    "candidate_generation",    # S7, stage_idx=6
    "revision",                # S8, stage_idx=7
    "evaluation",              # S9 (不进 TransitionCache, 用 S9Cache)
]

STAGES_IN_CACHE = list(range(8))  # 0-7, S9 不进

# 语义规范化字段 — 从中提取用于 Level 2 semantic hash 的 key
# 注意: 不包含 cost_tracker (时间相关), revision_history (执行相关)
SEMANTIC_FIELDS: set = {
    "question", "evidence", "db_id", "gold_sql",
    "keywords", "similar_values", "similar_columns",
    "schema_with_descriptions", "tentative_schema",
    "selected_tables", "column_roles",
    "candidates", "final_sql", "revision_count",
}

# 各算子的 INPUT_FIELDS — 声明该算子依赖 StatePayload 的哪些字段
# key: (stage_idx, tier), value: set of field names
# 仅 hash 这些字段, 不 hash cost_tracker 等无关字段
# 注意: T2/T3 和 T1 使用相同的字段集合.
# All tiers use the same Chess operator function with a different model binding.
# T2/T3 实际也读取和 T1 一样的 payload 字段.
# 对 1_new / 2_new_combine (T2/T3 是不同的算子, 确实少用了一些字段),
# 多 hash 几个字段是安全的 — 只会导致略少的缓存命中, 不会导致错误的命中.
OPERATOR_INPUT_FIELDS: Dict[Tuple[int, int], set] = {
    # S1: keyword_extraction
    (0, 0): {"question", "evidence", "db_id"},
    (0, 1): {"question", "evidence", "db_id"},
    (0, 2): {"question", "evidence", "db_id"},
    # S2: entity_retrieval
    (1, 0): {"keywords", "similar_values", "similar_columns", "db_id"},
    (1, 1): {"keywords", "similar_values", "similar_columns", "db_id"},
    (1, 2): {"keywords", "similar_values", "similar_columns", "db_id"},
    # S3: context_retrieval
    (2, 0): {"schema_with_descriptions", "db_id"},
    (2, 1): {"schema_with_descriptions", "db_id"},
    (2, 2): {"schema_with_descriptions", "db_id"},
    # S4: column_filtering
    (3, 0): {"tentative_schema", "schema_with_descriptions", "db_id"},
    (3, 1): {"tentative_schema", "schema_with_descriptions", "db_id"},
    (3, 2): {"tentative_schema", "schema_with_descriptions", "db_id"},
    # S5: table_selection
    (4, 0): {"tentative_schema", "schema_with_descriptions", "selected_tables", "db_id"},
    (4, 1): {"tentative_schema", "schema_with_descriptions", "selected_tables", "db_id"},
    (4, 2): {"tentative_schema", "schema_with_descriptions", "selected_tables", "db_id"},
    # S6: column_selection
    (5, 0): {"tentative_schema", "selected_tables", "column_roles", "db_id"},
    (5, 1): {"tentative_schema", "selected_tables", "column_roles", "db_id"},
    (5, 2): {"tentative_schema", "selected_tables", "column_roles", "db_id"},
    # S7: candidate_generation
    (6, 0): {"tentative_schema", "selected_tables", "column_roles",
             "schema_with_descriptions", "question", "evidence", "db_id"},
    (6, 1): {"tentative_schema", "selected_tables", "column_roles",
             "schema_with_descriptions", "question", "evidence", "db_id"},
    (6, 2): {"tentative_schema", "selected_tables", "column_roles",
             "schema_with_descriptions", "question", "evidence", "db_id"},
    # S8: revision
    (7, 0): {"candidates", "final_sql", "tentative_schema", "db_id"},
    (7, 1): {"candidates", "final_sql", "tentative_schema", "db_id"},
    (7, 2): {"candidates", "final_sql", "tentative_schema", "db_id"},
}

EXACT_OPERATOR_INPUT_FIELDS: Dict[int, Tuple[str, ...]] = {
    0: ("question", "evidence"),
    1: ("db_id", "keywords"),
    2: ("db_id", "question", "evidence", "keywords"),
    3: ("db_id", "question", "evidence", "similar_values"),
    4: (
        "db_id", "question", "evidence", "tentative_schema",
        "similar_values", "schema_with_descriptions",
    ),
    5: (
        "db_id", "question", "evidence", "tentative_schema",
        "similar_values", "schema_with_descriptions",
    ),
    6: (
        "db_id", "question", "evidence", "tentative_schema",
        "similar_values", "schema_with_descriptions",
    ),
    7: (
        "db_id", "question", "evidence", "final_sql",
        "similar_values", "schema_with_descriptions",
    ),
}

EXACT_IDENTITY_FIELDS: Tuple[str, ...] = (
    "contract_version",
    "operator_package",
    "stage_idx",
    "tier",
    "effective_model_config",
    "stage_source",
    "prompt",
    "parser",
    "database",
    "index",
)

_PROMPT_PARSER_STAGES = {0, 3, 4, 5, 6, 7}
_DATABASE_STAGES = {3, 4, 5, 6, 7}
_INDEX_STAGES = {1, 2, 3}
_EMBEDDING_STAGES = {1, 2}


# ============================================================
# TransitionKey
# ============================================================

@dataclass(frozen=True)
class TransitionKey:
    """缓存主键 — 不可变, 可哈希, 可序列化。"""
    task_type: str = "text-to-sql"
    question_id: int = 0
    stage_idx: int = 0        # 0-7 (S1-S8)
    tier: int = 0             # 0=T1, 1=T2, 2=T3
    input_payload_hash: str = ""   # 完整 128-bit MD5
    operator_version: int = 1

    @property
    def table_key(self) -> str:
        """SQLite 主键 — 用于 L1 exact lookup。"""
        return f"{self.question_id}|{self.stage_idx}|{self.tier}|" \
               f"{self.input_payload_hash}|v{self.operator_version}"

    @property
    def semantic_prefix(self) -> str:
        """语义查找前缀 — 同 qid+stage+tier+version, 不同 hash。"""
        return f"{self.question_id}|{self.stage_idx}|{self.tier}|v{self.operator_version}"

    def to_row(self) -> Tuple[str, int, int, int, str, int]:
        return (self.task_type, self.question_id, self.stage_idx, self.tier,
                self.input_payload_hash, self.operator_version)


# ============================================================
# Hash 计算
# ============================================================

def _semantic_normalize(field_name: str, raw_value: Any) -> str:
    """
    语义规范化 — 使同语义不同形式的 input 产生相同 hash。

    规则:
      - None → "__NONE__"
      - str → lowercase, strip
      - list[str] → sorted, deduplicated, lowercase
      - list[dict] → sorted by json.dumps of each item
      - dict → sorted by keys, recursive normalize
    """
    if raw_value is None:
        return "__NONE__"

    if isinstance(raw_value, (int, float)):
        return str(raw_value)

    if isinstance(raw_value, str):
        return raw_value.lower().strip()

    if isinstance(raw_value, list):
        if len(raw_value) == 0:
            return "[]"
        if all(isinstance(v, str) for v in raw_value):
            items = sorted(set(v.lower().strip() for v in raw_value))
            return json.dumps(items, sort_keys=True, ensure_ascii=False)
        # 混合类型: 每个 item 递归 normalize 后排序
        items = sorted(_semantic_normalize(f"{field_name}[]", v) for v in raw_value)
        return "[" + ",".join(items) + "]"

    if isinstance(raw_value, dict):
        if len(raw_value) == 0:
            return "{}"
        items = sorted(
            f"{_semantic_normalize(k, k)}:{_semantic_normalize(k, v)}"
            for k, v in raw_value.items()
        )
        return "{" + ",".join(items) + "}"

    return str(raw_value)


def compute_input_hash(
    payload_dict: Dict[str, Any],
    stage_idx: int,
    tier: int,
    *,
    exact_fields: Optional[Sequence[str]] = None,
    identity_envelope: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    计算算子输入的 hash。

    仅 hash OPERATOR_INPUT_FIELDS 声明的字段, 忽略 cost_tracker 等无关字段。
    使用完整 128-bit MD5, key=value 按字母序拼接。

    Args:
        payload_dict: StatePayload.to_json_dict() 的输出
        stage_idx: 0-7
        tier: 0=T1, 1=T2, 2=T3

    Returns:
        32 字符 MD5 hex digest

    Raises:
        ValueError: stage_idx 不在 0-7 范围内
    """
    if stage_idx not in STAGES_IN_CACHE:
        raise ValueError(f"stage_idx {stage_idx} not in {STAGES_IN_CACHE} (S9 not cached)")

    exact_requested = exact_fields is not None or identity_envelope is not None
    if exact_requested:
        if not exact_fields or identity_envelope is None:
            raise ValueError(
                "exact_fields and identity_envelope are both required for exact hashing"
            )
        if tuple(exact_fields) != EXACT_OPERATOR_INPUT_FIELDS.get(stage_idx):
            raise ValueError("exact_fields must match the fixed stage input schema")
        _validate_exact_identity(identity_envelope, stage_idx, tier)
        if _contains_api_key(identity_envelope):
            raise ValueError("exact identity must not contain an API key")
        projection = [[field, payload_dict.get(field)] for field in exact_fields]
        if not _is_json_value(projection):
            raise ValueError(
                "exact input fields must contain only JSON-compatible values"
            )
        encoded = json.dumps(
            [["payload", projection]],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    input_fields = OPERATOR_INPUT_FIELDS.get((stage_idx, tier), set())
    if not input_fields:
        # fallback: hash all semantic fields
        input_fields = SEMANTIC_FIELDS

    parts = []
    for field in sorted(input_fields):
        raw = payload_dict.get(field)
        normalized = _semantic_normalize(field, raw)
        parts.append(f"{field}={normalized}")

    canonical = "\n".join(parts)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _validate_exact_identity(
    identity: Mapping[str, Any], stage_idx: int, tier: int
) -> None:
    missing = [field for field in EXACT_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"exact identity is missing required fields: {missing}")
    if not _is_json_value(identity):
        raise ValueError("exact identity must contain only JSON-compatible values")
    if type(identity["stage_idx"]) is not int or identity["stage_idx"] != stage_idx:
        raise ValueError("exact identity stage_idx does not match hash stage_idx")
    if type(identity["tier"]) is not int or identity["tier"] != tier:
        raise ValueError("exact identity tier does not match hash tier")
    for field in ("contract_version", "operator_package"):
        if not isinstance(identity[field], str) or not identity[field].strip():
            raise ValueError(f"exact identity {field} must be a non-empty string")
    if stage_idx in _EMBEDDING_STAGES:
        _validate_embedding_config(identity["effective_model_config"])
    else:
        _validate_llm_config(identity["effective_model_config"])
    source = identity["stage_source"]
    if not isinstance(source, Mapping) or source.get("kind") != "source":
        raise ValueError("exact identity stage_source must be typed source")
    if not isinstance(source.get("module"), str) or not source["module"].strip():
        raise ValueError("exact identity stage_source.module must be non-empty")
    for digest_field in ("sha256", "callable_sha256"):
        if not _is_sha256(source.get(digest_field)):
            raise ValueError(
                f"exact identity stage_source.{digest_field} must be SHA-256"
            )
    applicability = {
        "prompt": stage_idx in _PROMPT_PARSER_STAGES,
        "parser": stage_idx in _PROMPT_PARSER_STAGES,
        "database": stage_idx in _DATABASE_STAGES,
        "index": stage_idx in _INDEX_STAGES,
    }
    for field, applicable in applicability.items():
        _validate_artifact_identity(field, identity[field], applicable)


def _validate_artifact_identity(field: str, value: Any, applicable: bool) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"exact identity {field} must be a typed mapping")
    if applicable:
        if value.get("kind") != "sha256" or not _is_sha256(value.get("sha256")):
            raise ValueError(
                f"exact identity {field} must contain a SHA-256 artifact"
            )
        return
    if value.get("kind") != "not_applicable":
        raise ValueError(
            f"exact identity {field} must use a not_applicable sentinel"
        )
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError(
            f"exact identity {field} sentinel reason must be non-empty"
        )


def _validate_llm_config(value: Any) -> None:
    if not isinstance(value, Mapping) or value.get("kind") != "model_config":
        raise ValueError(
            "exact identity effective_model_config must be typed model_config"
        )
    if not isinstance(value.get("model"), str) or not value["model"].strip():
        raise ValueError("exact identity effective_model_config.model must be non-empty")
    if type(value.get("temperature")) not in (int, float):
        raise ValueError(
            "exact identity effective_model_config.temperature is required"
        )
    if not isinstance(value.get("enable_thinking"), bool):
        raise ValueError(
            "exact identity effective_model_config.enable_thinking is required"
        )


def _validate_embedding_config(value: Any) -> None:
    if not isinstance(value, Mapping) or value.get("kind") != "embedding_config":
        raise ValueError(
            "exact identity effective_model_config must be typed embedding_config"
        )
    if not isinstance(value.get("model"), str) or not value["model"].strip():
        raise ValueError("exact identity embedding model must be non-empty")
    endpoint = value.get("endpoint")
    if not isinstance(endpoint, Mapping) or endpoint.get("kind") != "endpoint":
        raise ValueError("exact identity embedding endpoint must be typed endpoint")
    raw_url = endpoint.get("url")
    if raw_url != normalize_endpoint_identity(raw_url):
        raise ValueError(
            "exact identity embedding endpoint URL must be normalized"
        )


def normalize_endpoint_identity(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("embedding endpoint URL must be non-empty")
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("embedding endpoint URL must be non-sensitive HTTP(S)")
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("embedding endpoint URL has an invalid port") from exc
    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    return urlunsplit((scheme, host, parsed.path.rstrip("/"), "", ""))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_value(nested)
            for key, nested in value.items()
        )
    return False


def _contains_api_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = "".join(
                char for char in str(key).lower() if char.isalnum()
            )
            if "apikey" in normalized:
                allowed_flag = normalized.startswith("hasapikey") and isinstance(
                    nested, bool
                )
                allowed_hash = normalized.endswith("hash") and _is_sha256(nested)
                if not (allowed_flag or allowed_hash):
                    return True
            if _contains_api_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_api_key(item) for item in value)
    return False


def compute_semantic_hash(
    payload_dict: Dict[str, Any],
    stage_idx: int,
) -> str:
    """
    计算语义 hash — 用于 Level 2 semantic match。

    仅规范化 SEMANTIC_FIELDS 中的字段, 忽略 tier 和 cost。
    与 compute_input_hash 不同: 这个 hash 是跨 tier 可比较的。

    Args:
        payload_dict: StatePayload.to_json_dict() 的输出
        stage_idx: 0-7

    Returns:
        32 字符 MD5 hex digest
    """
    parts = []
    for field in sorted(SEMANTIC_FIELDS):
        raw = payload_dict.get(field)
        normalized = _semantic_normalize(field, raw)
        parts.append(f"{field}={normalized}")
    canonical = "\n".join(parts)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


# ============================================================
# TransitionCache
# ============================================================

class TransitionCache:
    """
    算子级 I/O KV 缓存 (SQLite 后端)。

    两层查找:
      L1 Exact:  table_key 精确匹配 → stored_hash == input_hash 二次确认
      L2 Semantic: 同 (qid, stage, tier, version) → 语义 hash 前缀匹配

    Thread-safe: SQLite WAL mode + 连接 per 线程。

    Attributes:
        db_path: SQLite 数据库路径
        conn: 线程级 SQLite 连接
    """

    def __init__(
        self,
        db_path: str = "cache/transitions.db",
        allow_semantic_lookup: bool = True,
    ):
        self.db_path = db_path
        self.allow_semantic_lookup = bool(allow_semantic_lookup)
        self._conn_lock = threading.RLock()
        self._conns: Dict[int, sqlite3.Connection] = {}
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    # ---- 数据库初始化 ----

    def _init_db(self):
        """创建表结构 + 索引。"""
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transitions (
                    task_type       TEXT NOT NULL DEFAULT 'text-to-sql',
                    question_id     INTEGER NOT NULL,
                    stage_idx       INTEGER NOT NULL,
                    tier            INTEGER NOT NULL,
                    input_hash      TEXT NOT NULL,
                    operator_version INTEGER NOT NULL DEFAULT 1,
                    semantic_hash   TEXT NOT NULL DEFAULT '',
                    output_payload  TEXT NOT NULL,
                    stored_hash     TEXT NOT NULL,
                    source          TEXT NOT NULL DEFAULT 'unknown',
                    created_at      REAL NOT NULL DEFAULT (strftime('%s','now')),
                    PRIMARY KEY (question_id, stage_idx, tier, input_hash, operator_version)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_lookup
                ON transitions(question_id, stage_idx, tier, operator_version)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stage
                ON transitions(stage_idx, tier, operator_version)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程级 SQLite 连接。"""
        tid = threading.get_ident()
        with self._conn_lock:
            if tid not in self._conns:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._conns[tid] = conn
            return self._conns[tid]

    # ---- L1 Exact Lookup ----

    def lookup(
        self,
        key: TransitionKey,
        stored_hash: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        L1 Exact 查找: table_key 完全匹配。

        Args:
            key: TransitionKey
            stored_hash: 如果提供, 与存储的 stored_hash 二次确认

        Returns:
            output_payload dict if found, else None
        """
        with self._get_conn() as conn:
            if stored_hash:
                row = conn.execute(
                    """SELECT output_payload, stored_hash FROM transitions
                       WHERE question_id=? AND stage_idx=? AND tier=?
                         AND input_hash=? AND operator_version=?""",
                    (key.question_id, key.stage_idx, key.tier,
                     key.input_payload_hash, key.operator_version)
                ).fetchone()
                if row and row["stored_hash"] == stored_hash:
                    return json.loads(row["output_payload"])
                return None
            else:
                row = conn.execute(
                    """SELECT output_payload FROM transitions
                       WHERE question_id=? AND stage_idx=? AND tier=?
                         AND input_hash=? AND operator_version=?""",
                    (key.question_id, key.stage_idx, key.tier,
                     key.input_payload_hash, key.operator_version)
                ).fetchone()
                if row:
                    return json.loads(row["output_payload"])
                return None

    # ---- L2 Semantic Lookup ----

    def lookup_semantic(
        self,
        question_id: int,
        stage_idx: int,
        tier: int,
        operator_version: int,
        semantic_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """
        L2 Semantic 查找: 同 qid+stage+tier+version, semantic_hash 匹配。

        用于跨 tier 复用: 虽然 tier 不同, 但语义等价的 input 可以复用 output。

        Args:
            question_id: 问题 ID
            stage_idx: 阶段索引
            tier: 目标 tier
            operator_version: 算子版本
            semantic_hash: 当前 input 的语义 hash

        Returns:
            output_payload dict if found, else None
        """
        if not self.allow_semantic_lookup:
            raise RuntimeError("Semantic cache lookup is disabled")
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT output_payload FROM transitions
                   WHERE question_id=? AND stage_idx=? AND tier=?
                     AND operator_version=? AND semantic_hash=? AND source='cold_start'
                   LIMIT 1""",
                (question_id, stage_idx, tier, operator_version, semantic_hash)
            ).fetchone()
            if row:
                return json.loads(row["output_payload"])
            return None

    # ---- Store ----

    def store(
        self,
        key: TransitionKey,
        output_payload: Dict[str, Any],
        source: str = "unknown",
        semantic_hash: str = "",
        stored_hash: str = "",
    ):
        """
        存储一条缓存记录。

        Args:
            key: TransitionKey
            output_payload: 算子执行后的 StatePayload (to_json_dict 格式)
            source: 来源标记 ("cold_start" / "rl_exploration" / "rl_on_policy")
            semantic_hash: 语义 hash (用于 L2 查找)
            stored_hash: 二次验证 hash (存储时记录的 payload hash)
        """
        payload_json = json.dumps(output_payload, ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO transitions
                   (task_type, question_id, stage_idx, tier, input_hash,
                    operator_version, semantic_hash, output_payload, stored_hash, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (key.task_type, key.question_id, key.stage_idx, key.tier,
                 key.input_payload_hash, key.operator_version,
                 semantic_hash, payload_json, stored_hash, source)
            )
            conn.commit()

    # ---- Version Management ----

    def get_version(self, stage_idx: int, tier: int) -> int:
        """获取当前生效的 operator_version (默认 1, 需要手动 bump)。"""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT MAX(operator_version) FROM transitions
                   WHERE stage_idx=? AND tier=?""",
                (stage_idx, tier)
            ).fetchone()
            max_v = row[0] if row and row[0] else 0
            return max(max_v, 1)

    def bump_version(self, stage_idx: int, tier: int):
        """
        递增 operator_version — 改 prompt 后调用。

        新 version = max(current_version) + 1。
        旧条目不会被删除, 但 lookup 时被 version 过滤。
        """
        new_v = self.get_version(stage_idx, tier) + 1
        logger.info(
            "Bumping operator version for stage_idx=%d tier=%d: %d → %d",
            stage_idx, tier, new_v - 1, new_v
        )
        # version bump 无需写入 — 下次 store 时自动使用新 version
        # 这里只记录日志, 实际由调用方传入新 version 的 TransitionKey
        return new_v

    # ---- Admin ----

    def count(self, stage_idx: Optional[int] = None, tier: Optional[int] = None) -> int:
        """统计缓存条目数。"""
        with self._get_conn() as conn:
            if stage_idx is not None and tier is not None:
                row = conn.execute(
                    "SELECT COUNT(*) FROM transitions WHERE stage_idx=? AND tier=?",
                    (stage_idx, tier)
                ).fetchone()
            elif stage_idx is not None:
                row = conn.execute(
                    "SELECT COUNT(*) FROM transitions WHERE stage_idx=?", (stage_idx,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM transitions").fetchone()
            return row[0] if row else 0

    def stats(self) -> Dict[str, Any]:
        """缓存统计信息。"""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
            by_stage = {}
            for si in range(8):
                for ti in range(3):
                    cnt = conn.execute(
                        "SELECT COUNT(*) FROM transitions WHERE stage_idx=? AND tier=?",
                        (si, ti)
                    ).fetchone()[0]
                    if cnt > 0:
                        by_stage[f"{STAGE_NAMES[si]}_T{ti+1}"] = cnt
            return {
                "total_entries": total,
                "by_stage_tier": by_stage,
                "db_path": self.db_path,
            }

    def close(self):
        """关闭所有连接。"""
        with self._conn_lock:
            for conn in self._conns.values():
                conn.close()
            self._conns.clear()
