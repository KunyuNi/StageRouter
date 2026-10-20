"""
EnvironmentWrapper ?Pipeline 

:
  1.  9  Pipeline ?step 
  2.  TransitionCache ?
  3. rollout_with_cache: ? ?
  4. get_action_mask: ?tier 

Stage ?Action :
  S1-S8: ?T1(0), T2(1), T3(2), Skip(3)
  S9:  (? ? ?LLM  ?, ?

Action :
  0 = T1 ()
  1 = T2 ()
  2 = T3 ()
  3 = Skip (, ? S7 ?

Usage:
  from rl.env_wrapper import EnvironmentWrapper, rollout_with_cache, get_action_mask
  env = EnvironmentWrapper(operator_pkg="env.1_new")
  payload, cost, hit = env.step(stage_idx=0, tier=0, payload=payload, ...)
"""

from __future__ import annotations

import copy
import errno
import hashlib
import importlib
import json
import logging
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from env.database_utils.execution import is_retryable_sqlite_infrastructure_error

from .api_guard import ApiQuotaExhaustedError, as_fatal_api_error
from .stage_mapping import (
    COLUMN_FILTERING_PIPELINE_IDX,
    FIXED_PIPELINE_STAGE_IDXS,
    FIXED_STAGE_DEFAULT_TIER,
    NON_SKIPPABLE_PIPELINE_STAGE_IDXS,
    NUM_PIPELINE_STAGES,
    NUM_RL_STAGES,
    PIPELINE_STAGE_NAMES,
    RL_DECISION_TO_PIPELINE_STAGE_IDXS,
    decision_to_pipeline_stage_idx,
    legal_decision_actions,
)


_EXACT_REPLAY_INPUT_FIELDS: Dict[int, Tuple[str, ...]] = {
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


def compute_exact_replay_input_hash(
    payload_dict: Dict[str, Any],
    stage_idx: int,
) -> str:
    """Compute the New2 cache key from its fixed ordered payload projection."""
    exact_fields = _EXACT_REPLAY_INPUT_FIELDS.get(stage_idx)
    if exact_fields is None:
        raise ValueError(f"stage_idx {stage_idx} has no exact cache schema")
    projection = [[field, payload_dict.get(field)] for field in exact_fields]
    try:
        encoded = json.dumps(
            [["payload", projection]],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("exact cache input must be finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()

try:
    from _shared.model_config import DEFAULT_LLM_MODEL
except ModuleNotFoundError:
    try:
        from env._shared.model_config import DEFAULT_LLM_MODEL
    except ModuleNotFoundError:
        from nl2sql.env._shared.model_config import DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)

# ============================================================
# 
# ============================================================

STAGE_NAMES = list(PIPELINE_STAGE_NAMES)
RL_STAGE_NAMES = [STAGE_NAMES[i] for i in RL_DECISION_TO_PIPELINE_STAGE_IDXS]
NUM_TIERS = 4              # T1, T2, T3, Skip

# Decision-stage budget weights for the six controllable stages [S1, S4, S5, S6, S7, S8].
STAGE_BUDGET_WEIGHTS = {
    0: 0.12,   # D0: S1
    1: 0.25,   # D1: S4 column_filtering
    2: 0.12,   # D2: S5 table_selection
    3: 0.12,   # D3: S6 column_selection
    4: 0.24,   # D4: S7 candidate_generation
    5: 0.15,   # D5: S8 revision
}

# ?(?
DEFAULT_BUDGET = 10.0


RETRYABLE_INFRASTRUCTURE_IDENTITIES = frozenset(
    {
        "api_connection",
        "api_rate_limit",
        "api_server",
        "network_connection",
        "network_timeout",
        "process_failure",
        "process_timeout",
        "sqlite_busy",
        "sqlite_connection",
        "transient_io",
    }
)


class RetryableInfrastructureError(RuntimeError):
    """Explicit wrapper for a known transient infrastructure contract."""

    def __init__(self, identity: str) -> None:
        if identity not in RETRYABLE_INFRASTRUCTURE_IDENTITIES:
            raise ValueError("invalid retryable infrastructure identity")
        super().__init__(identity)
        self.identity = identity


_TRANSIENT_OS_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "EAGAIN", None),
        getattr(errno, "EBUSY", None),
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "ECONNREFUSED", None),
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "EINTR", None),
        getattr(errno, "EIO", None),
        getattr(errno, "ENETDOWN", None),
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "ENOSPC", None),
        getattr(errno, "ESTALE", None),
        getattr(errno, "ETIMEDOUT", None),
    )
    if value is not None
)
def retryable_infrastructure_identity(exc: BaseException) -> str | None:
    """Classify only the outer, directly approved transient exception type."""
    if isinstance(exc, RetryableInfrastructureError):
        return exc.identity
    if isinstance(exc, subprocess.TimeoutExpired):
        return "process_timeout"
    if isinstance(exc, sqlite3.Error):
        if not is_retryable_sqlite_infrastructure_error(exc):
            return None
        error_code = getattr(exc, "sqlite_errorcode", None)
        if type(error_code) is int and (error_code & 0xFF) in {5, 6}:
            return "sqlite_busy"
        message = str(exc).lower()
        if message in {
            "database is busy",
            "database is locked",
            "database table is locked",
        }:
            return "sqlite_busy"
        return "sqlite_connection"
    if isinstance(exc, TimeoutError):
        return "network_timeout"
    if isinstance(exc, ConnectionError):
        return "network_connection"
    if isinstance(exc, OSError) and exc.errno in _TRANSIENT_OS_ERRNOS:
        return "transient_io"

    exception_type = type(exc)
    module = exception_type.__module__.split(".", 1)[0]
    name = exception_type.__name__
    if module in {"openai", "httpx", "httpcore", "requests", "dashscope"}:
        if name in {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectTimeout",
            "ConnectionError",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "Timeout",
            "TimeoutException",
        }:
            return "api_connection"
        if name in {"RateLimitError", "TooManyRequests"}:
            return "api_rate_limit"
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429} or (
            type(status_code) is int and status_code >= 500
        ):
            return "api_server"
    if isinstance(exc, RuntimeError) and str(exc).startswith(
        "DashScope API error: code="
    ):
        try:
            status_code = int(str(exc).split("code=", 1)[1].split(",", 1)[0])
        except (ValueError, IndexError):
            return None
        if status_code in {408, 409, 429} or status_code >= 500:
            return "api_server"
    return None


# ============================================================
# Action Mask
# ============================================================

def get_action_mask(
    stage_idx: int,
    payload: Any,                   # StatePayload
    budget_remaining: float = DEFAULT_BUDGET,
    total_budget: float = DEFAULT_BUDGET,
) -> List[int]:
    """
    ?action ?

    Action : 0=T1, 1=T2, 2=T3, 3=Skip

    :
      - S7 (candidate_generation):  SQL ??Skip (3)
      - S1-S6, S8: ?[0,1,2,3]
      - ?schema (S4-S6) ?T1 ?
      - S7 / ?T3 (?SQL ),  Skip
      - ?S7  ??Skip
      - ?S7  ?T3/Skip

    Args:
        stage_idx: 0-7 (S1-S8)
        payload: StatePayload
        budget_remaining:  (?
        total_budget: ?(?

    Returns:
        List[int] ??action 
    """
    return list(legal_decision_actions(stage_idx))


# ============================================================
# EnvironmentWrapper
# ============================================================

class EnvironmentWrapper:
    """
    Pipeline  ? +  + ?

     step ?stage:
      1. ?TransitionCache ?HIT ??output
      2. MISS ? ??? output

    Attributes:
        operator_pkg:  (?"env.1_new")
        model: LLM model name.
        db_manager_factory:  DatabaseManager ?
        total_budget: ?(?
    """

    def __init__(
        self,
        operator_pkg: str = "env.1_new",
        model: str = DEFAULT_LLM_MODEL,
        db_manager_factory: Optional[Callable] = None,
        total_budget: float = DEFAULT_BUDGET,
        cache_identity: Optional[Dict[Tuple[str, int, int], Dict[str, Any]]] = None,
        exact_only: bool = False,
        exact_replay_only: bool = False,
    ):
        self.operator_pkg = operator_pkg
        self.model = model
        self.db_manager_factory = db_manager_factory
        self.total_budget = total_budget
        self.cache_identity = cache_identity
        self.exact_only = bool(exact_only)
        self.exact_replay_only = bool(exact_replay_only)
        self._exact_operator_cache: Dict[Tuple[int, int], Callable] = {}
        self._exact_operator_source_hashes: Dict[Tuple[int, int], str] = {}
        self._last_step_latency_ms = 0.0

    def _allows_skip(self, pipeline_stage_idx: int) -> bool:
        """Apply the action-space contract of the selected operator package."""
        return int(pipeline_stage_idx) not in NON_SKIPPABLE_PIPELINE_STAGE_IDXS

    @property
    def non_skippable_decision_stage_idxs(self) -> tuple[int, ...]:
        return tuple(
            decision_idx
            for decision_idx in range(NUM_RL_STAGES)
            if not self._allows_skip(decision_to_pipeline_stage_idx(decision_idx))
        )

    # ---- Step ----

    def step(
        self,
        stage_idx: int,
        tier: int,
        payload: Any,                     # StatePayload
        cache,                            # TransitionCache
        question_id: int,
        db_manager: Any = None,           # DatabaseManager
        operator_version: int = 1,
        cache_only: bool = False,
    ) -> Tuple[Any, float, bool]:
        """
        ?

        :
          1.  input_hash ??TransitionCache
          2. HIT ??cached output  payload
          3. MISS ? ??

        Args:
            stage_idx: 0-7 (S1-S8)
            tier: 0=T1, 1=T2, 2=T3
            payload:  StatePayload
            cache: TransitionCache 
            question_id:  ID
            db_manager: DatabaseManager 
            operator_version: ?

        Returns:
            (new_payload, cost_dollar, cache_hit)
        """
        from rl.transition_cache import (
            EXACT_OPERATOR_INPUT_FIELDS,
            TransitionKey,
            compute_input_hash,
        )

        if stage_idx >= NUM_PIPELINE_STAGES:
            raise ValueError(f"stage_idx {stage_idx} >= {NUM_PIPELINE_STAGES}, use step_s9 instead")

        # 1.  input hash ??
        before_dict = copy.deepcopy(payload.to_json_dict())
        if tier == 3 and (self.exact_only or self.exact_replay_only):
            return self._skip_step(
                stage_idx=stage_idx,
                payload=payload,
                cache=cache,
                question_id=question_id,
                operator_version=operator_version,
                cache_only=cache_only,
                input_hash="",
            )

        identity_envelope = None
        exact_fields = None
        if self.exact_only:
            self._require_exact_cache(cache)
            identity_envelope = self._exact_cache_identity(payload, stage_idx, tier)
            exact_fields = EXACT_OPERATOR_INPUT_FIELDS[stage_idx]
        try:
            input_hash = (
                compute_exact_replay_input_hash(before_dict, stage_idx)
                if self.exact_replay_only
                else compute_input_hash(
                    before_dict,
                    stage_idx,
                    tier,
                    exact_fields=exact_fields,
                    identity_envelope=identity_envelope,
                )
            )
        except ValueError:
            if self.exact_only:
                raise
            # stage ??
            input_hash = ""

        exact_operator = None
        if self.exact_only:
            exact_operator = self._get_exact_operator(
                stage_idx, tier, identity_envelope
            )

        if tier == 3:
            return self._skip_step(
                stage_idx=stage_idx,
                payload=payload,
                cache=cache,
                question_id=question_id,
                operator_version=operator_version,
                cache_only=cache_only,
                input_hash=input_hash,
            )

        if input_hash:
            key = TransitionKey(
                question_id=question_id,
                stage_idx=stage_idx,
                tier=tier,
                input_payload_hash=input_hash,
                operator_version=operator_version,
            )
            cached = cache.lookup(key)
            if cached is not None:
                if self.exact_only or self.exact_replay_only:
                    restored = self._restore_exact_hit(
                        payload,
                        before_dict,
                        cached,
                    )
                    if restored is None:
                        cached = None
                    else:
                        new_payload, stage_cost_entries = restored
                else:
                    new_payload = type(payload).from_json_dict(cached)
                    stage_cost_entries = self._cached_stage_cost_entries(
                        cached, STAGE_NAMES[stage_idx]
                    )
            if cached is not None:
                new_payload.cost_tracker = self._merge_cached_stage_cost(
                    payload.cost_tracker, stage_cost_entries
                )
                cost = sum(float(entry["cost_dollar"]) for entry in stage_cost_entries)
                self._last_step_latency_ms = sum(
                    float(entry.get("latency_ms", 0.0)) for entry in stage_cost_entries
                )
                logger.debug(
                    "[qid=%d S%d T%d] Cache HIT", question_id, stage_idx + 1, tier + 1
                )
                return new_payload, cost, True

        # 2. Cache MISS ?
        if cache_only:
            raise RuntimeError(
                f"Cache-only replay MISS for qid={question_id} "
                f"stage={STAGE_NAMES[stage_idx]} tier={tier}"
            )
        if self.exact_replay_only:
            raise RuntimeError("Exact replay mode cannot execute or write operators")

        operator = exact_operator if self.exact_only else self._get_operator(stage_idx, tier)
        if operator is None:
            raise RuntimeError(
                f"No operator found for stage={STAGE_NAMES[stage_idx]} tier={tier}"
            )

        start = time.time()
        before_cost = payload.cost_tracker.total_cost_dollar()
        try:
            from _shared.llm_adapter import end_cost_context, start_cost_context

            start_cost_context(payload.cost_tracker, STAGE_NAMES[stage_idx])
            try:
                new_payload = self._call_operator(operator, payload, db_manager)
            finally:
                end_cost_context()
        except Exception as e:
            fatal_error = as_fatal_api_error(e)
            if fatal_error is not None:
                fatal_error.stage_index = stage_idx
                fatal_error.tier = tier
                fatal_error.original_error_type = getattr(
                    e,
                    "original_error_type",
                    type(e).__name__,
                )
                logger.error(
                    "[qid=%d S%d T%d] Fatal API error (%s); stopping collection: %s",
                    question_id, stage_idx + 1, tier + 1,
                    type(fatal_error).__name__, e,
                )
                raise fatal_error from e
            logger.error(
                "[qid=%d S%d T%d] Operator failed: %s",
                question_id, stage_idx + 1, tier + 1, e, exc_info=True,
            )
            raise

        latency_ms = (time.time() - start) * 1000.0
        after_cost = new_payload.cost_tracker.total_cost_dollar()
        cost = max(0.0, after_cost - before_cost)
        self._last_step_latency_ms = latency_ms

        # 3. ?
        if input_hash:
            after_dict = new_payload.to_json_dict()
            after_dict = self._inject_measured_step_metrics(
                after_dict,
                stage_name=STAGE_NAMES[stage_idx],
                latency_ms=latency_ms,
                cost_dollar=cost,
            )
            if self.exact_only:
                after_dict["_exact_cache"] = self._build_exact_cache_metadata(
                    before_dict, after_dict,
                )
                semantic_hash = ""
            else:
                from rl.transition_cache import compute_semantic_hash
                semantic_hash = compute_semantic_hash(after_dict, stage_idx)
            cache.store(
                key=key,
                output_payload=after_dict,
                source="rl_exploration",
                semantic_hash=semantic_hash,
                stored_hash=input_hash,
            )

        logger.debug(
            "[qid=%d S%d T%d] Executed: cost=%.6f, latency=%.0fms",
            question_id, stage_idx + 1, tier + 1, cost, latency_ms,
        )
        return new_payload, cost, False

    @staticmethod
    def _require_exact_cache(cache: Any) -> None:
        if getattr(cache, "allow_semantic_lookup", None) is not False:
            raise RuntimeError(
                "Exact-only cache requires semantic lookup to be explicitly disabled"
            )

    @staticmethod
    def _build_exact_cache_metadata(
        before_dict: Dict[str, Any],
        after_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload_delta = [
            [field, copy.deepcopy(value)]
            for field, value in after_dict.items()
            if field not in {"cost_tracker", "_exact_cache"}
            and before_dict.get(field) != value
        ]
        before_costs = before_dict.get("cost_tracker", {}).get("stage_costs", [])
        after_costs = after_dict.get("cost_tracker", {}).get("stage_costs", [])
        if after_costs[:len(before_costs)] != before_costs:
            raise RuntimeError("Exact operator rewrote existing stage cost history")
        return {
            "version": 1,
            "payload_delta": payload_delta,
            "stage_cost_delta": copy.deepcopy(after_costs[len(before_costs):]),
        }

    @staticmethod
    def _restore_exact_hit(
        payload: Any,
        before_dict: Dict[str, Any],
        cached: Dict[str, Any],
    ) -> Optional[Tuple[Any, List[Dict[str, Any]]]]:
        metadata = cached.get("_exact_cache") if isinstance(cached, dict) else None
        if not isinstance(metadata, dict) or metadata.get("version") != 1:
            return None
        if set(metadata) not in (
            {"version", "payload_delta", "stage_cost_delta"},
            {"version", "input_projection", "payload_delta", "stage_cost_delta"},
        ):
            return None
        payload_delta = metadata.get("payload_delta")
        stage_cost_delta = metadata.get("stage_cost_delta")
        if not isinstance(payload_delta, list) or not isinstance(stage_cost_delta, list):
            return None

        merged_payload = copy.deepcopy(before_dict)
        for item in payload_delta:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or item[0] in {"cost_tracker", "_exact_cache"}
            ):
                return None
            merged_payload[item[0]] = copy.deepcopy(item[1])
        if not all(isinstance(entry, dict) for entry in stage_cost_delta):
            return None
        return (
            type(payload).from_json_dict(merged_payload),
            copy.deepcopy(stage_cost_delta),
        )

    def _exact_cache_identity(
        self,
        payload: Any,
        stage_idx: int,
        tier: int,
    ) -> Dict[str, Any]:
        db_id = getattr(payload, "db_id", None)
        if not db_id or self.cache_identity is None:
            raise RuntimeError(
                f"Missing exact cache identity for db={db_id!r} stage={stage_idx} tier={tier}"
            )
        identity = self.cache_identity.get((db_id, stage_idx, tier))
        if identity is None:
            raise RuntimeError(
                f"Missing exact cache identity for db={db_id!r} stage={stage_idx} tier={tier}"
            )
        if identity.get("operator_package") != self.operator_pkg:
            raise RuntimeError(
                "Exact cache identity operator package does not match EnvironmentWrapper"
            )
        return identity

    def _get_exact_operator(
        self,
        stage_idx: int,
        tier: int,
        identity: Dict[str, Any],
    ) -> Callable:
        """Resolve env.chess without consulting the global package-agnostic registry."""
        cache_key = (stage_idx, tier)
        operator = self._exact_operator_cache.get(cache_key)
        if operator is None:
            package = importlib.import_module(self.operator_pkg)
            stage_functions = getattr(package, "stage_functions", None)
            fixed_model = getattr(package, "_fixed_model", None)
            if not isinstance(stage_functions, dict) or not callable(fixed_model):
                raise RuntimeError(
                    f"Exact operator package {self.operator_pkg!r} has no package-scoped resolver"
                )
            base_operator = stage_functions.get(STAGE_NAMES[stage_idx])
            if not callable(base_operator):
                raise RuntimeError(
                    f"No exact operator for {self.operator_pkg}.{STAGE_NAMES[stage_idx]}"
                )
            registry_tier = tier + 1
            if registry_tier not in {1, 2, 3}:
                raise RuntimeError(
                    f"Exact tier action {tier} has no 1-based registry tier {registry_tier}"
                )
            if stage_idx in {1, 2}:
                operator = base_operator
            else:
                from env._shared.model_config import TIER_LLM_MODELS

                expected_model = TIER_LLM_MODELS[registry_tier]
                operator = fixed_model(base_operator, expected_model)
            self._exact_operator_cache[cache_key] = operator

        source_module = getattr(operator, "__new2_source__", operator.__module__)
        if not (
            source_module == self.operator_pkg
            or source_module.startswith(f"{self.operator_pkg}.")
        ):
            raise RuntimeError("Exact operator callable does not belong to operator package")
        source_identity = identity["stage_source"]
        if source_identity["module"] != source_module:
            raise RuntimeError("Exact operator source module does not match identity")
        callable_sha256 = self._exact_operator_source_hashes.get(cache_key)
        if callable_sha256 is None:
            source_file = getattr(importlib.import_module(source_module), "__file__", None)
            if not source_file:
                raise RuntimeError("Exact operator source file cannot be resolved")
            callable_sha256 = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
            self._exact_operator_source_hashes[cache_key] = callable_sha256
        if source_identity["callable_sha256"] != callable_sha256:
            raise RuntimeError("Exact operator source hash does not match identity")

        model_config = identity["effective_model_config"]
        if stage_idx in {1, 2}:
            self._validate_exact_embedding_runtime(
                stage_idx, source_module, model_config
            )
        else:
            actual_model = getattr(operator, "__new2_model__", None)
            if actual_model != model_config["model"]:
                raise RuntimeError("Exact operator model does not match identity")
            if (
                model_config["temperature"] != 0
                or model_config["enable_thinking"] is not False
            ):
                raise RuntimeError("Exact operator effective config does not match env.chess")
        return operator

    @staticmethod
    def _validate_exact_embedding_runtime(
        stage_idx: int,
        source_module: str,
        model_config: Dict[str, Any],
    ) -> None:
        source = importlib.import_module(source_module)
        if stage_idx == 1:
            embedding = getattr(source, "EMBEDDING_FUNCTION", None)
        else:
            database_manager_module = importlib.import_module(
                source.DatabaseManager.__module__
            )
            embedding = getattr(database_manager_module, "EMBEDDING_FUNCTION", None)
        if embedding is None:
            raise RuntimeError("Exact embedding runtime is unavailable")

        actual_model = getattr(embedding, "model", None)
        if actual_model != model_config["model"]:
            raise RuntimeError("Exact embedding model does not match identity")
        client = getattr(embedding, "client", None)
        actual_base_url = getattr(client, "base_url", None)
        from rl.transition_cache import normalize_endpoint_identity

        try:
            actual_endpoint = normalize_endpoint_identity(str(actual_base_url))
        except ValueError as exc:
            raise RuntimeError("Exact embedding endpoint cannot be resolved") from exc
        if actual_endpoint != model_config["endpoint"]["url"]:
            raise RuntimeError("Exact embedding endpoint does not match identity")

    def run_physical_stage(
        self,
        *,
        stage_idx: int,
        tier: int,
        payload: Any,
        cache: Any,
        question_id: int,
        db_manager: Any = None,
        operator_version: int = 1,
        cache_only: bool = False,
    ) -> Tuple[Any, float, bool]:
        """Run one physical S1-S8 stage. This keeps step() semantics explicit."""
        return self.step(
            stage_idx=stage_idx,
            tier=tier,
            payload=payload,
            cache=cache,
            question_id=question_id,
            db_manager=db_manager,
            operator_version=operator_version,
            cache_only=cache_only,
        )

    def run_fixed_stages_after_decision(
        self,
        *,
        decision_idx: int,
        payload: Any,
        cache: Any,
        question_id: int,
        db_manager: Any = None,
        operator_version: int = 1,
        cache_only: bool = False,
    ) -> Tuple[Any, List[Tuple[int, float, bool, float]]]:
        """Run fixed embedding/retrieval stages after D0=S1."""
        if int(decision_idx) != 0:
            return payload, []

        current_payload = payload
        results: List[Tuple[int, float, bool, float]] = []
        for fixed_stage_idx in FIXED_PIPELINE_STAGE_IDXS:
            current_payload, cost, hit = self.run_physical_stage(
                stage_idx=fixed_stage_idx,
                tier=FIXED_STAGE_DEFAULT_TIER,
                payload=current_payload,
                cache=cache,
                question_id=question_id,
                db_manager=db_manager,
                operator_version=operator_version,
                cache_only=cache_only,
            )
            latency_ms = float(getattr(self, "_last_step_latency_ms", 0.0))
            results.append((fixed_stage_idx, cost, hit, latency_ms))
        return current_payload, results

    def _skip_step(
        self,
        stage_idx: int,
        payload: Any,
        cache: Any,
        question_id: int,
        operator_version: int,
        cache_only: bool,
        input_hash: str,
    ) -> Tuple[Any, float, bool]:
        """Handle explicit Skip without calling any operator."""
        if not self._allows_skip(stage_idx):
            raise ValueError(f"S{stage_idx + 1} does not allow Skip")
        if self.exact_only or self.exact_replay_only:
            new_payload = type(payload).from_json_dict(payload.to_json_dict())
            if self.exact_only:
                logger.debug("[qid=%d S%d] Skip (exact-only)", question_id, stage_idx + 1)
            self._last_step_latency_ms = 0.0
            return new_payload, 0.0, False
        if not input_hash:
            raise RuntimeError(
                f"Cannot compute Skip cache key for qid={question_id} "
                f"stage={STAGE_NAMES[stage_idx]}"
            )
        if cache_only:
            new_payload = type(payload).from_json_dict(payload.to_json_dict())
            logger.debug("[qid=%d S%d] Skip (deterministic replay)", question_id, stage_idx + 1)
            self._last_step_latency_ms = 0.0
            return new_payload, 0.0, False

        from rl.transition_cache import (
            TransitionKey,
            compute_semantic_hash,
        )

        key = TransitionKey(
            question_id=question_id,
            stage_idx=stage_idx,
            tier=3,
            input_payload_hash=input_hash,
            operator_version=operator_version,
        )
        cached = cache.lookup(key)
        if cached is None:
            output_payload = payload.to_json_dict()
            output_payload["_rl_skip_marker"] = {
                "stage_idx": stage_idx,
                "stage": STAGE_NAMES[stage_idx],
                "reason": "skip_action",
            }
            cache.store(
                key=key,
                output_payload=output_payload,
                source="skip",
                semantic_hash=compute_semantic_hash(output_payload, stage_idx),
                stored_hash=input_hash,
            )
            cached = output_payload

        new_payload = type(payload).from_json_dict(cached)
        logger.debug("[qid=%d S%d] Skip", question_id, stage_idx + 1)
        self._last_step_latency_ms = 0.0
        return new_payload, 0.0, False

    @staticmethod
    def _cached_stage_cost_entries(cached_payload: Dict[str, Any], stage_name: str) -> List[Dict[str, Any]]:
        cost_tracker = cached_payload.get("cost_tracker", {}) if isinstance(cached_payload, dict) else {}
        stage_costs = cost_tracker.get("stage_costs", [])
        return [
            copy.deepcopy(entry)
            for entry in stage_costs
            if entry.get("stage") == stage_name
        ]

    @staticmethod
    def _inject_measured_step_metrics(
        output_payload: Dict[str, Any],
        stage_name: str,
        latency_ms: float,
        cost_dollar: float,
    ) -> Dict[str, Any]:
        """Persist measured operator wall-clock latency for future cache HITs."""
        updated = copy.deepcopy(output_payload)
        cost_tracker = updated.setdefault("cost_tracker", {})
        stage_costs = cost_tracker.setdefault("stage_costs", [])
        matching = [entry for entry in stage_costs if entry.get("stage") == stage_name]

        if matching:
            previous_total = sum(float(entry.get("latency_ms", 0.0)) for entry in matching)
            if len(matching) == 1:
                matching[0]["latency_ms"] = float(latency_ms)
            elif previous_total > 0.0:
                scale = float(latency_ms) / previous_total
                for entry in matching:
                    entry["latency_ms"] = float(entry.get("latency_ms", 0.0)) * scale
            else:
                matching[0]["latency_ms"] = float(latency_ms)
                for entry in matching[1:]:
                    entry["latency_ms"] = 0.0
            for entry in matching:
                entry.setdefault("cost_dollar", float(cost_dollar))
        else:
            stage_costs.append({
                "stage": stage_name,
                "model": "operator_wall_clock",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": float(latency_ms),
                "cost_dollar": float(cost_dollar),
            })

        cost_tracker["total_latency_ms"] = sum(
            float(entry.get("latency_ms", 0.0)) for entry in stage_costs
        )
        return updated

    @staticmethod
    def _merge_cached_stage_cost(current_tracker: Any, stage_cost_entries: List[Dict[str, Any]]) -> Any:
        if any("cost_dollar" not in entry for entry in stage_cost_entries):
            raise ValueError("cached stage-cost entry lacks cost_dollar; rebuild the cache")
        merged = copy.deepcopy(current_tracker)
        for entry in stage_cost_entries:
            merged.total_input_tokens += int(entry.get("input_tokens", 0))
            merged.total_output_tokens += int(entry.get("output_tokens", 0))
            merged.total_latency_ms += float(entry.get("latency_ms", 0.0))
            merged.stage_costs.append(copy.deepcopy(entry))
        return merged

    def step_s9(
        self,
        payload: Any,                    # StatePayload
        db_manager: Any,                 # DatabaseManager
    ) -> Dict[str, Any]:
        """
         S9 (evaluation) ?,  TransitionCache, ?(? ? ?LLM )?

        Args:
            payload: S8 ?StatePayload
            db_manager: DatabaseManager 

        Returns:
            evaluation dict: {"exec_res": int, "exec_err": str}
        """
        #  ?S9 ?execute_sql + compare with gold, ? 
        operator = self._get_operator(8, 0)  # S9  T1
        if operator is None:
            raise RuntimeError("No operator found for S9 evaluation")

        try:
            new_payload = self._call_operator(operator, payload, db_manager)
        except Exception as exc:
            if isinstance(exc, ApiQuotaExhaustedError):
                raise
            if retryable_infrastructure_identity(exc) is not None:
                logger.warning(
                    "[qid=%d S9] Retryable evaluation failure: %s",
                    payload.question_id,
                    exc,
                )
            raise

        evaluation = getattr(new_payload, 'evaluation', None) or {}
        exec_res = evaluation.get("exec_res", 0)
        exec_err = evaluation.get("exec_err", "")

        return {"exec_res": exec_res, "exec_err": exec_err}

    # ---- Rollout ----

    def rollout_with_cache(
        self,
        question_id: int,
        action_sequence: List[int],      # len=6,  0/1/2/3
        cache,                           # TransitionCache
        state_encoder=None,
        operator_version: int = 1,
        cache_only: bool = False,
        evaluate_terminal: bool = True,
        capture_candidate_sql: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        ? ?

        Args:
            question_id:  ID
            action_sequence: [a0, ..., a5],  0=T1, 1=T2, 2=T3, 3=Skip
            cache: TransitionCache
            state_encoder: Optional state encoder for recorded states
            operator_version: 

        Returns:
            {
                "question_id": int,
                "actions": List[int],
                "states": List[np.ndarray],       # ?state
                "rewards": List[float],           # ?reward (sparse: ?
                "exec_res": int,                  # ?EX
                "total_cost": float,              # ?
                "stage_costs": List[float],       # 
                "total_latency_ms": float,
                "stage_latencies_ms": List[float],
                "cache_hits": List[bool],         # 
                "action_masks": List[List[float]],#  action mask
                "log_probs": List[float],         #  log P_F (?trainer )
            }
            Operator errors are propagated so the collector can defer the pair.
        """
        from env._shared.state_payload import StatePayload, CostTracker

        if len(action_sequence) != NUM_RL_STAGES:
            raise ValueError(
                f"action_sequence must contain {NUM_RL_STAGES} decision actions, "
                f"got {len(action_sequence)}"
            )

        # ?payload
        payload = StatePayload(
            question_id=question_id,
            db_id="",         # ?
            question="",      # ?
            cost_tracker=CostTracker(),
        )

        #  question metadata
        # (?Dataset  question/evidence/db_id ??
        db_id = self._get_question_db_id(question_id, cache) or ""

        # ?question 
        task = self._load_question(question_id)
        if task:
            payload.question = task.get("question", "")
            payload.evidence = task.get("evidence", "")
            payload.db_id = task.get("db_id", db_id)
            payload.gold_sql = task.get("SQL", "")
            db_id = payload.db_id or db_id

        db_manager = self._get_db_manager(db_id) if db_id and self.db_manager_factory else None

        states = []
        action_masks = []
        stage_costs = [0.0] * NUM_PIPELINE_STAGES
        stage_latencies_ms = [0.0] * NUM_PIPELINE_STAGES
        decision_stage_costs = []
        decision_stage_latencies_ms = []
        fixed_stage_costs = [0.0] * NUM_PIPELINE_STAGES
        fixed_stage_latencies_ms = [0.0] * NUM_PIPELINE_STAGES
        cache_hits = [False] * NUM_PIPELINE_STAGES
        skip_stages = []
        total_cost = 0.0
        total_latency_ms = 0.0
        question_emb = None
        candidate_sql = None

        for decision_idx, tier in enumerate(action_sequence):
            #  state (pre-step)
            if state_encoder is not None:
                # ?question_emb (?
                if decision_idx == 0:
                    question_emb = state_encoder.encode_question(
                        payload.question or "", payload.evidence or ""
                    )
                budget_remaining = max(0, self.total_budget - total_cost)
                tier_history = action_sequence[:decision_idx]
                cost_history = decision_stage_costs[:]
                valid_actions = self.get_action_mask(decision_idx, payload, budget_remaining)
                mask = [0.0] * NUM_TIERS
                for action in valid_actions:
                    mask[action] = 1.0
                action_masks.append(mask)
                state = state_encoder.encode(
                    payload, decision_idx, budget_remaining,
                    tier_history, cost_history, question_emb,
                    action_mask=np.asarray(mask, dtype=np.float32),
                )
                states.append(state)

            #  step
            physical_stage_idx = decision_to_pipeline_stage_idx(decision_idx)
            try:
                new_payload, cost, hit = self.step(
                    stage_idx=physical_stage_idx,
                    tier=tier,
                    payload=payload,
                    cache=cache,
                    question_id=question_id,
                    db_manager=db_manager,
                    operator_version=operator_version,
                    cache_only=cache_only,
                )
            except Exception as e:
                fatal_error = as_fatal_api_error(e)
                if fatal_error is not None:
                    fatal_error.stage_index = getattr(
                        e,
                        "stage_index",
                        physical_stage_idx,
                    )
                    fatal_error.tier = getattr(e, "tier", tier)
                    fatal_error.original_error_type = getattr(
                        e,
                        "original_error_type",
                        type(e).__name__,
                    )
                    logger.error(
                        "[qid=%d] Fatal API error (%s) at S%d T%d; stopping collection: %s",
                        question_id, type(fatal_error).__name__,
                        physical_stage_idx + 1, tier + 1, e,
                    )
                    raise fatal_error from e
                logger.error(
                    "[qid=%d] Rollout failed at S%d T%d: %s",
                    question_id, physical_stage_idx + 1, tier + 1, e,
                )
                e.stage_index = physical_stage_idx
                e.tier = tier
                e.original_error_type = type(e).__name__
                raise

            payload = new_payload
            if capture_candidate_sql and decision_idx == 4:
                candidates = getattr(payload, "candidates", None)
                if (
                    isinstance(candidates, list)
                    and candidates
                    and isinstance(candidates[0], dict)
                ):
                    candidate_sql = str(candidates[0].get("sql") or "")
                else:
                    candidate_sql = str(getattr(payload, "final_sql", "") or "")
            total_cost += cost
            latency_ms = float(getattr(self, "_last_step_latency_ms", 0.0))
            total_latency_ms += latency_ms
            stage_costs[physical_stage_idx] = cost
            stage_latencies_ms[physical_stage_idx] = latency_ms
            cache_hits[physical_stage_idx] = hit
            decision_cost = cost
            decision_latency_ms = latency_ms
            if tier == 3:
                skip_stages.append(decision_idx)

            payload, fixed_results = self.run_fixed_stages_after_decision(
                decision_idx=decision_idx,
                payload=payload,
                cache=cache,
                question_id=question_id,
                db_manager=db_manager,
                operator_version=operator_version,
                cache_only=cache_only,
            )
            for fixed_stage_idx, fixed_cost, fixed_hit, fixed_latency_ms in fixed_results:
                total_cost += fixed_cost
                total_latency_ms += fixed_latency_ms
                stage_costs[fixed_stage_idx] = fixed_cost
                stage_latencies_ms[fixed_stage_idx] = fixed_latency_ms
                fixed_stage_costs[fixed_stage_idx] = fixed_cost
                fixed_stage_latencies_ms[fixed_stage_idx] = fixed_latency_ms
                cache_hits[fixed_stage_idx] = fixed_hit

            decision_stage_costs.append(decision_cost)
            decision_stage_latencies_ms.append(decision_latency_ms)

        # S9: Evaluation is needed for live rollout/materialization, but replay
        # training already has terminal EX/cost in TrajectoryRecord.
        if not evaluate_terminal:
            eval_result = {"exec_res": 0, "exec_err": "terminal_evaluation_skipped"}
        elif db_manager is not None:
            eval_result = self.step_s9(payload, db_manager)
        else:
            eval_result = {"exec_res": 0, "exec_err": "no_db_manager"}

        #  final state (S8 )
        if state_encoder is not None:
            terminal_state = getattr(state_encoder, "terminal_state", None)
            if callable(terminal_state):
                final_state = terminal_state()
            else:
                budget_remaining = max(0, self.total_budget - total_cost)
                tier_history = list(action_sequence)
                cost_history = decision_stage_costs[:]
                final_state = state_encoder.encode(
                    payload, NUM_RL_STAGES - 1, budget_remaining,
                    tier_history, cost_history, question_emb,
                )
            states.append(final_state)

        trajectory = {
            "question_id": question_id,
            "db_id": payload.db_id,
            "actions": list(action_sequence),
            "states": states,
            "exec_res": eval_result["exec_res"],
            "total_cost": total_cost,
            "stage_costs": stage_costs,
            "decision_stage_costs": decision_stage_costs,
            "fixed_stage_costs": fixed_stage_costs,
            "total_latency_ms": total_latency_ms,
            "stage_latencies_ms": stage_latencies_ms,
            "decision_stage_latencies_ms": decision_stage_latencies_ms,
            "fixed_stage_latencies_ms": fixed_stage_latencies_ms,
            "cache_hits": cache_hits,
            "skip_stages": skip_stages,
            "fixed_stage_idxs": list(FIXED_PIPELINE_STAGE_IDXS),
            "fixed_stage_default_tier": FIXED_STAGE_DEFAULT_TIER,
            "action_masks": action_masks,
            "final_sql": payload.final_sql or "",
            "error": str(eval_result.get("exec_err") or ""),
            "log_probs": [],           # ?trainer 
            "reward": 0.0,             # ?caller 
        }
        if capture_candidate_sql:
            trajectory["candidate_sql"] = candidate_sql or ""
        return trajectory

    # ---- Action Mask ----

    def get_action_mask(
        self,
        stage_idx: int,
        payload: Any,
        budget_remaining: float,
    ) -> List[int]:
        """Return available actions for the current stage."""
        pipeline_stage_idx = decision_to_pipeline_stage_idx(stage_idx)
        if self._allows_skip(pipeline_stage_idx):
            return [0, 1, 2, 3]
        return [0, 1, 2]

    # ---- Internal Helpers ----

    def _get_operator(self, stage_idx: int, tier: int) -> Optional[Callable]:
        """Get an operator from TierRegistry, importing the operator package if needed."""
        import importlib
        from env._shared.tier_registry import TierRegistry

        stage_name = STAGE_NAMES[stage_idx]
        operator = TierRegistry.get(stage_name, tier + 1)  # tier 1-indexed in registry

        if operator is None:
            # ?( __init__.py  TierRegistry.register)
            try:
                importlib.import_module(self.operator_pkg)
                operator = TierRegistry.get(stage_name, tier + 1)
            except ImportError:
                pass

        if operator is None:
            # ?fallback: 
            try:
                pkg = importlib.import_module(self.operator_pkg)
                func_name = f"{stage_name}_tier{tier + 1}"
                operator = getattr(pkg, func_name, None)
            except (ImportError, AttributeError):
                pass

        return operator

    def _call_operator(
        self,
        operator: Callable,
        payload: Any,
        db_manager: Any,
    ) -> Any:
        """Call an operator with supported keyword arguments."""
        import inspect
        sig = inspect.signature(operator)
        params = sig.parameters

        kwargs = {}
        if "model" in params:
            kwargs["model"] = self.model
        if "db_manager" in params and db_manager is not None:
            kwargs["db_manager"] = db_manager

        try:
            return operator(payload, **kwargs)
        except TypeError:
            # :  payload
            return operator(payload)

    def _get_db_manager(self, db_id: str) -> Any:
        """Create a DatabaseManager if a factory is configured."""
        if self.db_manager_factory:
            return self.db_manager_factory(db_id)
        return None

    def _load_question(self, question_id: int) -> Optional[Dict[str, Any]]:
        """Load question metadata from the configured question map."""
        # Placeholder ??self.question_map
        if hasattr(self, '_question_map'):
            return self._question_map.get(question_id)
        return None

    def set_question_map(self, question_map: Dict[int, Dict[str, Any]]):
        """Set question metadata by question_id."""
        self._question_map = question_map

    def _get_question_db_id(self, question_id: int, cache) -> Optional[str]:
        """Infer db_id from a cached payload for this question_id."""
        import sqlite3
        if hasattr(cache, '_get_conn'):
            try:
                conn = cache._get_conn()
                row = conn.execute(
                    "SELECT output_payload FROM transitions WHERE question_id=? LIMIT 1",
                    (question_id,)
                ).fetchone()
                if row:
                    import json
                    data = json.loads(row[0])
                    return data.get("db_id")
            except Exception:
                pass
        return None


# ============================================================
# Module-level rollout_with_cache (convenience)
# ============================================================

def rollout_with_cache(
    question_id: int,
    action_sequence: List[int],
    env: EnvironmentWrapper,
    cache,                             # TransitionCache
    state_encoder=None,
    reward_fn: Optional[Callable] = None,
    cache_only: bool = False,
    evaluate_terminal: bool = True,
    capture_candidate_sql: bool = False,
) -> Optional[Dict[str, Any]]:
    """
     ? reward?

    Args:
        question_id:  ID
        action_sequence: 6 RL decision actions
        env: EnvironmentWrapper
        cache: TransitionCache
        state_encoder: Optional state encoder
        reward_fn: (exec_res, total_cost) ?float

    Returns:
        trajectory dict with "reward" field populated
    """
    traj = env.rollout_with_cache(
        question_id=question_id,
        action_sequence=action_sequence,
        cache=cache,
        state_encoder=state_encoder,
        cache_only=cache_only,
        evaluate_terminal=evaluate_terminal,
        capture_candidate_sql=capture_candidate_sql,
    )
    if traj is not None and reward_fn is not None:
        traj["reward"] = reward_fn(
            exec_res=traj["exec_res"],
            total_cost=traj["total_cost"],
        )
    return traj

