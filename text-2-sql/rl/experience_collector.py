"""Collect fixed-route trajectories for offline routing policy learning."""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from .api_guard import ApiInfrastructureError, as_fatal_api_error
from .reward import RewardConfig, RewardEvaluator
from .route_ladder import RouteTemplate, route_key, route_name
from .trajectory_buffer import TrajectoryBuffer, TrajectoryRecord

logger = logging.getLogger(__name__)

class ExperienceCollector:
    """Collect scheduled routes with exact transition reuse."""

    def __init__(
        self,
        env,
        cache,
        traj_buffer: TrajectoryBuffer,
        reward_fn: Optional[Callable[[int, float], float]] = None,
        reward_alpha: float = 25.0,
        reward_beta: float = 2.0,
        reward_config: Optional[RewardConfig] = None,
        reward_evaluator: Optional[RewardEvaluator] = None,
        collection_policy: str = "adaptive",
        min_routes_per_query: int = 1,
        max_routes_per_query: Optional[int] = None,
        stop_on_first_success: bool = True,
        audit_route_rate: float = 0.0,
        route_templates: Optional[Sequence[RouteTemplate]] = None,
        route_names: Optional[List[str]] = None,
        n_workers: int = 4,
        autosave_path: Optional[str] = None,
        autosave_every: int = 0,
        failure_queue_path: Optional[str] = None,
        retry_deferred_pairs: bool = True,
    ):
        self.env = env
        self.cache = cache
        self.traj_buffer = traj_buffer
        if reward_fn is not None:
            raise ValueError("Custom reward_fn is no longer supported; use RewardConfig in rl.reward")
        config = (
            reward_config
            or RewardConfig(beta=float(reward_beta), alpha=float(reward_alpha))
        ).normalized()
        self.reward_evaluator = reward_evaluator or RewardEvaluator(config)
        self.reward_config = self.reward_evaluator.config
        if collection_policy != "adaptive":
            raise ValueError("fixed-route collection requires collection_policy=adaptive")
        self.collection_policy = collection_policy
        self.min_routes_per_query = max(1, int(min_routes_per_query))
        self.max_routes_per_query = max_routes_per_query
        self.stop_on_first_success = bool(stop_on_first_success)
        self.audit_route_rate = max(0.0, float(audit_route_rate))
        self.route_ladder = self._select_route_ladder(route_names, route_templates)
        self.n_workers = n_workers
        self.autosave_path = autosave_path
        self.autosave_every = max(0, int(autosave_every))
        self.failure_queue_path = Path(failure_queue_path) if failure_queue_path else None
        self.retry_deferred_pairs = bool(retry_deferred_pairs)
        self._total_cache_hits = 0
        self._total_cache_steps = 0
        self._autosave_completed_units = 0
        self._autosave_lock = Lock()
        self._failure_queue_lock = Lock()

    @staticmethod
    def _empty_failure_queue() -> Dict[str, Any]:
        return {
            "version": "failed_pair_queue_v1",
            "unresolved": [],
            "resolved": [],
        }

    def _read_failure_queue(self) -> Dict[str, Any]:
        path = self.failure_queue_path
        if path is None or not path.exists():
            return self._empty_failure_queue()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("version") != "failed_pair_queue_v1"
            or not isinstance(payload.get("unresolved"), list)
            or not isinstance(payload.get("resolved"), list)
        ):
            raise ValueError(f"Invalid failure pair queue: {path}")
        return payload

    def _write_failure_queue(self, payload: Dict[str, Any]) -> None:
        path = self.failure_queue_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _failure_identity(record: Dict[str, Any]) -> tuple[int, str]:
        return int(record["question_id"]), str(record["route_key"])

    def _record_infrastructure_failure(
        self, question_id: int, error: BaseException
    ) -> None:
        if self.failure_queue_path is None:
            return
        fallback_template = self.route_ladder[0] if len(self.route_ladder) == 1 else None
        actions = list(getattr(error, "route_actions", ()) or ())
        route_key_value = getattr(error, "route_key", None)
        if not route_key_value and actions:
            route_key_value = route_key(actions)
        if not route_key_value and fallback_template is not None:
            actions = list(fallback_template.actions)
            route_key_value = route_key(actions)
        route_name_value = getattr(error, "route_name", None)
        if not route_name_value and fallback_template is not None:
            route_name_value = fallback_template.name
        if not route_key_value:
            raise ValueError("Pair failure is missing route identity")
        stage_index = getattr(error, "stage_index", None)
        tier = getattr(error, "tier", None)
        now = datetime.now(timezone.utc).isoformat()
        identity = (int(question_id), str(route_key_value))
        with self._failure_queue_lock:
            payload = self._read_failure_queue()
            prior = next(
                (
                    record for record in payload["unresolved"]
                    if self._failure_identity(record) == identity
                ),
                None,
            )
            payload["unresolved"] = [
                record for record in payload["unresolved"]
                if self._failure_identity(record) != identity
            ]
            payload["resolved"] = [
                record for record in payload["resolved"]
                if self._failure_identity(record) != identity
            ]
            record = {
                "question_id": identity[0],
                "route_name": str(route_name_value or "unknown"),
                "route_key": identity[1],
                "stage_index": int(stage_index) if stage_index is not None else None,
                "stage_name": f"S{int(stage_index) + 1}" if stage_index is not None else None,
                "tier": int(tier) if tier is not None else None,
                "tier_name": f"T{int(tier) + 1}" if tier is not None else None,
                "error_type": type(error).__name__,
                "cause_type": str(
                    getattr(error, "original_error_type", type(error).__name__)
                ),
                "attempts": int(prior.get("attempts", 0)) + 1 if prior else 1,
                "first_failed_at": prior.get("first_failed_at", now) if prior else now,
                "last_failed_at": now,
            }
            payload["unresolved"].append(record)
            payload["unresolved"].sort(
                key=lambda item: self._failure_identity(item)
            )
            self._write_failure_queue(payload)

    def _resolve_infrastructure_failures(
        self, question_id: int, trajectories: Sequence[Dict[str, Any]]
    ) -> None:
        if self.failure_queue_path is None or not trajectories:
            return
        identities = {
            (int(question_id), str(traj.get("metadata", {}).get("route_key", "")))
            for traj in trajectories
        }
        identities.discard((int(question_id), ""))
        if not identities:
            return
        with self._failure_queue_lock:
            payload = self._read_failure_queue()
            moved = [
                record for record in payload["unresolved"]
                if self._failure_identity(record) in identities
            ]
            if not moved:
                return
            payload["unresolved"] = [
                record for record in payload["unresolved"]
                if self._failure_identity(record) not in identities
            ]
            resolved_at = datetime.now(timezone.utc).isoformat()
            resolved_by_identity = {
                self._failure_identity(record): record for record in payload["resolved"]
            }
            for record in moved:
                resolved_by_identity[self._failure_identity(record)] = {
                    **record,
                    "resolved_at": resolved_at,
                }
            payload["resolved"] = [
                resolved_by_identity[key] for key in sorted(resolved_by_identity)
            ]
            self._write_failure_queue(payload)

    @staticmethod
    def _select_route_ladder(
        route_names: Optional[List[str]],
        available_templates: Optional[Sequence[RouteTemplate]] = None,
    ) -> List[Any]:
        if available_templates is None:
            raise ValueError("route templates must be supplied by the collection design")
        templates = available_templates
        if not route_names:
            return list(templates)
        wanted = [str(name).strip() for name in route_names if str(name).strip()]
        available = {template.name: template for template in templates}
        missing = [name for name in wanted if name not in available]
        if missing:
            raise ValueError(
                f"Unknown route_names: {missing}. Available: {sorted(available)}"
            )
        return [available[name] for name in wanted]

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate observed across completed collection calls."""
        return self._total_cache_hits / max(self._total_cache_steps, 1)

    def _maybe_autosave(self, completed_units: int = 1) -> None:
        if not self.autosave_path or self.autosave_every <= 0:
            return
        with self._autosave_lock:
            self._autosave_completed_units += max(0, int(completed_units))
            if self._autosave_completed_units < self.autosave_every:
                return
            self.traj_buffer.save(self.autosave_path)
            self._autosave_completed_units = 0

    def collect_from_questions(
        self,
        question_ids: Iterable[int],
        max_queries: Optional[int] = None,
        stop_cache_hit_rate: Optional[float] = None,
        trajectories_per_query: int = 2,
        min_routes_per_query: Optional[int] = None,
        max_routes_per_query: Optional[int] = None,
        stop_on_first_success: Optional[bool] = None,
        audit_route_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        qids = list(question_ids)
        if max_queries is not None:
            qids = qids[:max_queries]
        trajectories_per_query = max(1, int(trajectories_per_query))
        return self._collect_adaptive_from_questions(
            qids,
            max_routes_per_query=max_routes_per_query or self.max_routes_per_query or trajectories_per_query,
            min_routes_per_query=min_routes_per_query or self.min_routes_per_query,
            stop_on_first_success=self.stop_on_first_success
            if stop_on_first_success is None
            else bool(stop_on_first_success),
            audit_route_rate=self.audit_route_rate if audit_route_rate is None else float(audit_route_rate),
            stop_cache_hit_rate=stop_cache_hit_rate,
        )

    def _collect_route(
        self,
        question_id: int,
        actions: List[int],
        trajectory_idx: int = 0,
        route_template_name: Optional[str] = None,
        collection_logprob: Optional[float] = None,
        policy_logprob: Optional[float] = None,
        policy_logprob_source: str = "not_scored",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            traj = self.env.rollout_with_cache(
                question_id=question_id,
                action_sequence=actions,
                cache=self.cache,
            )
        except Exception as exc:
            fatal_error = as_fatal_api_error(exc)
            if fatal_error is not None:
                fatal_error.question_id = int(question_id)
                fatal_error.route_actions = list(actions)
                fatal_error.route_key = route_key(actions)
                fatal_error.route_name = route_template_name or route_name(actions)
                raise fatal_error from exc
            raise
        if traj is None:
            return None

        reward_result = self.reward_evaluator.evaluate(
            exec_res=traj["exec_res"],
            total_cost=traj["total_cost"],
            decision_stage_costs=traj.get("decision_stage_costs", []),
        )
        key = route_key(traj["actions"])
        metadata = {
            "cache_hits": list(traj.get("cache_hits", [])),
            "skip_stages": list(traj.get("skip_stages", [])),
            "action_masks": list(traj.get("action_masks", [])),
            "decision_stage_costs": list(traj.get("decision_stage_costs", [])),
            "decision_stage_latencies_ms": list(traj.get("decision_stage_latencies_ms", [])),
            "fixed_stage_costs": list(traj.get("fixed_stage_costs", [])),
            "fixed_stage_latencies_ms": list(traj.get("fixed_stage_latencies_ms", [])),
            "fixed_stage_idxs": list(traj.get("fixed_stage_idxs", [])),
            "fixed_stage_default_tier": traj.get("fixed_stage_default_tier"),
            "route_key": key,
            "route_name": route_template_name or route_name(traj["actions"]),
            **self.reward_evaluator.metadata(reward_result),
            "operator_version": 1,
            "policy_logprob": policy_logprob,
            "policy_logprob_source": policy_logprob_source,
            "collection_policy": self.collection_policy,
            "collection_logprob": collection_logprob,
            "trajectory_idx": int(trajectory_idx),
            "trajectory_outcome": "success" if int(traj["exec_res"]) == 1 else "failed",
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        traj["reward"] = reward_result.reward
        traj["log_reward"] = reward_result.log_reward
        traj["metadata"] = metadata
        record = TrajectoryRecord(
            question_id=question_id,
            actions=list(traj["actions"]),
            reward=reward_result.reward,
            exec_res=traj["exec_res"],
            total_cost=traj["total_cost"],
            stage_costs=list(traj["stage_costs"]),
            total_latency_ms=float(traj.get("total_latency_ms", 0.0)),
            stage_latencies_ms=list(traj.get("stage_latencies_ms", [])),
            decision_stage_costs=list(traj.get("decision_stage_costs", [])),
            decision_stage_latencies_ms=list(traj.get("decision_stage_latencies_ms", [])),
            fixed_stage_costs=list(traj.get("fixed_stage_costs", [])),
            fixed_stage_latencies_ms=list(traj.get("fixed_stage_latencies_ms", [])),
            metadata=metadata,
        )
        replaced_existing = self.traj_buffer.upsert_by_question_route(record)
        if replaced_existing:
            logger.info(
                "TrajectoryBuffer upsert replaced existing record: qid=%s route=%s",
                question_id,
                key,
            )
        self._maybe_autosave()
        traj["_record"] = record
        return traj

    def _collect_adaptive_from_questions(
        self,
        qids: List[int],
        max_routes_per_query: int,
        min_routes_per_query: int,
        stop_on_first_success: bool,
        audit_route_rate: float,
        stop_cache_hit_rate: Optional[float],
    ) -> Dict[str, float]:
        trajectories: List[Dict[str, Any]] = []
        total_hits = 0
        total_steps = 0
        max_workers = max(1, self.n_workers)
        total_queries = len(qids)
        qid_iter = iter(enumerate(qids, start=1))
        deferred_infrastructure: List[tuple[int, int]] = []

        def accept_trajectories(qid: int, q_trajs: Sequence[Dict[str, Any]]) -> None:
            nonlocal total_hits, total_steps
            self._resolve_infrastructure_failures(qid, q_trajs)
            for traj in q_trajs:
                trajectories.append(traj)
                total_hits += sum(traj.get("cache_hits", []))
                total_steps += len(traj.get("cache_hits", [])) or len(traj.get("actions", []))
                self._total_cache_hits += sum(traj.get("cache_hits", []))
                self._total_cache_steps += len(traj.get("cache_hits", [])) or len(traj.get("actions", []))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for _ in range(max_workers):
                try:
                    query_position, qid = next(qid_iter)
                except StopIteration:
                    break
                futures[pool.submit(
                    self._collect_adaptive_for_question,
                    qid,
                    query_position,
                    total_queries,
                    max_routes_per_query,
                    min_routes_per_query,
                    stop_on_first_success,
                    audit_route_rate,
                )] = (qid, query_position)

            while futures:
                fut = next(as_completed(futures))
                qid, query_position = futures.pop(fut)
                try:
                    q_trajs = fut.result()
                except Exception as exc:
                    fatal_error = as_fatal_api_error(exc)
                    if fatal_error is not None:
                        if (
                            isinstance(fatal_error, ApiInfrastructureError)
                            and self.failure_queue_path is not None
                        ):
                            self._record_infrastructure_failure(qid, fatal_error)
                            deferred_infrastructure.append((qid, query_position))
                            logger.warning(
                                "Deferred infrastructure failure: query=%d/%d qid=%s; "
                                "continuing normal pairs before one sequential retry",
                                query_position, total_queries, qid,
                            )
                            q_trajs = []
                        else:
                            for pending in futures:
                                pending.cancel()
                            logger.error(
                                "Fatal API error (%s); stopping collection: "
                                "query=%d/%d qid=%s error=%s",
                                type(fatal_error).__name__, query_position,
                                total_queries, qid, exc,
                            )
                            raise fatal_error from exc
                    else:
                        if self.failure_queue_path is not None:
                            self._record_infrastructure_failure(qid, exc)
                            deferred_infrastructure.append((qid, query_position))
                            logger.warning(
                                "Deferred pair failure (%s): query=%d/%d qid=%s; "
                                "continuing normal pairs before one sequential retry",
                                type(exc).__name__, query_position, total_queries, qid,
                            )
                        else:
                            logger.warning(
                                "Adaptive collection failed for query=%d/%d qid=%s: %s",
                                query_position, total_queries, qid, exc,
                            )
                        q_trajs = []
                accept_trajectories(qid, q_trajs)

                if stop_cache_hit_rate is not None and total_steps:
                    if total_hits / total_steps >= stop_cache_hit_rate:
                        for pending in futures:
                            pending.cancel()
                        break

                try:
                    next_query_position, next_qid = next(qid_iter)
                except StopIteration:
                    continue
                futures[pool.submit(
                    self._collect_adaptive_for_question,
                    next_qid,
                    next_query_position,
                    total_queries,
                    max_routes_per_query,
                    min_routes_per_query,
                    stop_on_first_success,
                    audit_route_rate,
                )] = (next_qid, next_query_position)

        retry_queue = deferred_infrastructure if self.retry_deferred_pairs else ()
        for qid, query_position in retry_queue:
            try:
                q_trajs = self._collect_adaptive_for_question(
                    qid,
                    query_position,
                    total_queries,
                    max_routes_per_query,
                    min_routes_per_query,
                    stop_on_first_success,
                    audit_route_rate,
                )
            except Exception as exc:
                fatal_error = as_fatal_api_error(exc)
                if (
                    isinstance(fatal_error, ApiInfrastructureError)
                    and self.failure_queue_path is not None
                ):
                    self._record_infrastructure_failure(qid, fatal_error)
                    logger.warning(
                        "Deferred pair retry still has infrastructure failure: "
                        "query=%d/%d qid=%s; leaving it unresolved for --resume",
                        query_position, total_queries, qid,
                    )
                    continue
                if fatal_error is not None:
                    raise fatal_error from exc
                if self.failure_queue_path is not None:
                    self._record_infrastructure_failure(qid, exc)
                    logger.warning(
                        "Deferred pair retry still has pair failure (%s): "
                        "query=%d/%d qid=%s; leaving it unresolved for --resume",
                        type(exc).__name__, query_position, total_queries, qid,
                    )
                else:
                    logger.warning(
                        "Deferred pair retry failed for query=%d/%d qid=%s: %s",
                        query_position, total_queries, qid, exc,
                    )
                continue
            accept_trajectories(qid, q_trajs)

        return {
            "n_collected": len(trajectories),
            "cache_hit_rate": total_hits / max(total_steps, 1),
            "avg_reward": float(np.mean([t["reward"] for t in trajectories])) if trajectories else 0.0,
            "avg_ex": float(np.mean([t["exec_res"] for t in trajectories])) if trajectories else 0.0,
            "avg_cost": float(np.mean([t["total_cost"] for t in trajectories])) if trajectories else 0.0,
            "buffer_size": len(self.traj_buffer),
        }

    def _collect_adaptive_for_question(
        self,
        question_id: int,
        query_position: int,
        total_queries: int,
        max_routes_per_query: int,
        min_routes_per_query: int,
        stop_on_first_success: bool,
        audit_route_rate: float,
    ) -> List[Dict[str, Any]]:
        max_routes = max(1, min(int(max_routes_per_query), len(self.route_ladder)))
        collected: List[Dict[str, Any]] = []
        success_idx: Optional[int] = None

        for idx, template in enumerate(self.route_ladder[:max_routes]):
            logger.info(
                "Collect progress: query=%d/%d qid=%s template=%d/%d route=%s actions=%s buffer_size=%d",
                query_position,
                total_queries,
                question_id,
                idx + 1,
                max_routes,
                template.name,
                template.actions,
                len(self.traj_buffer),
            )
            traj = self._collect_route(
                question_id=question_id,
                actions=list(template.actions),
                trajectory_idx=idx,
                route_template_name=template.name,
                collection_logprob=None,
                policy_logprob=None,
                policy_logprob_source="not_scored",
                extra_metadata={"route_rank": template.rank},
            )
            if traj is None:
                logger.warning(
                    "Collect route skipped: query=%d/%d qid=%s template=%d/%d route=%s buffer_size=%d",
                    query_position,
                    total_queries,
                    question_id,
                    idx + 1,
                    max_routes,
                    template.name,
                    len(self.traj_buffer),
                )
                continue
            collected.append(traj)
            logger.info(
                "Collect route done: query=%d/%d qid=%s template=%d/%d route=%s exec_res=%s cost=%.6f reward=%.6g buffer_size=%d",
                query_position,
                total_queries,
                question_id,
                idx + 1,
                max_routes,
                template.name,
                traj.get("exec_res"),
                float(traj.get("total_cost", 0.0)),
                float(traj.get("reward", 0.0)),
                len(self.traj_buffer),
            )
            if int(traj.get("exec_res", 0)) == 1 and success_idx is None:
                success_idx = idx
                if stop_on_first_success:
                    break

        audited = False
        if collected and success_idx == 0 and audit_route_rate > 0.0:
            if random.random() < audit_route_rate and len(collected) < max_routes:
                audit_template = self.route_ladder[min(1, len(self.route_ladder) - 1)]
                traj = self._collect_route(
                    question_id=question_id,
                    actions=list(audit_template.actions),
                    trajectory_idx=len(collected),
                    route_template_name=audit_template.name,
                    collection_logprob=None,
                    policy_logprob=None,
                    policy_logprob_source="not_scored",
                    extra_metadata={"route_rank": audit_template.rank, "audit_route": True},
                )
                if traj is not None:
                    collected.append(traj)
                    audited = True

        if not collected:
            return []
        if success_idx == 0:
            outcome = "cheap_success_audited" if audited else "cheap_success"
        elif success_idx is not None:
            outcome = "rescued_by_stronger_route"
        else:
            outcome = "unresolved_hard_negative"
        for traj in collected:
            metadata = traj.get("metadata", {})
            metadata["query_outcome"] = outcome
            if success_idx is not None:
                metadata["rescue_rank"] = success_idx
        return collected

