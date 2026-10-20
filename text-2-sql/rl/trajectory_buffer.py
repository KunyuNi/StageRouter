"""
TrajectoryBuffer — 高分轨迹元数据索引池

核心设计:
  - 不存 state vector (避免跨训练 representation staleness)
  - 只存元数据: (qid, actions, reward, exec_res, cost)
  - Replay 时用当前 encoder 重编码 → 零 LLM 调用
  - 淘汰策略: top 70% reward + 30% random (保留高奖励, 保持多样性)

与 PrioritizedReplayBuffer 的区别:
  - PRB: 存单步 (s,a,r,s') transition, 含 state vector
  - TrajectoryBuffer: 存完整轨迹元数据, 不含 state vector
  - 两个独立使用, 不互相替代

Usage:
  from rl.trajectory_buffer import TrajectoryBuffer, TrajectoryRecord
  buf = TrajectoryBuffer(capacity=50000)
  buf.add(TrajectoryRecord(qid=1, actions=[0,1,0,2,0,0],
                           reward=0.8, exec_res=1, total_cost=0.025))
  records = buf.sample_high_reward(16)
  trajectories = buf.replay_as_trajectories(32, env, cache,
                                            state_encoder, reward_fn)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .reward import REWARD_METADATA_KEYS

logger = logging.getLogger(__name__)


def _atomic_json_dump(data: Any, file_path: str) -> None:
    target = os.path.abspath(file_path)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.",
        suffix=".tmp",
        dir=parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        if os.name == "posix":
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


# ============================================================
# TrajectoryRecord — 轨迹元数据
# ============================================================

@dataclass(frozen=True)
class TrajectoryRecord:
    """
    一条完整轨迹的元数据 — 不含 state vector。

    Attributes:
        question_id: BIRD 问题 ID
        actions: 6 个 RL 决策 action，对应 [S1, S4, S5, S6, S7, S8]
        reward: Collected route reward
        exec_res: 最终 EX (1=正确, 0=错误)
        total_cost: Total monetary cost in dollars
        stage_costs: 8 个物理阶段花费明细，对应 S1-S8
        decision_stage_costs: 6 个决策阶段花费，对应 [S1, S4, S5, S6, S7, S8]
        fixed_stage_costs: 8 个物理阶段固定花费明细，当前只记录 S2/S3
        total_latency_ms: 总耗时 (毫秒)
        stage_latencies_ms: 8 个物理阶段耗时明细，对应 S1-S8
        decision_stage_latencies_ms: 6 个决策阶段耗时，对应 [S1, S4, S5, S6, S7, S8]
        fixed_stage_latencies_ms: 8 个物理阶段固定耗时明细，当前只记录 S2/S3
        metadata: 扩展字段
    """
    question_id: int
    actions: List[int]            # len=6, 每个 0/1/2/3
    reward: float
    exec_res: int                 # 0 or 1
    total_cost: float = 0.0
    stage_costs: List[float] = field(default_factory=list)  # len=8
    decision_stage_costs: List[float] = field(default_factory=list)  # len=6
    fixed_stage_costs: List[float] = field(default_factory=list)  # len=8
    total_latency_ms: float = 0.0
    stage_latencies_ms: List[float] = field(default_factory=list)  # len=8
    decision_stage_latencies_ms: List[float] = field(default_factory=list)  # len=6
    fixed_stage_latencies_ms: List[float] = field(default_factory=list)  # len=8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "actions": self.actions,
            "reward": self.reward,
            "exec_res": self.exec_res,
            "total_cost": self.total_cost,
            "stage_costs": self.stage_costs,
            "decision_stage_costs": self.decision_stage_costs,
            "fixed_stage_costs": self.fixed_stage_costs,
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies_ms": self.stage_latencies_ms,
            "decision_stage_latencies_ms": self.decision_stage_latencies_ms,
            "fixed_stage_latencies_ms": self.fixed_stage_latencies_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryRecord":
        return cls(
            question_id=d["question_id"],
            actions=d.get("actions", []),
            reward=d.get("reward", 0.0),
            exec_res=d.get("exec_res", 0),
            total_cost=d.get("total_cost", 0.0),
            stage_costs=d.get("stage_costs", []),
            decision_stage_costs=d.get("decision_stage_costs", []),
            fixed_stage_costs=d.get("fixed_stage_costs", []),
            total_latency_ms=d.get("total_latency_ms", 0.0),
            stage_latencies_ms=d.get("stage_latencies_ms", []),
            decision_stage_latencies_ms=d.get("decision_stage_latencies_ms", []),
            fixed_stage_latencies_ms=d.get("fixed_stage_latencies_ms", []),
            metadata=d.get("metadata", {}),
        )


# ============================================================
# TrajectoryBuffer
# ============================================================

class TrajectoryBuffer:
    """
    高分轨迹元数据索引池。

    Attributes:
        capacity: 最大容量
        records: List[TrajectoryRecord]
        _sorted_by_reward: 是否按 reward 排序 (惰性排序)
        _reward_dirty: 标记是否需要重新排序
    """

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.records: List[TrajectoryRecord] = []
        self._reward_dirty = True
        self._lock = threading.RLock()
        self.last_replay_cache_miss_skipped_count = 0

    # ---- CRUD ----

    def add(self, record: TrajectoryRecord):
        """
        添加一条轨迹记录。

        若超过容量, 触发淘汰: 保留 top 70% + 随机 30%。
        """
        with self._lock:
            self.records.append(record)
            self._reward_dirty = True

            if len(self.records) > self.capacity:
                self._evict()

    def add_batch(self, records: List[TrajectoryRecord]):
        """批量添加。"""
        with self._lock:
            for r in records:
                self.records.append(r)
            self._reward_dirty = True

            if len(self.records) > self.capacity:
                self._evict()

    def replace(self, old_record: TrajectoryRecord, new_record: TrajectoryRecord):
        """Atomically replace one existing record without changing buffer length."""
        with self._lock:
            for idx, record in enumerate(self.records):
                if record is old_record:
                    self.records[idx] = new_record
                    self._reward_dirty = True
                    return
            raise ValueError("old_record not found in TrajectoryBuffer")

    def upsert_by_question_route(self, record: TrajectoryRecord) -> bool:
        """Add or replace one record keyed by (question_id, route_key)."""
        new_key = _record_route_key(record)
        with self._lock:
            for idx, existing in enumerate(self.records):
                if int(existing.question_id) == int(record.question_id) and _record_route_key(existing) == new_key:
                    self.records[idx] = record
                    self._reward_dirty = True
                    return True
            self.records.append(record)
            self._reward_dirty = True
            if len(self.records) > self.capacity:
                self._evict()
            return False

    def _evict(self):
        """
        淘汰策略: 保留 top 70% reward + 30% random。

        保证高奖励轨迹不丢失, 同时保留探索多样性。
        """
        if len(self.records) <= self.capacity:
            return

        self._sort_by_reward()
        keep_top = int(self.capacity * 0.7)
        keep_random = self.capacity - keep_top

        # top 70%: 最高 reward 的
        top = self.records[:keep_top]

        # random 30%: 从剩余中随机采样
        remaining = self.records[keep_top:]
        if len(remaining) > keep_random:
            random_sample = random.sample(remaining, keep_random)
        else:
            random_sample = remaining

        self.records = top + random_sample
        self._reward_dirty = True
        logger.debug(
            "TrajectoryBuffer evicted: %d → %d (top=%d, random=%d)",
            len(self.records) + len(remaining) - keep_random,
            len(self.records), keep_top, len(random_sample),
        )

    # ---- 排序 ----

    def _sort_by_reward(self):
        """按 reward 降序排序 (惰性)。"""
        if self._reward_dirty:
            self.records.sort(key=lambda r: r.reward, reverse=True)
            self._reward_dirty = False

    # ---- 采样 ----

    def sample_high_reward(self, n: int) -> List[TrajectoryRecord]:
        """
        采样 top-N 高 reward 轨迹。

        Args:
            n: 采样数量

        Returns:
            List[TrajectoryRecord] — 最多 n 条, 可能少于 n
        """
        with self._lock:
            self._sort_by_reward()
            return self.records[:min(n, len(self.records))]

    def sample_by_exec_res(self, n: int) -> List[TrajectoryRecord]:
        """
        按 EX 采样 (reward stale 时的 fallback)。

        优先采样 EX=1 的轨迹 (不限 reward 值)。

        Args:
            n: 采样数量

        Returns:
            List[TrajectoryRecord] — EX=1 和 EX=0 各半
        """
        with self._lock:
            ex1 = [r for r in self.records if r.exec_res == 1]
            ex0 = [r for r in self.records if r.exec_res == 0]

            n1 = min(n // 2, len(ex1))
            n0 = min(n - n1, len(ex0))
            if n1 + n0 < n:
                # 不够的从任意中补
                n1 = min(n, len(ex1))
                n0 = min(n - n1, len(ex0))

            result = random.sample(ex1, n1) if ex1 and n1 > 0 else []
            result += random.sample(ex0, n0) if ex0 and n0 > 0 else []
            return result

    def sample_mixed(
        self,
        n: int,
        high_reward_ratio: float = 0.6,
        reward_stale: bool = False,
    ) -> List[TrajectoryRecord]:
        """
        混合采样: 高奖励 + 随机。

        Args:
            n: 采样数量
            high_reward_ratio: 高奖励轨迹占比
            reward_stale: True → 用 exec_res 代替 reward

        Returns:
            List[TrajectoryRecord]
        """
        with self._lock:
            if reward_stale:
                # reward 过期 → 用 EX 采样
                return self.sample_by_exec_res(n)

            n_high = int(n * high_reward_ratio)
            n_random = n - n_high

            self._sort_by_reward()
            top = self.records[:max(n_high, min(len(self.records), n_high * 2))]
            high_sample = random.sample(top, min(n_high, len(top))) if top else []

            # random 从未选中的记录中取；TrajectoryRecord 内含 list/dict, 不能直接 set().
            selected_ids = {id(r) for r in high_sample}
            rest = [r for r in self.records if id(r) not in selected_ids]
            if rest and n_random > 0:
                random_sample = random.sample(rest, min(n_random, len(rest)))
            else:
                random_sample = []

            return high_sample + random_sample

    def sample_query_balanced(
        self,
        n: int,
        reward_stale: bool = False,
        routes_per_query: int = 2,
    ) -> List[TrajectoryRecord]:
        """Sample qids first, then random routes within each qid, with replacement.

        The active cold-start buffer is a multi-route counterfactual dataset: one qid
        usually has several route-ladder rollouts. Training should therefore see
        multiple routes for the same qid instead of always taking the highest-reward
        route first.
        """
        with self._lock:
            if n <= 0 or not self.records:
                return []

            groups: Dict[int, List[TrajectoryRecord]] = {}
            for record in self.records:
                groups.setdefault(record.question_id, []).append(record)

            qids = list(groups.keys())
            routes_per_query = max(1, int(routes_per_query))
            selected: List[TrajectoryRecord] = []

            while len(selected) < n and qids:
                qid = random.choice(qids)
                records = groups[qid]
                for _ in range(routes_per_query):
                    selected.append(random.choice(records))
                    if len(selected) >= n:
                        break

            return selected

    # ---- Replay (核心路径) ----

    def replay_as_trajectories(
        self,
        n: int,
        env,                    # EnvironmentWrapper
        cache,                  # TransitionCache
        state_encoder,
        reward_evaluator,
        reward_stale: bool = False,
        routes_per_query: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        从 TrajectoryBuffer 采样 → 全部 TransitionCache HIT → 零 LLM → 构建 Trajectory。

        这是"改 RL 模型重训零 LLM 调用"的核心路径:
          1. 采样元数据 (不含 state vector)
          2. rollout_with_cache(qid, actions) — 全部命中 TransitionCache
          3. 用当前 encoder 重编码 — 解决 representation staleness
          4. 计算当前 reward

        Args:
            n: 采样数量
            env: EnvironmentWrapper 实例
            cache: TransitionCache 实例
            state_encoder: Current routing-state encoder
            reward_evaluator: 唯一 reward evaluator
            reward_stale: True 时才按当前 reward 版本重算

        Returns:
            List[Dict] — 完整 trajectory dict, 含 states, actions, rewards, log_probs
        """
        records = self.sample_query_balanced(
            n,
            reward_stale=reward_stale,
            routes_per_query=routes_per_query,
        )
        trajectories = []
        skipped = 0

        for rec in records:
            try:
                # rollout_with_cache: 全部 TransitionCache HIT → 零 LLM
                traj = env.rollout_with_cache(
                    question_id=rec.question_id,
                    action_sequence=rec.actions,
                    cache=cache,
                    state_encoder=state_encoder,
                    cache_only=True,
                    evaluate_terminal=False,
                )
                if traj is None:
                    skipped += 1
                    continue
                if traj is not None:
                    traj["exec_res"] = rec.exec_res
                    traj["total_cost"] = rec.total_cost
                    traj["stage_costs"] = list(rec.stage_costs or traj.get("stage_costs", []))
                    traj["decision_stage_costs"] = list(rec.decision_stage_costs or traj.get("decision_stage_costs", []))
                    traj["fixed_stage_costs"] = list(rec.fixed_stage_costs or traj.get("fixed_stage_costs", []))
                    traj["total_latency_ms"] = float(rec.total_latency_ms or traj.get("total_latency_ms", 0.0))
                    traj["stage_latencies_ms"] = list(rec.stage_latencies_ms or traj.get("stage_latencies_ms", []))
                    traj["decision_stage_latencies_ms"] = list(rec.decision_stage_latencies_ms or traj.get("decision_stage_latencies_ms", []))
                    traj["fixed_stage_latencies_ms"] = list(rec.fixed_stage_latencies_ms or traj.get("fixed_stage_latencies_ms", []))
                    metadata = {**rec.metadata}
                    if reward_stale:
                        result = reward_evaluator.evaluate(
                            exec_res=rec.exec_res,
                            total_cost=rec.total_cost,
                            decision_stage_costs=list(rec.decision_stage_costs),
                        )
                        for key in REWARD_METADATA_KEYS:
                            metadata.pop(key, None)
                        metadata.update(reward_evaluator.metadata(result))
                        traj["reward"] = result.reward
                        traj["log_reward"] = result.log_reward
                    else:
                        traj["reward"] = rec.reward
                        if "log_reward" in metadata:
                            traj["log_reward"] = float(metadata["log_reward"])
                    traj["metadata"] = metadata
                    trajectories.append(traj)
            except Exception as e:
                skipped += 1
                logger.warning(
                    "Replay failed for qid=%d: %s", rec.question_id, e
                )
                continue

        self.last_replay_cache_miss_skipped_count = skipped
        logger.debug(
            "replay_as_trajectories: %d/%d trajectories replayed (zero LLM)",
            len(trajectories), len(records),
        )
        return trajectories

    # ---- Persistence ----

    def save(self, file_path: str):
        """保存为 JSON 文件。"""
        with self._lock:
            data = copy.deepcopy([r.to_dict() for r in self.records])
        _atomic_json_dump(data, file_path)
        logger.info("TrajectoryBuffer saved: %d records → %s", len(data), file_path)

    def load(self, file_path: str):
        """从 JSON 文件加载。"""
        if not os.path.exists(file_path):
            logger.warning("TrajectoryBuffer load: file not found: %s", file_path)
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            self.records = [TrajectoryRecord.from_dict(d) for d in data]
            self._reward_dirty = True
            n_records = len(self.records)
        logger.info("TrajectoryBuffer loaded: %d records ← %s", n_records, file_path)

    # ---- Admin ----

    def __len__(self) -> int:
        with self._lock:
            return len(self.records)

    def stats(self) -> Dict[str, Any]:
        """Buffer 统计。"""
        with self._lock:
            if not self.records:
                return {"total": 0}
            rewards = [r.reward for r in self.records]
            ex1 = sum(1 for r in self.records if r.exec_res == 1)
            costs = [r.total_cost for r in self.records]
            n_records = len(self.records)
            return {
                "total": n_records,
                "capacity": self.capacity,
                "avg_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
                "ex1_ratio": ex1 / n_records,
                "avg_cost": sum(costs) / len(costs),
            }


def _record_route_key(record: TrajectoryRecord) -> str:
    metadata = record.metadata or {}
    return str(metadata.get("route_key") or "-".join(str(int(action)) for action in record.actions))
