from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# 旧版线性 reward，只用于读取/重算历史 buffer；新采集和训练默认不再使用。
LOG_LINEAR_V1 = "log_linear_v1"

# 当前默认 reward：严格 EX 优先，同时在成功样本内部继续鼓励低 cost。
EX_FIRST_COST_CLIPPED_V1 = "ex_first_cost_clipped_v1"

# 新版 reward：使用 train buffer 拟合并冻结的六阶段成本尺度，只惩罚 Router 可控成本。
EX_FIRST_GLOBAL_STAGE_COST_V1 = "ex_first_global_stage_cost_v1"
GLOBAL_STAGE_COST_NORMALIZATION = "global_train_p99_weighted_v1"

# 默认版本必须是当前线上训练/采集使用的版本。
DEFAULT_REWARD_VERSION = EX_FIRST_COST_CLIPPED_V1

# beta：log_reward 的整体缩放系数。beta > 0 时不改变 route 排序，只改变 TB loss 的 reward 尺度。
DEFAULT_REWARD_BETA = 2.0

# alpha：旧版 log_linear_v1 的 cost 惩罚系数。只给历史兼容使用，不是当前推荐调参方向。
DEFAULT_LEGACY_ALPHA = 25.0

# success_bonus：EX 成功的基础奖励。当前设为 1.0，表示成功优先级的主项。
DEFAULT_SUCCESS_BONUS = 1.0

# cost_weight：cost_norm 的惩罚权重。必须小于 success_bonus，才能保证任意成功 route
# 的分数高于任意失败 route，避免“便宜失败 > 昂贵成功”的 reward inversion。
DEFAULT_COST_WEIGHT = 0.5

# cost_clip：cost 归一化上限，cost_norm = min(total_cost / cost_clip, 1.0)。
# 当前值来自 active 55,300 条 buffer 的 reward_cost_diagnostics.json 推荐配置：
# p99 cost clip，success/failure inversion_rate=0，rescued_order_ok_rate=1。
DEFAULT_COST_CLIP = 0.059603804

# REWARD_EPS：保留给需要 log(reward) 的兼容路径，避免 log(0)。
REWARD_EPS = 1e-300

# reward metadata 中所有 reward 参数字段。重算 reward 前会先清理这些旧字段，
# 再写入当前版本字段，避免新旧 reward 参数混在同一条 trajectory metadata 中。
REWARD_METADATA_KEYS = (
    "reward_version",
    "reward_alpha",
    "reward_beta",
    "reward_success_bonus",
    "reward_cost_weight",
    "reward_cost_clip",
    "reward_cost_normalization",
    "reward_normalizer_sha256",
    "controllable_cost_norm",
    "log_reward",
)


@dataclass(frozen=True)
class RewardConfig:
    version: str = DEFAULT_REWARD_VERSION
    beta: float = DEFAULT_REWARD_BETA
    alpha: float = DEFAULT_LEGACY_ALPHA
    success_bonus: float = DEFAULT_SUCCESS_BONUS
    cost_weight: float = DEFAULT_COST_WEIGHT
    cost_clip: float = DEFAULT_COST_CLIP

    def normalized(self) -> "RewardConfig":
        version = str(self.version or DEFAULT_REWARD_VERSION)
        if version not in {
            LOG_LINEAR_V1,
            EX_FIRST_COST_CLIPPED_V1,
            EX_FIRST_GLOBAL_STAGE_COST_V1,
        }:
            raise ValueError(f"Unknown reward version: {version}")
        beta = float(self.beta)
        alpha = float(self.alpha)
        success_bonus = float(self.success_bonus)
        cost_weight = float(self.cost_weight)
        cost_clip = float(self.cost_clip)
        if beta <= 0.0:
            raise ValueError("reward beta must be positive")
        if version == LOG_LINEAR_V1 and alpha < 0.0:
            raise ValueError("legacy reward alpha must be non-negative")
        if version in {EX_FIRST_COST_CLIPPED_V1, EX_FIRST_GLOBAL_STAGE_COST_V1}:
            if success_bonus <= 0.0:
                raise ValueError("success_bonus must be positive")
            if cost_weight < 0.0:
                raise ValueError("cost_weight must be non-negative")
            if cost_clip <= 0.0:
                raise ValueError("cost_clip must be positive")
            if success_bonus <= cost_weight:
                raise ValueError("success_bonus must be greater than cost_weight for EX-first ordering")
        return RewardConfig(
            version=version,
            beta=beta,
            alpha=alpha,
            success_bonus=success_bonus,
            cost_weight=cost_weight,
            cost_clip=cost_clip,
        )


def compute_log_reward(
    exec_res: int,
    total_cost: float,
    config: RewardConfig | None = None,
) -> float:
    cfg = (config or RewardConfig()).normalized()
    exec_value = 1.0 if int(exec_res) == 1 else 0.0
    cost = max(float(total_cost), 0.0)

    if cfg.version == LOG_LINEAR_V1:
        return cfg.beta * (exec_value - cfg.alpha * cost)

    if cfg.version == EX_FIRST_GLOBAL_STAGE_COST_V1:
        raise ValueError(
            "global stage reward requires RewardEvaluator and a frozen normalizer"
        )

    cost_norm = min(cost / cfg.cost_clip, 1.0)
    score = cfg.success_bonus * exec_value - cfg.cost_weight * cost_norm
    return cfg.beta * score


def compute_reward_from_log(log_reward: float) -> float:
    return float(math.exp(float(log_reward)))


def compute_reward(
    exec_res: int,
    total_cost: float,
    config: RewardConfig | None = None,
) -> float:
    return compute_reward_from_log(compute_log_reward(exec_res, total_cost, config=config))


def normalize_global_stage_cost(
    decision_stage_costs: Sequence[float],
    stage_clips: Sequence[float],
) -> float:
    if len(decision_stage_costs) != 6 or len(stage_clips) != 6:
        raise ValueError("decision stage costs and clips must contain 6 values")
    costs = tuple(float(value) for value in decision_stage_costs)
    clips = tuple(float(value) for value in stage_clips)
    if any(not math.isfinite(value) or value < 0.0 for value in costs):
        raise ValueError("decision stage costs must be finite and non-negative")
    if any(not math.isfinite(value) or value <= 0.0 for value in clips):
        raise ValueError("stage clips must be finite and positive")
    denominator = sum(clips)
    numerator = sum(min(cost, clip) for cost, clip in zip(costs, clips))
    return min(max(numerator / denominator, 0.0), 1.0)


@dataclass(frozen=True)
class RewardResult:
    log_reward: float
    reward: float
    controllable_cost_norm: float


class RewardEvaluator:
    def __init__(self, config: RewardConfig, normalizer=None):
        self.config = config.normalized()
        self.normalizer = normalizer

    def evaluate(
        self,
        exec_res: int,
        total_cost: float,
        decision_stage_costs: Sequence[float] | None = None,
    ) -> RewardResult:
        if self.config.version == EX_FIRST_GLOBAL_STAGE_COST_V1:
            if self.normalizer is None:
                raise ValueError("global stage reward requires a frozen normalizer")
            cost_norm = normalize_global_stage_cost(
                decision_stage_costs or (),
                self.normalizer.stage_clips,
            )
            exec_value = 1.0 if int(exec_res) == 1 else 0.0
            score = (
                self.config.success_bonus * exec_value
                - self.config.cost_weight * cost_norm
            )
            log_reward = self.config.beta * score
        else:
            log_reward = compute_log_reward(
                exec_res=exec_res,
                total_cost=total_cost,
                config=self.config,
            )
            cost = max(float(total_cost), 0.0)
            cost_norm = min(cost / self.config.cost_clip, 1.0)
        return RewardResult(
            log_reward=float(log_reward),
            reward=compute_reward_from_log(log_reward),
            controllable_cost_norm=float(cost_norm),
        )

    def metadata(self, result: RewardResult | None = None) -> dict[str, float | str]:
        return reward_metadata(
            self.config,
            normalizer=self.normalizer,
            result=result,
        )


def reward_metadata(
    config: RewardConfig | None = None,
    *,
    normalizer=None,
    result: RewardResult | None = None,
) -> dict[str, float | str]:
    cfg = (config or RewardConfig()).normalized()
    metadata: dict[str, float | str] = {
        "reward_version": cfg.version,
        "reward_beta": cfg.beta,
    }
    if cfg.version == LOG_LINEAR_V1:
        metadata["reward_alpha"] = cfg.alpha
    elif cfg.version == EX_FIRST_COST_CLIPPED_V1:
        metadata.update(
            {
                "reward_success_bonus": cfg.success_bonus,
                "reward_cost_weight": cfg.cost_weight,
                "reward_cost_clip": cfg.cost_clip,
            }
        )
    else:
        if normalizer is None:
            raise ValueError("global stage reward metadata requires a frozen normalizer")
        metadata.update(
            {
                "reward_success_bonus": cfg.success_bonus,
                "reward_cost_weight": cfg.cost_weight,
                "reward_cost_normalization": GLOBAL_STAGE_COST_NORMALIZATION,
                "reward_normalizer_sha256": str(normalizer.artifact_sha256),
            }
        )
        if result is not None:
            metadata["controllable_cost_norm"] = result.controllable_cost_norm
    if result is not None:
        metadata["log_reward"] = result.log_reward
    return metadata


def reward_config_from_args(args: Any) -> RewardConfig:
    return RewardConfig(
        version=getattr(args, "reward_version", DEFAULT_REWARD_VERSION),
        beta=getattr(args, "reward_beta", getattr(args, "beta", DEFAULT_REWARD_BETA)),
        alpha=getattr(args, "reward_alpha", getattr(args, "alpha", DEFAULT_LEGACY_ALPHA)),
        success_bonus=getattr(args, "reward_success_bonus", DEFAULT_SUCCESS_BONUS),
        cost_weight=getattr(args, "reward_cost_weight", DEFAULT_COST_WEIGHT),
        cost_clip=getattr(args, "reward_cost_clip", DEFAULT_COST_CLIP),
    ).normalized()


def reward_config_from_metadata(metadata: Mapping[str, Any] | None) -> RewardConfig:
    metadata = metadata or {}
    version = str(metadata.get("reward_version") or LOG_LINEAR_V1)
    return RewardConfig(
        version=version,
        beta=float(metadata.get("reward_beta", DEFAULT_REWARD_BETA)),
        alpha=float(metadata.get("reward_alpha", DEFAULT_LEGACY_ALPHA)),
        success_bonus=float(metadata.get("reward_success_bonus", DEFAULT_SUCCESS_BONUS)),
        cost_weight=float(metadata.get("reward_cost_weight", DEFAULT_COST_WEIGHT)),
        cost_clip=float(metadata.get("reward_cost_clip", DEFAULT_COST_CLIP)),
    ).normalized()
