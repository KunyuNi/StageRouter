"""
TierRegistry — 统一注册/调度各阶段 Tier 1/2/3 算子

注册由各版本的 __init__.py 负责, 不在 _shared/ 中自动注册。
用法:
  from _shared.tier_registry import TierRegistry
  TierRegistry.register("keyword_extraction", 1, my_func)
  operator = TierRegistry.get("keyword_extraction", tier=1)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional


class TierRegistry:
    _operators: Dict[str, Dict[int, Callable]] = {}

    @classmethod
    def register(cls, stage_name: str, tier: int, func: Callable):
        if stage_name not in cls._operators:
            cls._operators[stage_name] = {}
        cls._operators[stage_name][tier] = func

    @classmethod
    def get(cls, stage_name: str, tier: int) -> Optional[Callable]:
        return cls._operators.get(stage_name, {}).get(tier)

    @classmethod
    def has(cls, stage_name: str, tier: int) -> bool:
        return cls.get(stage_name, tier) is not None

    @classmethod
    def available_tiers(cls, stage_name: str) -> list:
        return sorted(cls._operators.get(stage_name, {}).keys())

    @classmethod
    def registered_stages(cls) -> list:
        return sorted(cls._operators.keys())

    @classmethod
    def clear(cls):
        cls._operators.clear()
