"""Registration and lookup for tiered RAG LLM operators."""

from __future__ import annotations

from typing import Callable, Dict, Optional


class TierRegistry:
    _operators: Dict[str, Dict[int, Callable]] = {}

    @classmethod
    def register(cls, stage_name: str, tier: int, func: Callable) -> None:
        cls._operators.setdefault(stage_name, {})[int(tier)] = func

    @classmethod
    def get(cls, stage_name: str, tier: int) -> Optional[Callable]:
        return cls._operators.get(stage_name, {}).get(int(tier))

    @classmethod
    def require(cls, stage_name: str, tier: int) -> Callable:
        operator = cls.get(stage_name, tier)
        if operator is None:
            raise LookupError(
                f"No RAG operator registered for stage={stage_name!r}, tier={tier!r}"
            )
        return operator

    @classmethod
    def has(cls, stage_name: str, tier: int) -> bool:
        return cls.get(stage_name, tier) is not None

    @classmethod
    def available_tiers(cls, stage_name: str) -> list[int]:
        return sorted(cls._operators.get(stage_name, {}))

    @classmethod
    def registered_stages(cls) -> list[str]:
        return sorted(cls._operators)

    @classmethod
    def clear(cls) -> None:
        cls._operators.clear()

