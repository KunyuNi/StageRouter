"""Action policy for RAG decision stages.

Action ids follow the NL2SQL router contract:
0=T1, 1=T2, 2=T3, 3=SKIP.  SKIP is a routing action, not an LLM tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

T1_ACTION = 0
T2_ACTION = 1
T3_ACTION = 2
SKIP_ACTION = 3


@dataclass(frozen=True)
class StageActionSpec:
    allowed_actions: Tuple[int, ...]
    fixed_followups_by_action: Mapping[int, Tuple[str, ...]]


class StageActionRegistry:
    _stages: Dict[str, StageActionSpec] = {}

    @classmethod
    def register(
        cls,
        stage_name: str,
        allowed_actions: Iterable[int],
        fixed_followups_by_action: Mapping[int, Iterable[str]] | None = None,
    ) -> None:
        actions = tuple(sorted({int(action) for action in allowed_actions}))
        followups = {
            int(action): tuple(names)
            for action, names in (fixed_followups_by_action or {}).items()
        }
        unknown = set(followups) - set(actions)
        if unknown:
            raise ValueError(
                f"Follow-ups configured for disallowed actions at {stage_name!r}: "
                f"{sorted(unknown)}"
            )
        cls._stages[stage_name] = StageActionSpec(actions, followups)

    @classmethod
    def available_actions(cls, stage_name: str) -> list[int]:
        try:
            return list(cls._stages[stage_name].allowed_actions)
        except KeyError as exc:
            raise KeyError(f"No RAG decision stage registered as {stage_name!r}") from exc

    @classmethod
    def fixed_followups(cls, stage_name: str, action: int) -> list[str]:
        spec = cls._stages.get(stage_name)
        if spec is None:
            raise KeyError(f"No RAG decision stage registered as {stage_name!r}")
        action = int(action)
        if action not in spec.allowed_actions:
            raise ValueError(f"RAG stage {stage_name!r} does not allow action {action}")
        return list(spec.fixed_followups_by_action.get(action, ()))

    @classmethod
    def registered_stages(cls) -> list[str]:
        return list(cls._stages)

    @classmethod
    def clear(cls) -> None:
        cls._stages.clear()

