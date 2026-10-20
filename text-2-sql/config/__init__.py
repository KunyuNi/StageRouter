"""Configuration entry point for the locked StageRouter run."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .new2_state_v26 import (
    LOCKED_RUN_NAME,
    NEW2_STATE_V26_RUN_CONFIGS,
    validate_new2_state_v26_config,
)

DEFAULT_RUN_NAME = LOCKED_RUN_NAME
DEFAULT_MODEL_ROOT = "model"


def validate_run_config(config: Mapping[str, Any]) -> None:
    validate_new2_state_v26_config(config)


def get_run_config(run_name: str | None = None) -> dict[str, Any]:
    name = run_name or DEFAULT_RUN_NAME
    if name not in NEW2_STATE_V26_RUN_CONFIGS:
        raise ValueError(f"Unknown run_name: {name}")
    config = deepcopy(NEW2_STATE_V26_RUN_CONFIGS[name])
    validate_run_config(config)
    return config


def run_config_value(run_name: str, key: str, fallback: Any = None) -> Any:
    return get_run_config(run_name).get(key, fallback)
