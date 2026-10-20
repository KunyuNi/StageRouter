"""Construction and shared initialization for State V26 networks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch

from .state_v26_network import StateV26ScalarDuelingNetwork
from .state_v26_schema import CONTEXT_FIELD_NAMES, CONTEXT_SLICES, STATE_V26_DIM


def state_v26_schema_identity() -> dict[str, Any]:
    payload = {
        "state_dim": STATE_V26_DIM,
        "blocks": {
            block: {
                "fields": list(fields),
                "start": CONTEXT_SLICES[block].start,
                "stop": CONTEXT_SLICES[block].stop,
            }
            for block, fields in CONTEXT_FIELD_NAMES.items()
        },
    }
    canonical_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "canonical_sha256": canonical_sha256}


def build_state_v26_network(config: Mapping[str, Any]) -> StateV26ScalarDuelingNetwork:
    required = {
        "algorithm": "state_v26_scalar_double_dqn",
        "state_version": "state_v26",
        "state_dim": 5256,
        "hidden_dim": 512,
        "repr_dim": 128,
    }
    for field, expected in required.items():
        if config.get(field) != expected:
            raise ValueError(f"State V26 network requires {field}={expected!r}")
    return StateV26ScalarDuelingNetwork()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def create_state_v26_initialization_bundle(*, seed: int = 1145) -> dict[str, Any]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = StateV26ScalarDuelingNetwork()
    state_dict = {
        key: value.detach().cpu().clone() for key, value in network.state_dict().items()
    }
    return {
        "format_version": 1,
        "seed": seed,
        "state_schema_hash": state_v26_schema_identity(),
        "state_dict": state_dict,
        "tensor_sha256": {
            key: _tensor_sha256(value) for key, value in state_dict.items()
        },
    }


def load_state_v26_initialization(
    network: StateV26ScalarDuelingNetwork, bundle: Mapping[str, Any]
) -> None:
    if bundle.get("format_version") != 1:
        raise ValueError("State V26 initialization format mismatch")
    if bundle.get("state_schema_hash") != state_v26_schema_identity():
        raise ValueError("State V26 initialization schema mismatch")
    state_dict = bundle.get("state_dict")
    hashes = bundle.get("tensor_sha256")
    if not isinstance(state_dict, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("State V26 initialization bundle is incomplete")
    for key, value in state_dict.items():
        if hashes.get(key) != _tensor_sha256(value):
            raise ValueError(f"State V26 initialization tensor drift: {key}")
    network.load_state_dict(state_dict, strict=True)
