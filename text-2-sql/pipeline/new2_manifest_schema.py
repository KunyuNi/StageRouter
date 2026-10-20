"""Shared versioned manifest extension used by New2 collection."""

from __future__ import annotations

from typing import Any, Mapping

from rl.exact_cache_contract import canonical_sha


FROZEN_EXACT_CACHE_KEY = "frozen_exact_cache"
FROZEN_EXACT_CACHE_VERSION = "new2_frozen_exact_v1"


def frozen_exact_cache_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": FROZEN_EXACT_CACHE_VERSION,
        "contract": dict(contract),
        "contract_sha256": canonical_sha(contract),
    }


def validate_frozen_exact_cache(
    manifest: Mapping[str, Any],
    *,
    expected_sha256: str,
    expected_contract: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    frozen = manifest.get(FROZEN_EXACT_CACHE_KEY)
    if frozen is None:
        return None
    if (
        not isinstance(frozen, Mapping)
        or set(frozen) != {"version", "contract", "contract_sha256"}
        or frozen.get("version") != FROZEN_EXACT_CACHE_VERSION
        or not isinstance(frozen.get("contract"), Mapping)
    ):
        raise ValueError("frozen exact cache schema mismatch")
    contract = frozen["contract"]
    digest = canonical_sha(contract)
    if frozen.get("contract_sha256") != digest or digest != expected_sha256:
        raise ValueError("frozen exact cache identity mismatch")
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("frozen exact cache contract mismatch")
    return contract
