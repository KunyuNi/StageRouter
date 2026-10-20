"""Validate and expand the frozen exact-cache identity manifest."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pipeline.new2_manifest_schema import (
    FROZEN_EXACT_CACHE_KEY,
    FROZEN_EXACT_CACHE_VERSION,
)
from rl.exact_cache_contract import CONTRACT_KEYS, canonical_sha, expand_exact_cache_contract


def load_exact_cache_identity_manifest(
    path: str | os.PathLike[str],
    question_map: Mapping[int, Mapping[str, Any]],
    *,
    operator_pkg: str,
    project_root: str | os.PathLike[str] | None = None,
    db_root: str | os.PathLike[str] | None = None,
    db_mode: str = "train",
    embedding_endpoint: str | None = None,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    manifest_path = Path(path)
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load exact cache identity manifest: {manifest_path}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("exact cache identity manifest must contain a mapping")
    if set(loaded) == CONTRACT_KEYS:
        contract = loaded
    else:
        frozen = loaded.get(FROZEN_EXACT_CACHE_KEY)
        build_identity = loaded.get("build_identity")
        if (
            not isinstance(frozen, Mapping)
            or set(frozen) != {"version", "contract", "contract_sha256"}
            or frozen.get("version") != FROZEN_EXACT_CACHE_VERSION
            or not isinstance(frozen.get("contract"), Mapping)
            or not isinstance(build_identity, Mapping)
        ):
            raise ValueError("exact cache identity manifest strict schema mismatch")
        contract = frozen["contract"]
        if loaded.get("build_identity_sha256") != canonical_sha(build_identity):
            raise ValueError("exact cache identity manifest build identity mismatch")
        if frozen.get("contract_sha256") != canonical_sha(contract):
            raise ValueError("exact cache identity frozen contract digest mismatch")
        if build_identity.get("exact_cache_contract_sha256") != canonical_sha(contract):
            raise ValueError("exact cache identity manifest digest mismatch")
    return expand_exact_cache_contract(
        contract,
        question_map,
        operator_pkg=operator_pkg,
        project_root=project_root,
        db_root=db_root,
        db_mode=db_mode,
        embedding_endpoint=embedding_endpoint,
    )
