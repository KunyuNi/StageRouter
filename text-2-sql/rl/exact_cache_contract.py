"""Typed, credential-free validation for the New2 exact-cache contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .transition_cache import _validate_exact_identity, normalize_endpoint_identity


CONTRACT_KEYS = {
    "contract_version", "operator_package", "tiers", "stages", "databases"
}
APPROVED_CONTRACT_VERSION = "new2-exact-v1"
ARTIFACT_KEYS = {"kind", "path", "sha256"}
RESOLVED_DIRECTORY_KEYS = {"kind", "path"}
SOURCE_KEYS = {"kind", "module", "path", "sha256", "callable_sha256"}
NOT_APPLICABLE_KEYS = {"kind", "reason"}
MODEL_KEYS = {"kind", "model", "temperature", "enable_thinking"}
EMBEDDING_KEYS = {"kind", "model", "endpoint"}
ENDPOINT_KEYS = {"kind", "url"}
DATABASE_KEYS = {"sqlite", "lsh", "minhash", "chroma", "description_csv"}
STAGE_KEYS = {"stage_source", "prompt", "parser"}
TIER_KEYS = {"default", "embedding"}
SENSITIVE_KEY_PARTS = (
    "authorization", "token", "password", "secret", "bearer", "apikey"
)
STAGE_NAMES = (
    "keyword_extraction",
    "entity_retrieval",
    "context_retrieval",
    "column_filtering",
    "table_selection",
    "column_selection",
    "candidate_generation",
    "revision",
)
PROMPT_STAGES = {0, 3, 4, 5, 6, 7}


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_secrets(value: Any, path: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = "".join(
                char for char in str(key).casefold() if char.isalnum()
            )
            if any(part in normalized for part in SENSITIVE_KEY_PARTS):
                raise ValueError(
                    f"secret-bearing field is forbidden at {path}.{key}"
                )
            _reject_secrets(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secrets(nested, f"{path}[{index}]")
    elif isinstance(value, str) and "bearer " in value.casefold():
        raise ValueError(f"raw bearer secret is forbidden at {path}")


def _strict_mapping(
    value: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"exact contract {label} schema mismatch")
    return value


def _artifact(
    value: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    row = _strict_mapping(value, ARTIFACT_KEYS, label)
    if row.get("kind") != "sha256":
        raise ValueError(f"exact contract {label} kind mismatch")
    raw_path = row.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"exact contract {label} path must be absolute")
    resolved = Path(raw_path).resolve()
    if os.path.normcase(str(resolved)) != os.path.normcase(raw_path):
        raise ValueError(f"exact contract {label} path must be resolved")
    if expected_path is not None and os.path.normcase(
        str(resolved)
    ) != os.path.normcase(str(expected_path.resolve())):
        raise ValueError(f"exact contract {label} resolved source path mismatch")
    if not resolved.exists():
        raise ValueError(f"exact contract {label} artifact path is missing")
    return dict(row)


def _resolved_directory(
    value: Any,
    label: str,
    *,
    expected_path: Path,
) -> dict[str, Any]:
    row = _strict_mapping(value, RESOLVED_DIRECTORY_KEYS, label)
    if row.get("kind") != "resolved_directory":
        raise ValueError(f"exact contract {label} kind mismatch")
    raw_path = row.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"exact contract {label} path must be absolute")
    resolved = Path(raw_path).resolve()
    if os.path.normcase(str(resolved)) != os.path.normcase(raw_path):
        raise ValueError(f"exact contract {label} path must be resolved")
    if os.path.normcase(str(resolved)) != os.path.normcase(
        str(expected_path.resolve())
    ):
        raise ValueError(f"exact contract {label} resolved source path mismatch")
    if not resolved.is_dir():
        raise ValueError(f"exact contract {label} directory is missing")
    return dict(row)


def _not_applicable(
    value: Any, label: str, *, expected_reason: str
) -> dict[str, Any]:
    row = _strict_mapping(value, NOT_APPLICABLE_KEYS, label)
    if row.get("kind") != "not_applicable" or row.get("reason") != expected_reason:
        raise ValueError(f"exact contract {label} not-applicable identity mismatch")
    return dict(row)


def _source(value: Any, label: str, *, expected_path: Path) -> dict[str, Any]:
    row = _strict_mapping(value, SOURCE_KEYS, label)
    if (
        row.get("kind") != "source"
        or not isinstance(row.get("module"), str)
        or not row["module"]
    ):
        raise ValueError(f"exact contract {label} source schema mismatch")
    artifact = _artifact(
        {"kind": "sha256", "path": row["path"], "sha256": row["sha256"]},
        label,
        expected_path=expected_path,
    )
    if row.get("callable_sha256") != artifact["sha256"]:
        raise ValueError(f"exact contract {label} callable source identity mismatch")
    return dict(row)


def _model(value: Any, label: str) -> dict[str, Any]:
    row = _strict_mapping(value, MODEL_KEYS, label)
    if (
        row.get("kind") != "model_config"
        or not isinstance(row.get("model"), str)
        or type(row.get("temperature")) not in (int, float)
        or not isinstance(row.get("enable_thinking"), bool)
    ):
        raise ValueError(f"exact contract {label} model schema mismatch")
    return dict(row)


def _embedding(value: Any, label: str) -> dict[str, Any]:
    row = _strict_mapping(value, EMBEDDING_KEYS, label)
    endpoint = _strict_mapping(
        row.get("endpoint"), ENDPOINT_KEYS, f"{label}.endpoint"
    )
    if (
        row.get("kind") != "embedding_config"
        or not isinstance(row.get("model"), str)
        or endpoint.get("kind") != "endpoint"
        or not isinstance(endpoint.get("url"), str)
    ):
        raise ValueError(f"exact contract {label} embedding schema mismatch")
    return dict(row)


def _indexed(
    value: Any, label: str, count: int
) -> dict[int, Mapping[str, Any]]:
    expected = {str(index) for index in range(count)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"exact contract {label} schema mismatch")
    return {index: value[str(index)] for index in range(count)}


def expand_exact_cache_contract(
    contract: Mapping[str, Any],
    question_map: Mapping[int, Mapping[str, Any]],
    *,
    operator_pkg: str,
    project_root: str | os.PathLike[str] | None = None,
    db_root: str | os.PathLike[str] | None = None,
    db_mode: str = "train",
    embedding_endpoint: str | None = None,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Validate real artifacts and expand compact exact-cache identities."""
    _reject_secrets(contract)
    root = _strict_mapping(contract, CONTRACT_KEYS, "root")
    if root.get("operator_package") != operator_pkg:
        raise ValueError("exact contract operator package mismatch")
    version = root.get("contract_version")
    if version != APPROVED_CONTRACT_VERSION:
        raise ValueError("exact contract version is not approved")
    resolved_project_root = Path(
        project_root or Path(__file__).resolve().parents[1]
    ).resolve()
    resolved_db_root = Path(
        db_root or resolved_project_root / "data" / "train"
    ).resolve()
    stages = _indexed(root.get("stages"), "stage", 8)
    tiers = _indexed(root.get("tiers"), "tier", 3)
    databases = root.get("databases")
    if not isinstance(databases, Mapping):
        raise ValueError("exact contract databases schema mismatch")
    db_ids = {str(row.get("db_id", "")) for row in question_map.values()}
    if "" in db_ids or not db_ids <= set(databases):
        raise ValueError(
            f"exact contract is missing db identity: {sorted(db_ids - set(databases))}"
        )

    validated_stages: dict[int, dict[str, Any]] = {}
    for stage_idx, raw in stages.items():
        row = _strict_mapping(raw, STAGE_KEYS, f"stage {stage_idx}")
        applicable = stage_idx in PROMPT_STAGES
        stage_name = STAGE_NAMES[stage_idx]
        source_path = (
            resolved_project_root / "env" / operator_pkg.rsplit(".", 1)[-1] / f"{stage_name}.py"
        )
        prompt_path = (
            resolved_project_root
            / "env"
            / "templates"
            / f"template_{stage_name}.txt"
        )
        parser_path = resolved_project_root / "env" / "llm" / "parsers.py"
        validated_stages[stage_idx] = {
            "stage_source": _source(
                row["stage_source"],
                f"source stage {stage_idx}",
                expected_path=source_path,
            ),
            "prompt": (
                _artifact(
                    row["prompt"],
                    f"prompt stage {stage_idx}",
                    expected_path=prompt_path,
                )
                if applicable
                else _not_applicable(
                    row["prompt"],
                    f"prompt stage {stage_idx}",
                    expected_reason="stage_has_no_prompt",
                )
            ),
            "parser": (
                _artifact(
                    row["parser"],
                    f"parser stage {stage_idx}",
                    expected_path=parser_path,
                )
                if applicable
                else _not_applicable(
                    row["parser"],
                    f"parser stage {stage_idx}",
                    expected_reason="stage_has_no_parser",
                )
            ),
        }
        if validated_stages[stage_idx]["stage_source"]["module"] != (
            f"{operator_pkg}.{stage_name}"
        ):
            raise ValueError(f"exact contract stage {stage_idx} module mismatch")

    from env._shared.model_config import EMBEDDING_MODEL, TIER_LLM_MODELS

    normalized_endpoint = (
        normalize_endpoint_identity(embedding_endpoint)
        if embedding_endpoint is not None
        else None
    )
    validated_tiers: dict[int, dict[str, Any]] = {}
    for tier, raw in tiers.items():
        row = _strict_mapping(raw, TIER_KEYS, f"tier {tier}")
        validated_tiers[tier] = {
            "default": _model(row["default"], f"tier {tier}.default"),
            "embedding": _embedding(
                row["embedding"], f"tier {tier}.embedding"
            ),
        }
        if validated_tiers[tier]["default"]["model"] != TIER_LLM_MODELS[tier + 1]:
            raise ValueError(f"exact contract tier {tier} model mismatch")
        default = validated_tiers[tier]["default"]
        if default["temperature"] != 0 or default["enable_thinking"] is not False:
            raise ValueError(f"exact contract tier {tier} model settings mismatch")
        embedding = validated_tiers[tier]["embedding"]
        if embedding["model"] != EMBEDDING_MODEL:
            raise ValueError(f"exact contract tier {tier} embedding model mismatch")
        declared_endpoint = normalize_endpoint_identity(
            embedding["endpoint"]["url"]
        )
        if declared_endpoint != embedding["endpoint"]["url"]:
            raise ValueError(
                f"exact contract tier {tier} embedding endpoint is not normalized"
            )
        if normalized_endpoint is not None and declared_endpoint != normalized_endpoint:
            raise ValueError(f"exact contract tier {tier} embedding endpoint mismatch")

    validated_databases: dict[str, dict[str, Any]] = {}
    for db_id, raw in databases.items():
        row = _strict_mapping(raw, DATABASE_KEYS, f"database {db_id}")
        database_root = resolved_db_root / f"{db_mode}_databases" / str(db_id)
        expected_paths = {
            "sqlite": database_root / f"{db_id}.sqlite",
            "lsh": database_root / "preprocessed" / f"{db_id}_lsh.pkl",
            "minhash": (
                database_root / "preprocessed" / f"{db_id}_minhashes.pkl"
            ),
            "chroma": database_root / "context_vector_db",
            "description_csv": database_root / "database_description",
        }
        validated_databases[str(db_id)] = {
            name: (
                _resolved_directory(
                    row[name],
                    f"{name} database {db_id}",
                    expected_path=expected_paths[name],
                )
                if name == "chroma"
                else _artifact(
                    row[name],
                    f"{name} database {db_id}",
                    expected_path=expected_paths[name],
                )
            )
            for name in sorted(DATABASE_KEYS)
        }

    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    for db_id in sorted(db_ids):
        database = validated_databases[db_id]
        index = {
            "kind": "sha256",
            "sha256": canonical_sha(database),
            "lsh": database["lsh"],
            "minhash": database["minhash"],
            "chroma": database["chroma"],
            "description_csv": database["description_csv"],
        }
        for stage_idx in range(8):
            stage = validated_stages[stage_idx]
            for tier in range(3):
                identity = {
                    "contract_version": version,
                    "operator_package": operator_pkg,
                    "stage_idx": stage_idx,
                    "tier": tier,
                    "effective_model_config": validated_tiers[tier][
                        "embedding" if stage_idx in {1, 2} else "default"
                    ],
                    **stage,
                    "database": (
                        database["sqlite"]
                        if stage_idx >= 3
                        else _not_applicable(
                            {
                                "kind": "not_applicable",
                                "reason": "stage_has_no_sqlite_dependency",
                            },
                            "database",
                            expected_reason="stage_has_no_sqlite_dependency",
                        )
                    ),
                    "index": (
                        index
                        if stage_idx in {1, 2, 3}
                        else _not_applicable(
                            {
                                "kind": "not_applicable",
                                "reason": "stage_has_no_index_dependency",
                            },
                            "index",
                            expected_reason="stage_has_no_index_dependency",
                        )
                    ),
                }
                _validate_exact_identity(identity, stage_idx, tier)
                result[(db_id, stage_idx, tier)] = identity
    if len(result) != len(db_ids) * 24:
        raise ValueError("exact contract expansion is incomplete")
    return result
