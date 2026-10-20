"""Deterministic execution identity for RAG caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def _canonical(value: Any) -> Any:
    """Convert mappings and tier keys to a stable JSON-compatible shape."""

    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


@dataclass(frozen=True)
class WorkflowIdentity:
    """Values that can change a physical RAG rollout's outputs.

    F1 and pricing versions are deliberately absent: they describe how cached
    outputs are evaluated, not which outputs the operators produce.
    """

    train_dataset_sha256: str = ""
    corpus_sha256: str = ""
    index_identity: str = ""
    operator_package: str = "RAG.env.0_old"
    operator_version: str | int = 1
    prompt_version: str = "1"
    tier_models: Mapping[str, str] | Mapping[int, str] = None  # type: ignore[assignment]
    embedding_model: str = "text-embedding-v3"

    def __post_init__(self) -> None:
        if self.tier_models is None:
            object.__setattr__(self, "tier_models", {})
        else:
            object.__setattr__(self, "tier_models", dict(self.tier_models))

    @classmethod
    def from_inputs(
        cls,
        train_dataset_sha256: str | None = None,
        corpus_sha256: str = "",
        index_identity: str = "",
        operator_package: str | None = None,
        operator_version: str | int = 1,
        prompt_version: str = "1",
        tier_models: Mapping[str, str] | Mapping[int, str] | None = None,
        embedding_model: str = "text-embedding-v3",
        **aliases: Any,
    ) -> "WorkflowIdentity":
        """Build an identity from manifest/config inputs.

        A few descriptive aliases are accepted so callers can pass the names
        used by existing manifests (``dataset_sha256`` and ``operator_pkg``)
        without creating a second identity format.
        """

        if train_dataset_sha256 is None:
            train_dataset_sha256 = aliases.pop("dataset_sha256", "")
        if operator_package is None:
            operator_package = aliases.pop("operator_pkg", "RAG.env.0_old")
        if tier_models is None:
            tier_models = aliases.pop("model_mapping", aliases.pop("tier_mapping", {}))
        if aliases:
            unknown = ", ".join(sorted(aliases))
            raise TypeError(f"unknown workflow identity inputs: {unknown}")
        return cls(
            train_dataset_sha256=str(train_dataset_sha256 or ""),
            corpus_sha256=str(corpus_sha256 or ""),
            index_identity=str(index_identity or ""),
            operator_package=str(operator_package or ""),
            operator_version=operator_version,
            prompt_version=str(prompt_version),
            tier_models=tier_models or {},
            embedding_model=str(embedding_model),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "train_dataset_sha256": self.train_dataset_sha256,
            "corpus_sha256": self.corpus_sha256,
            "index_identity": self.index_identity,
            "operator_package": self.operator_package,
            "operator_version": self.operator_version,
            "prompt_version": self.prompt_version,
            "tier_models": _canonical(self.tier_models),
            "embedding_model": self.embedding_model,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            _canonical(self.to_json_dict()),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def dataset_sha256(self) -> str:
        """Alias used by manifests that omit the ``train_`` qualifier."""

        return self.train_dataset_sha256

    def __str__(self) -> str:
        return self.sha256
