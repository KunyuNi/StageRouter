"""Frozen local-GTE configuration and identity helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SemanticModelConfig:
    model_path: str
    pooling: str = "cls"
    normalize: bool = True
    dimension: int = 1024
    max_length: int = 512
    batch_size: int = 32
    trust_remote_code: bool = True
    local_files_only: bool = True
    output_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.pooling != "cls":
            raise ValueError("State V1 semantic pooling must be 'cls'")
        if int(self.dimension) != 1024:
            raise ValueError("State V1 semantic dimension must be 1024")
        if int(self.max_length) <= 0 or int(self.batch_size) <= 0:
            raise ValueError("max_length and batch_size must be positive")
        if self.output_dtype != "float32":
            raise ValueError("State V1 semantic output_dtype must be float32")
        if not self.local_files_only:
            raise ValueError("State V1 semantic loading must be local-files-only")

    def identity(self, *, model_fingerprint: str) -> str:
        if not model_fingerprint:
            raise ValueError("model_fingerprint is required")
        contract = asdict(self)
        contract["model_path"] = str(Path(self.model_path).resolve())
        contract["model_fingerprint"] = str(model_fingerprint)
        canonical = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_model_directory(model_path: str | Path) -> str:
    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(f"local semantic model directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"local semantic model directory is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


__all__ = ["SemanticModelConfig", "fingerprint_model_directory"]
