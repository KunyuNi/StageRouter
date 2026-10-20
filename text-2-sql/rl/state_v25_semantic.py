"""Long-text embeddings and strict SQLite cache for State V25."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rl.state_v3_semantic import model_directory_fingerprint
from rl.state_v25_schema import SEMANTIC_DIM


CANONICALIZER_VERSION = "state_v25_lines_nfc_ws_v1"
CHUNK_POLICY = "line_boundary_then_token_split_weighted_cls_v1"
SEMANTIC_ROLES = frozenset(
    {"task", "database", "retrieval", "current_schema", "candidate_sql"}
)


def validate_semantic_cache_audit_manifest(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("semantic_cache_integrity_check") != "ok"
        or manifest.get("all_vectors_finite") is not True
    ):
        raise ValueError("State V25 semantic cache audit controls are not successful")


def canonicalize_text(text: Any) -> str:
    lines = []
    for raw_line in unicodedata.normalize("NFC", str(text or "")).splitlines():
        line = " ".join(raw_line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tokenizer_fingerprint(model_dir: Path) -> str:
    candidates = sorted(
        (
            path
            for path in model_dir.rglob("*")
            if path.is_file()
            and any(
                marker in path.name.casefold()
                for marker in ("tokenizer", "vocab", "merges", "sentencepiece")
            )
        ),
        key=lambda path: path.relative_to(model_dir).as_posix(),
    )
    if not candidates:
        raise ValueError(f"no tokenizer artifacts found in {model_dir}")
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.relative_to(model_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class EmbeddingAudit:
    total_non_special_tokens: int
    covered_non_special_tokens: int
    chunk_count: int
    silently_truncated: bool


class StateV25TextEncoder:
    """Frozen CLS encoder that covers every token through deterministic chunks."""

    dimension = SEMANTIC_DIM

    def __init__(
        self,
        model_or_directory: Any,
        tokenizer: Any | None = None,
        *,
        device: str = "cpu",
        max_length: int = 8192,
        batch_size: int = 8,
        trust_remote_code: bool = True,
        model_fingerprint: str | None = None,
        tokenizer_fingerprint: str | None = None,
    ) -> None:
        if isinstance(max_length, bool) or int(max_length) <= 2:
            raise ValueError("State V25 max_length must exceed special-token width")
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")

        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

        if isinstance(model_or_directory, (str, Path)):
            from transformers import AutoModel, AutoTokenizer

            model_dir = Path(model_or_directory).expanduser()
            if not model_dir.is_dir():
                raise FileNotFoundError(
                    f"local State V25 model directory not found: {model_dir}"
                )
            model_dir = model_dir.resolve()
            load_options = {
                "local_files_only": True,
                "trust_remote_code": bool(trust_remote_code),
            }
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), **load_options)
            model = AutoModel.from_pretrained(str(model_dir), **load_options)
            model_fingerprint = model_directory_fingerprint(model_dir)
            tokenizer_fingerprint = _tokenizer_fingerprint(model_dir)
        else:
            model = model_or_directory
            if tokenizer is None:
                raise TypeError("an injected State V25 model requires a tokenizer")
            if not model_fingerprint or not tokenizer_fingerprint:
                raise ValueError(
                    "injected model and tokenizer require explicit fingerprints"
                )

        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size != SEMANTIC_DIM:
            raise ValueError(
                f"State V25 requires hidden_size={SEMANTIC_DIM}, got {hidden_size!r}"
            )
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.model_fingerprint = str(model_fingerprint)
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        self._payload_width = self.max_length - special_count
        if self._payload_width <= 0:
            raise ValueError("max_length is too small for tokenizer special tokens")
        contract = {
            "version": 1,
            "dimension": SEMANTIC_DIM,
            "model_fingerprint": self.model_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "max_length": self.max_length,
            "pooling": "cls",
            "normalization": "l2",
            "canonicalizer": CANONICALIZER_VERSION,
            "chunk_policy": CHUNK_POLICY,
        }
        self.contract_sha256 = _sha256_bytes(
            json.dumps(
                contract, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )

    def _line_token_ids(self, text: str) -> list[list[int]]:
        canonical = canonicalize_text(text)
        if not canonical:
            return []
        return [
            list(self.tokenizer.encode(line, add_special_tokens=False))
            for line in canonical.splitlines()
        ]

    def token_count(self, text: str) -> int:
        return sum(len(ids) for ids in self._line_token_ids(text))

    def _chunks(self, text: str) -> list[list[int]]:
        chunks: list[list[int]] = []
        current: list[int] = []
        for line_ids in self._line_token_ids(text):
            remaining = list(line_ids)
            while remaining:
                available = self._payload_width - len(current)
                if available == 0:
                    chunks.append(current)
                    current = []
                    available = self._payload_width
                if len(remaining) <= available:
                    current.extend(remaining)
                    remaining = []
                elif current:
                    chunks.append(current)
                    current = []
                else:
                    chunks.append(remaining[: self._payload_width])
                    remaining = remaining[self._payload_width :]
        if current:
            chunks.append(current)
        return chunks

    def _batch_inputs(self, chunks: list[list[int]]):
        rows = [
            list(self.tokenizer.build_inputs_with_special_tokens(chunk))
            for chunk in chunks
        ]
        if any(len(row) > self.max_length for row in rows):
            raise AssertionError("State V25 chunk exceeds max_length")
        width = max(len(row) for row in rows)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        input_ids = []
        attention_mask = []
        for row in rows:
            padding = width - len(row)
            input_ids.append(row + [int(pad_token_id)] * padding)
            attention_mask.append([1] * len(row) + [0] * padding)
        return {
            "input_ids": self._torch.tensor(
                input_ids, dtype=self._torch.long, device=self.device
            ),
            "attention_mask": self._torch.tensor(
                attention_mask, dtype=self._torch.long, device=self.device
            ),
        }

    def encode(self, role: str, text: str) -> tuple[np.ndarray, EmbeddingAudit]:
        if role not in SEMANTIC_ROLES:
            raise ValueError(f"unsupported State V25 semantic role: {role}")
        chunks = self._chunks(text)
        total_tokens = sum(len(chunk) for chunk in chunks)
        if not chunks:
            return (
                np.zeros(SEMANTIC_DIM, dtype=np.float32),
                EmbeddingAudit(0, 0, 0, False),
            )

        vectors = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            inputs = self._batch_inputs(batch)
            with self._torch.inference_mode():
                output = self.model(**inputs)
                hidden = output.last_hidden_state
                if hidden.ndim != 3 or hidden.shape[-1] != SEMANTIC_DIM:
                    raise ValueError(
                        "State V25 model output must have shape (batch, tokens, 1024)"
                    )
                pooled = hidden[:, 0, :].to(dtype=self._torch.float32)
                norms = self._torch.linalg.vector_norm(
                    pooled, dim=1, keepdim=True
                )
                if not self._torch.isfinite(pooled).all() or not self._torch.all(
                    norms > 0
                ):
                    raise ValueError(
                        "State V25 model produced non-finite or zero CLS vectors"
                    )
                pooled = pooled / norms
            vectors.append(pooled.cpu().numpy())

        matrix = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
        weights = np.asarray([len(chunk) for chunk in chunks], dtype=np.float64)
        combined = np.average(matrix.astype(np.float64), axis=0, weights=weights)
        norm = float(np.linalg.norm(combined))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("State V25 chunk combination produced an invalid vector")
        vector = np.ascontiguousarray(combined / norm, dtype=np.float32)
        return vector, EmbeddingAudit(
            total_non_special_tokens=total_tokens,
            covered_non_special_tokens=total_tokens,
            chunk_count=len(chunks),
            silently_truncated=False,
        )


class StateV25SemanticCache:
    """Exact content-addressed cache for State V25 semantic vectors."""

    def __init__(
        self,
        path: str | Path,
        encoder: Any | None,
        *,
        writable: bool,
        encoder_contract_sha256: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.encoder = encoder
        self.writable = bool(writable)
        self.miss_count = 0
        self.write_attempts = 0
        contract = encoder_contract_sha256 or getattr(
            encoder, "contract_sha256", None
        )
        if not isinstance(contract, str) or len(contract) != 64:
            raise ValueError("State V25 semantic cache requires a contract SHA-256")
        self.encoder_contract_sha256 = contract
        if self.writable:
            if encoder is None or not callable(getattr(encoder, "encode", None)):
                raise TypeError("writable semantic cache requires an encoder")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path)
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    encoder_contract_sha256 TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (encoder_contract_sha256, role, text_sha256)
                )
                """
            )
            self._connection.commit()
        else:
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"State V25 semantic cache not found: {self.path}"
                )
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True)

    @staticmethod
    def _validate_vector_blob(blob: bytes) -> np.ndarray:
        expected = SEMANTIC_DIM * np.dtype("<f4").itemsize
        if len(blob) != expected:
            raise ValueError(
                f"State V25 semantic vector byte length must be {expected}"
            )
        vector = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
        if vector.shape != (SEMANTIC_DIM,) or not np.isfinite(vector).all():
            raise ValueError("State V25 semantic cache contains an invalid vector")
        return vector

    def vector(self, role: str, text: str) -> np.ndarray:
        if role not in SEMANTIC_ROLES:
            raise ValueError(f"unsupported State V25 semantic role: {role}")
        canonical = canonicalize_text(text)
        if not canonical:
            return np.zeros(SEMANTIC_DIM, dtype=np.float32)
        text_sha256 = _sha256_bytes(canonical.encode("utf-8"))
        key = (self.encoder_contract_sha256, role, text_sha256)
        row = self._connection.execute(
            """
            SELECT vector FROM embeddings
            WHERE encoder_contract_sha256 = ? AND role = ? AND text_sha256 = ?
            """,
            key,
        ).fetchone()
        if row is not None:
            return self._validate_vector_blob(row[0])
        self.miss_count += 1
        if not self.writable:
            raise KeyError(
                f"State V25 semantic cache miss for role={role} text_sha256={text_sha256}"
            )
        self.write_attempts += 1
        vector, audit = self.encoder.encode(role, canonical)
        vector = np.ascontiguousarray(vector, dtype="<f4")
        if (
            vector.shape != (SEMANTIC_DIM,)
            or not np.isfinite(vector).all()
            or audit.silently_truncated
            or audit.covered_non_special_tokens != audit.total_non_special_tokens
        ):
            raise ValueError("State V25 encoder returned an invalid vector or audit")
        blob = vector.tobytes(order="C")
        self._connection.execute(
            """
            INSERT OR IGNORE INTO embeddings
            (encoder_contract_sha256, role, text_sha256, vector)
            VALUES (?, ?, ?, ?)
            """,
            (*key, blob),
        )
        self._connection.commit()
        stored = self._connection.execute(
            """
            SELECT vector FROM embeddings
            WHERE encoder_contract_sha256 = ? AND role = ? AND text_sha256 = ?
            """,
            key,
        ).fetchone()
        if stored is None or stored[0] != blob:
            raise ValueError("State V25 semantic cache vector conflict")
        return self._validate_vector_blob(stored[0])

    def row_count(self) -> int:
        return int(
            self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        )

    def audit(self) -> dict[str, Any]:
        if self.writable:
            raise ValueError("State V25 semantic cache audit requires read-only mode")
        integrity_rows = [
            str(row[0]) for row in self._connection.execute("PRAGMA integrity_check")
        ]
        if integrity_rows != ["ok"]:
            raise ValueError(
                "State V25 semantic cache integrity_check is not ok: "
                + "; ".join(integrity_rows)
            )
        count = 0
        for (blob,) in self._connection.execute("SELECT vector FROM embeddings"):
            self._validate_vector_blob(blob)
            count += 1
        return {
            "semantic_cache_integrity_check": "ok",
            "all_vectors_finite": True,
            "cache_row_count": count,
        }

    def close(self) -> None:
        self._connection.close()
