"""Local GTE inference and persistent text-embedding cache for State V1."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_config import SemanticModelConfig, fingerprint_model_directory


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


class SemanticEmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), timeout=30.0, check_same_thread=False
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_embeddings (
                model_identity TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (model_identity, text_hash)
            )
            """
        )
        self._connection.commit()

    def get_many(
        self, model_identity: str, text_hashes: Sequence[str]
    ) -> dict[str, np.ndarray]:
        hashes = tuple(dict.fromkeys(str(value) for value in text_hashes))
        if not hashes:
            return {}
        result: dict[str, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(hashes), 900):
                batch = hashes[start : start + 900]
                placeholders = ",".join("?" for _ in batch)
                rows = self._connection.execute(
                    f"""
                    SELECT text_hash, dimension, dtype, vector
                    FROM semantic_embeddings
                    WHERE model_identity = ? AND text_hash IN ({placeholders})
                    """,
                    (str(model_identity), *batch),
                ).fetchall()
                for value_hash, dimension, dtype, blob in rows:
                    if dtype != "float32":
                        raise ValueError(
                            f"cached semantic dtype for {value_hash} is {dtype!r}"
                        )
                    vector = np.frombuffer(blob, dtype="<f4").copy()
                    if vector.shape != (int(dimension),):
                        raise ValueError(
                            f"cached semantic dimension mismatch for {value_hash}"
                        )
                    if not np.isfinite(vector).all():
                        raise ValueError(
                            f"cached semantic vector for {value_hash} is non-finite"
                        )
                    result[str(value_hash)] = vector.astype(np.float32, copy=False)
        return result

    def put_many(
        self,
        model_identity: str,
        embeddings_by_text_hash: Mapping[str, np.ndarray],
    ) -> None:
        rows = []
        for value_hash, raw_vector in embeddings_by_text_hash.items():
            vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
            if not vector.size or not np.isfinite(vector).all():
                raise ValueError(f"invalid semantic embedding for {value_hash}")
            canonical = np.asarray(vector, dtype="<f4")
            rows.append(
                (
                    str(model_identity),
                    str(value_hash),
                    int(canonical.size),
                    "float32",
                    canonical.tobytes(order="C"),
                )
            )
        if not rows:
            return
        with self._lock:
            self._connection.executemany(
                """
                INSERT INTO semantic_embeddings
                    (model_identity, text_hash, dimension, dtype, vector)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_identity, text_hash) DO UPDATE SET
                    dimension=excluded.dimension,
                    dtype=excluded.dtype,
                    vector=excluded.vector
                """,
                rows,
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class LocalGTESemanticEncoder:
    def __init__(
        self,
        config: SemanticModelConfig,
        *,
        model: Any = None,
        tokenizer: Any = None,
        cache: SemanticEmbeddingCache | None = None,
        model_fingerprint: str | None = None,
    ) -> None:
        self.config = config
        self.model_path = Path(config.model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"local semantic model directory not found: {self.model_path}"
            )
        self.model_fingerprint = model_fingerprint or fingerprint_model_directory(
            self.model_path
        )
        self.model_identity = config.identity(
            model_fingerprint=self.model_fingerprint
        )
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self._device: Any = None
        self._inference_lock = threading.RLock()

    def _load_real_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        load_kwargs = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), **load_kwargs
        )
        self.model = AutoModel.from_pretrained(str(self.model_path), **load_kwargs)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self.model.eval()

    def _encode_uncached(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.config.dimension), dtype=np.float32)
        if self.model is not None and hasattr(self.model, "encode_texts"):
            values = self.model.encode_texts(texts)
        else:
            self._load_real_model()
            import torch

            tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(self._device) for name, value in tokens.items()}
            with torch.inference_mode():
                outputs = self.model(**tokens)
                vectors = outputs.last_hidden_state[:, 0, :]
                values = vectors.cpu().to(torch.float32).numpy()
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(texts), self.config.dimension):
            raise ValueError(
                f"semantic model returned {values.shape}, expected "
                f"{(len(texts), self.config.dimension)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("semantic model returned non-finite values")
        if self.config.normalize:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if np.any(norms <= 0.0):
                raise ValueError("semantic model returned a zero vector")
            values = values / norms
        return values.astype(np.float32, copy=False)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        ordered_texts = [str(value) for value in texts]
        if not ordered_texts:
            return np.zeros((0, self.config.dimension), dtype=np.float32)
        hashes = [text_hash(value) for value in ordered_texts]
        cached = (
            self.cache.get_many(self.model_identity, hashes)
            if self.cache is not None
            else {}
        )
        missing_text_by_hash: dict[str, str] = {}
        for value_hash, value_text in zip(hashes, ordered_texts):
            if value_hash not in cached:
                missing_text_by_hash.setdefault(value_hash, value_text)
        if missing_text_by_hash:
            missing_items = list(missing_text_by_hash.items())
            generated: dict[str, np.ndarray] = {}
            with self._inference_lock:
                for start in range(0, len(missing_items), self.config.batch_size):
                    batch = missing_items[start : start + self.config.batch_size]
                    vectors = self._encode_uncached([text for _hash, text in batch])
                    for (value_hash, _text), vector in zip(batch, vectors):
                        generated[value_hash] = vector
            if self.cache is not None:
                self.cache.put_many(self.model_identity, generated)
            cached.update(generated)
        result = np.stack([cached[value_hash] for value_hash in hashes])
        if result.shape != (len(ordered_texts), self.config.dimension):
            raise RuntimeError("semantic cache assembly returned an invalid shape")
        return result.astype(np.float32, copy=False)


def load_stage_prompt_embeddings(
    stage_prompt_path: str | Path,
    encoder: LocalGTESemanticEncoder,
    cache: SemanticEmbeddingCache | None = None,
) -> np.ndarray:
    path = Path(stage_prompt_path)
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            label, separator, prompt = line.rstrip("\r\n").partition("\t")
            if not separator or not prompt.strip():
                raise ValueError(
                    f"invalid stage prompt TSV row {line_number} in {path}"
                )
            rows.append((label.strip(), prompt.strip()))
    expected = ["S1", "S2", "S3", "S4"]
    if [label for label, _prompt in rows] != expected:
        raise ValueError(f"stage prompt rows must be exactly {expected}")
    if cache is not None and encoder.cache is not None and cache is not encoder.cache:
        raise ValueError("stage prompt cache must match encoder cache")
    values = encoder.encode_texts([prompt for _label, prompt in rows])
    return np.asarray(values, dtype=np.float32)


__all__ = [
    "LocalGTESemanticEncoder",
    "SemanticEmbeddingCache",
    "load_stage_prompt_embeddings",
    "text_hash",
]
