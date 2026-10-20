"""Local text embeddings, schema cache, and State V3 schema alignment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import OrderedDict, namedtuple
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .semantic_config import DEFAULT_SEMANTIC_TRUST_REMOTE_CODE
from .state_v3_context import StaticSchemaContext


EMBEDDING_DIM = 1024
DEFAULT_TOP_K = 8
SOFTMAX_TEMPERATURE = 0.07

_CACHE_FORMAT_VERSION = 1
_MODEL_FILE_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".py",
    ".safetensors",
    ".tiktoken",
    ".txt",
    ".vocab",
}
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

AlignmentCacheKey = namedtuple(
    "AlignmentCacheKey",
    (
        "question_sha256",
        "db_id",
        "active_schema_sha256",
        "visible_description_sha256",
    ),
)
DescribedSchemaCacheKey = namedtuple(
    "DescribedSchemaCacheKey",
    ("source_fingerprint", "visible_description_sha256"),
)
QuestionEmbeddingCacheKey = namedtuple(
    "QuestionEmbeddingCacheKey",
    ("model_fingerprint", "question_sha256"),
)


class A3ReadOnlyQuestionEmbeddingProvider:
    """Immutable formal-question embeddings bound to model and data identity."""

    read_only = True

    def __init__(
        self,
        records: Sequence[tuple[int, str, np.ndarray]],
        *,
        model_fingerprint: str,
        question_data_sha256: str,
    ) -> None:
        model_identity = str(model_fingerprint).strip()
        data_identity = str(question_data_sha256).strip().lower()
        if not model_identity or not re.fullmatch(r"[0-9a-f]{64}", data_identity):
            raise ValueError("A3 question provider identity is invalid")
        if not records:
            raise ValueError("A3 question provider records must not be empty")

        embeddings: dict[tuple[str, str, str], np.ndarray] = {}
        qids: set[int] = set()
        content_digest = hashlib.sha256()
        for raw_qid, raw_question, raw_embedding in records:
            qid = int(raw_qid)
            if qid in qids:
                raise ValueError("A3 question provider qids must be unique")
            qids.add(qid)
            question = str(raw_question or "")
            vector = np.asarray(raw_embedding, dtype=np.float32)
            if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
                raise ValueError(
                    "A3 question provider embeddings must be finite and 1024D"
                )
            vector = np.ascontiguousarray(vector)
            vector.setflags(write=False)
            key = (model_identity, data_identity, _sha256_text(question))
            previous = embeddings.setdefault(key, vector)
            if not np.array_equal(previous, vector):
                raise ValueError(
                    "duplicate A3 question text produced inconsistent embeddings"
                )
            content_digest.update(str(qid).encode("ascii"))
            content_digest.update(b"\0")
            content_digest.update(key[2].encode("ascii"))
            content_digest.update(vector.tobytes(order="C"))

        self.model_fingerprint = model_identity
        self.question_data_sha256 = data_identity
        self.qid_count = len(qids)
        self.embedding_sha256 = content_digest.hexdigest()
        self._embeddings = MappingProxyType(embeddings)
        self.hit_count = 0
        self.miss_count = 0

    def get(
        self,
        question: str,
        *,
        model_fingerprint: str,
        question_data_sha256: str,
    ) -> np.ndarray:
        model_identity = str(model_fingerprint).strip()
        data_identity = str(question_data_sha256).strip().lower()
        if (
            model_identity != self.model_fingerprint
            or data_identity != self.question_data_sha256
        ):
            self.miss_count += 1
            raise RuntimeError("read-only A3 question provider identity mismatch")
        key = (model_identity, data_identity, _sha256_text(str(question or "")))
        embedding = self._embeddings.get(key)
        if embedding is None:
            self.miss_count += 1
            raise RuntimeError("read-only A3 question embedding provider miss")
        self.hit_count += 1
        return embedding


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(str(value).encode("utf-8"))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_files(model_dir: str | Path) -> tuple[Path, ...]:
    resolved = Path(model_dir).expanduser().resolve()
    return tuple(
        sorted(
            (
                candidate
                for candidate in resolved.rglob("*")
                if candidate.is_file()
                and (
                    candidate.suffix.casefold() in _MODEL_FILE_SUFFIXES
                    or candidate.name.casefold().startswith("tokenizer")
                )
            ),
            key=lambda candidate: candidate.relative_to(resolved).as_posix(),
        )
    )


def _file_set_snapshot(
    files: Sequence[Path],
    root: str | Path,
) -> tuple[tuple[str, int, int, int, int], ...]:
    resolved_root = Path(root).expanduser().resolve()
    snapshot = []
    for path in files:
        stat = path.stat()
        snapshot.append(
            (
                path.resolve().relative_to(resolved_root).as_posix(),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
                int(getattr(stat, "st_ino", 0)),
            )
        )
    return tuple(snapshot)


def model_directory_fingerprint(model_dir: str | Path) -> str:
    """Hash the resolved model path and all load-relevant local artifacts."""
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"local model directory not found: {path}")
    resolved = path.resolve()
    files = _model_artifact_files(resolved)
    if not files:
        raise ValueError(f"local model directory has no model artifacts: {resolved}")

    digest = hashlib.sha256()
    digest.update(str(resolved).encode("utf-8"))
    digest.update(b"\0")
    for file_path in files:
        relative = file_path.relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


class StateV3TextEncoder:
    """Thin wrapper around one explicitly local frozen Transformer model."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
        trust_remote_code: bool = DEFAULT_SEMANTIC_TRUST_REMOTE_CODE,
    ) -> None:
        path = Path(model_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"local model directory not found: {path}")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_dir = path.resolve()
        self.model_fingerprint = model_directory_fingerprint(self.model_dir)
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._torch = torch

        load_options = {
            "local_files_only": True,
            "trust_remote_code": bool(trust_remote_code),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), **load_options
        )
        model = AutoModel.from_pretrained(str(self.model_dir), **load_options)
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size != EMBEDDING_DIM:
            raise ValueError(
                "State V3 requires local model hidden_size=1024, "
                f"got {hidden_size!r}"
            )
        self.model = model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def _move_inputs(self, inputs: Any) -> Mapping[str, Any]:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode every text with the same CLS pooling and L2 normalization."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")
        normalized_texts = [str(text) for text in texts]
        if not normalized_texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(normalized_texts), self.batch_size):
            batch_texts = normalized_texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            moved = self._move_inputs(inputs)
            with self._torch.inference_mode():
                outputs = self.model(**moved)
                hidden = outputs.last_hidden_state
                if hidden.ndim != 3 or hidden.shape[-1] != EMBEDDING_DIM:
                    raise ValueError(
                        "model last_hidden_state must have shape (n, tokens, 1024)"
                    )
                pooled = hidden[:, 0, :].to(dtype=self._torch.float32)
                norms = self._torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
                if not self._torch.isfinite(pooled).all() or not self._torch.all(
                    norms > 0
                ):
                    raise ValueError("model produced non-finite or zero embeddings")
                pooled = pooled / norms
            batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))

        embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        _validate_embedding_matrix(
            embeddings,
            expected_rows=len(normalized_texts),
            name="encoded texts",
        )
        return embeddings


@dataclass(frozen=True)
class SchemaItemEmbeddings:
    """Real schema identities, their exact encoder text, and embeddings."""

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    tables: tuple[str, ...]
    columns: tuple[str, ...]
    texts: tuple[str, ...]
    embeddings: np.ndarray
    source_fingerprint: str = ""

    def __post_init__(self) -> None:
        fields = ("names", "kinds", "tables", "columns", "texts")
        for field_name in fields:
            values = tuple(str(value) for value in getattr(self, field_name))
            object.__setattr__(self, field_name, values)
        item_count = len(self.names)
        for field_name in fields[1:]:
            if len(getattr(self, field_name)) != item_count:
                raise ValueError(f"{field_name} length does not match names")
        if any(kind not in {"table", "column"} for kind in self.kinds):
            raise ValueError("schema item kinds must be table or column")

        embeddings = np.ascontiguousarray(self.embeddings, dtype=np.float32)
        _validate_embedding_matrix(
            embeddings,
            expected_rows=item_count,
            name="schema item embeddings",
        )
        embeddings.setflags(write=False)
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "source_fingerprint", str(self.source_fingerprint))

    @classmethod
    def empty(cls, source_fingerprint: str = "") -> "SchemaItemEmbeddings":
        return cls(
            names=(),
            kinds=(),
            tables=(),
            columns=(),
            texts=(),
            embeddings=np.empty((0, EMBEDDING_DIM), dtype=np.float32),
            source_fingerprint=source_fingerprint,
        )

    @property
    def table_mask(self) -> np.ndarray:
        return np.asarray([kind == "table" for kind in self.kinds], dtype=bool)

    @property
    def column_mask(self) -> np.ndarray:
        return np.asarray([kind == "column" for kind in self.kinds], dtype=bool)


@dataclass(frozen=True)
class SchemaAlignment:
    table_top1: float
    table_top3_mean: float
    table_margin: float
    table_entropy: float
    column_top1: float
    column_top3_mean: float
    column_margin: float
    column_entropy: float
    token_coverage: float
    active_table_ratio: float
    active_column_ratio: float
    active_alignment_delta_from_full: float
    schema_embedding: np.ndarray
    available: bool

    def __post_init__(self) -> None:
        scalar_names = (
            "table_top1",
            "table_top3_mean",
            "table_margin",
            "table_entropy",
            "column_top1",
            "column_top3_mean",
            "column_margin",
            "column_entropy",
            "token_coverage",
            "active_table_ratio",
            "active_column_ratio",
            "active_alignment_delta_from_full",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        embedding = np.asarray(self.schema_embedding, dtype=np.float32)
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"schema_embedding must have shape ({EMBEDDING_DIM},)"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("schema_embedding must be finite")
        embedding = np.ascontiguousarray(embedding)
        embedding.setflags(write=False)
        object.__setattr__(self, "schema_embedding", embedding)
        object.__setattr__(self, "available", bool(self.available))

    @property
    def table_top1_top2_margin(self) -> float:
        return self.table_margin

    @property
    def table_similarity_entropy(self) -> float:
        return self.table_entropy

    @property
    def column_top1_top2_margin(self) -> float:
        return self.column_margin

    @property
    def column_similarity_entropy(self) -> float:
        return self.column_entropy

    @property
    def schema_semantic_available(self) -> float:
        return float(self.available)

    def as_feature_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.table_top1,
                self.table_top3_mean,
                self.table_margin,
                self.table_entropy,
                self.column_top1,
                self.column_top3_mean,
                self.column_margin,
                self.column_entropy,
                self.token_coverage,
                self.active_table_ratio,
                self.active_column_ratio,
                self.active_alignment_delta_from_full,
                float(self.available),
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class _SchemaItemSpec:
    name: str
    kind: str
    table: str
    column: str
    text: str


class _LRU:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("LRU capacity must be positive")
        self.capacity = int(capacity)
        self.values: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Any:
        try:
            value = self.values.pop(key)
        except KeyError:
            return None
        self.values[key] = value
        return value

    def put(self, key: Any, value: Any) -> None:
        self.values.pop(key, None)
        self.values[key] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)

    def discard_where(self, predicate) -> None:
        for key in tuple(self.values):
            if predicate(key):
                del self.values[key]

    def clear(self) -> None:
        self.values.clear()


class SchemaSemanticCache:
    """Validated disk cache plus route-local process LRU caches."""

    def __init__(
        self,
        *,
        encoder: Any,
        db_mode: str,
        cache_root: str | Path = "cache/state_v3_schema_embeddings",
        lru_size: int = 128,
        alignment_lru_size: int | None = None,
    ) -> None:
        if not str(db_mode):
            raise ValueError("db_mode must not be empty")
        if not callable(getattr(encoder, "encode_texts", None)):
            raise TypeError("encoder must provide encode_texts(texts)")
        self.encoder = encoder
        self.db_mode = str(db_mode)
        self.cache_root = Path(cache_root)
        self._schema_memory: dict[str, SchemaItemEmbeddings] = {}
        self._source_by_db: dict[str, str] = {}
        alignment_capacity = (
            int(lru_size)
            if alignment_lru_size is None
            else int(alignment_lru_size)
        )
        if alignment_capacity <= 0:
            raise ValueError("alignment_lru_size must be positive")
        self._alignment_lru = _LRU(alignment_capacity)
        self._described_schema_lru = _LRU(lru_size)
        self._question_embedding_lru = _LRU(lru_size)
        self._candidate_sql_lru = _LRU(lru_size)
        self._question_model_fingerprint: str | None = None
        self._candidate_model_fingerprint: str | None = None
        self._a3_question_embedding_provider: (
            A3ReadOnlyQuestionEmbeddingProvider | None
        ) = None
        self._a3_question_data_sha256: str | None = None
        self._model_source_cache: tuple[
            Path,
            tuple[tuple[str, int, int, int, int], ...],
            tuple[str, str],
        ] | None = None
        self._file_hash_cache: dict[
            Path,
            tuple[tuple[int, int, int, int], str],
        ] = {}
        self._description_hash_cache: dict[
            Path,
            tuple[tuple[tuple[str, int, int, int, int], ...], str],
        ] = {}

    @property
    def alignment_cache_keys(self) -> tuple[AlignmentCacheKey, ...]:
        return tuple(self._alignment_lru.values)

    @property
    def candidate_sql_cache_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._candidate_sql_lru.values)

    @property
    def described_schema_cache_keys(self) -> tuple[DescribedSchemaCacheKey, ...]:
        return tuple(self._described_schema_lru.values)

    @property
    def question_embedding_cache_keys(
        self,
    ) -> tuple[QuestionEmbeddingCacheKey, ...]:
        return tuple(self._question_embedding_lru.values)

    @property
    def question_embedding_provider(
        self,
    ) -> A3ReadOnlyQuestionEmbeddingProvider | None:
        return self._a3_question_embedding_provider

    @property
    def question_embedding_provider_hit_count(self) -> int:
        provider = self._a3_question_embedding_provider
        return 0 if provider is None else provider.hit_count

    @property
    def question_embedding_provider_miss_count(self) -> int:
        provider = self._a3_question_embedding_provider
        return 0 if provider is None else provider.miss_count

    def install_a3_question_embedding_provider(
        self,
        provider: A3ReadOnlyQuestionEmbeddingProvider,
        *,
        model_fingerprint: str,
        question_data_sha256: str,
    ) -> None:
        if self._a3_question_embedding_provider is not None:
            raise ValueError("A3 question embedding provider is already installed")
        if not isinstance(provider, A3ReadOnlyQuestionEmbeddingProvider):
            raise TypeError(
                "provider must be an A3ReadOnlyQuestionEmbeddingProvider"
            )
        model_identity = str(model_fingerprint).strip()
        data_identity = str(question_data_sha256).strip().lower()
        _, loaded_model_fingerprint = self._model_source()
        if (
            provider.read_only is not True
            or provider.model_fingerprint != model_identity
            or provider.question_data_sha256 != data_identity
            or loaded_model_fingerprint != model_identity
        ):
            raise ValueError("A3 question embedding provider identity mismatch")
        self._question_embedding_lru.clear()
        self._question_model_fingerprint = None
        self._a3_question_embedding_provider = provider
        self._a3_question_data_sha256 = data_identity

    def database_fingerprint(self, static_context: StaticSchemaContext) -> str:
        if not static_context.db_path.is_file():
            raise FileNotFoundError(
                f"SQLite database not found: {static_context.db_path}"
            )
        path = static_context.db_path.resolve()
        stat = path.stat()
        snapshot = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            int(getattr(stat, "st_ino", 0)),
        )
        cached = self._file_hash_cache.get(path)
        if cached is not None and cached[0] == snapshot:
            return cached[1]
        fingerprint = _sha256_file(path)
        self._file_hash_cache[path] = (snapshot, fingerprint)
        return fingerprint

    def _model_source(self) -> tuple[str, str]:
        model_dir = getattr(self.encoder, "model_dir", None)
        if model_dir is not None:
            resolved = Path(model_dir).expanduser().resolve()
            files = _model_artifact_files(resolved)
            if not files:
                raise ValueError(
                    f"local model directory has no model artifacts: {resolved}"
                )
            snapshot = _file_set_snapshot(files, resolved)
            cached = self._model_source_cache
            if cached is not None and cached[0] == resolved and cached[1] == snapshot:
                return cached[2]
            source = (
                _sha256_text(str(resolved)),
                model_directory_fingerprint(resolved),
            )
            self._model_source_cache = (resolved, snapshot, source)
            return source
        fingerprint = str(getattr(self.encoder, "model_fingerprint", ""))
        if not fingerprint:
            raise ValueError(
                "encoder must expose model_dir or a verifiable model_fingerprint"
            )
        return _sha256_text(fingerprint), fingerprint

    def _description_fingerprint(
        self, static_context: StaticSchemaContext
    ) -> str:
        description_dir = static_context.db_path.parent / "database_description"
        csv_files = tuple(
            sorted(
                description_dir.rglob("*.csv"),
                key=lambda path: path.relative_to(description_dir).as_posix(),
            )
        ) if description_dir.is_dir() else ()
        resolved_dir = description_dir.resolve()
        snapshot = _file_set_snapshot(csv_files, resolved_dir)
        cached = self._description_hash_cache.get(resolved_dir)
        if cached is not None and cached[0] == snapshot:
            return cached[1]
        digest = hashlib.sha256()
        for csv_path in csv_files:
            digest.update(
                csv_path.relative_to(description_dir).as_posix().encode("utf-8")
            )
            digest.update(b"\0")
            digest.update(_sha256_file(csv_path).encode("ascii"))
            digest.update(b"\0")
        fingerprint = digest.hexdigest()
        self._description_hash_cache[resolved_dir] = (snapshot, fingerprint)
        return fingerprint

    def _source_metadata(
        self,
        static_context: StaticSchemaContext,
        specs: Sequence[_SchemaItemSpec],
    ) -> dict[str, Any]:
        model_path_sha256, model_files_sha256 = self._model_source()
        item_sha256 = _sha256_text(
            _canonical_json(
                [
                    [spec.name, spec.kind, spec.table, spec.column, spec.text]
                    for spec in specs
                ]
            )
        )
        metadata = {
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "model_path_sha256": model_path_sha256,
            "model_files_sha256": model_files_sha256,
            "sqlite_sha256": self.database_fingerprint(static_context),
            "description_csv_sha256": self._description_fingerprint(
                static_context
            ),
            "db_mode": self.db_mode,
            "db_id": str(static_context.db_id),
            "schema_items_sha256": item_sha256,
        }
        metadata["source_fingerprint"] = _sha256_text(_canonical_json(metadata))
        return metadata

    def _cache_path(
        self, static_context: StaticSchemaContext, model_fingerprint: str
    ) -> Path:
        return (
            self.cache_root
            / model_fingerprint
            / _safe_component(self.db_mode)
            / f"{_safe_component(static_context.db_id)}.npz"
        )

    def get_or_build(
        self, static_context: StaticSchemaContext
    ) -> SchemaItemEmbeddings:
        specs = _schema_item_specs(static_context)
        metadata = self._source_metadata(static_context, specs)
        source_fingerprint = str(metadata["source_fingerprint"])
        db_id = str(static_context.db_id)
        previous = self._source_by_db.get(db_id)
        if previous is not None and previous != source_fingerprint:
            self._alignment_lru.discard_where(lambda key: key.db_id == db_id)
            self._described_schema_lru.discard_where(
                lambda key: key.source_fingerprint == previous
            )
        self._source_by_db[db_id] = source_fingerprint

        memory_hit = self._schema_memory.get(source_fingerprint)
        if memory_hit is not None:
            return memory_hit

        cache_path = self._cache_path(
            static_context, str(metadata["model_files_sha256"])
        )
        disk_hit = _load_npz(cache_path, specs, source_fingerprint)
        if disk_hit is not None:
            self._schema_memory[source_fingerprint] = disk_hit
            return disk_hit

        if specs:
            embeddings = self.encoder.encode_texts(
                tuple(spec.text for spec in specs)
            )
            items = _items_from_specs(specs, embeddings, source_fingerprint)
        else:
            items = SchemaItemEmbeddings.empty(source_fingerprint)
        _write_npz_atomic(cache_path, items, metadata)
        self._schema_memory[source_fingerprint] = items
        return items

    def align(
        self,
        *,
        question: str,
        active_schema: Mapping[str, Sequence[str]],
        static_context: StaticSchemaContext,
        visible_descriptions: Mapping[str, Mapping[str, str]] | None = None,
    ) -> SchemaAlignment:
        """Compute or reuse alignment keyed only by stage-visible inputs."""
        static_items = self.get_or_build(static_context)
        visible_description_sha256 = _sha256_text(
            _canonical_json(_descriptions_for_hash(visible_descriptions))
        )
        key = AlignmentCacheKey(
            question_sha256=_sha256_text(question),
            db_id=str(static_context.db_id),
            active_schema_sha256=_sha256_text(
                _canonical_json(_schema_for_hash(active_schema))
            ),
            visible_description_sha256=visible_description_sha256,
        )
        hit = self._alignment_lru.get(key)
        if hit is not None:
            return hit

        if visible_descriptions is None:
            items = static_items
        else:
            items = self._get_described_schema_items(
                static_context=static_context,
                static_items=static_items,
                visible_descriptions=visible_descriptions,
                visible_description_sha256=visible_description_sha256,
            )
        question_embedding = self._get_question_embedding(question)
        alignment = compute_schema_alignment(
            question,
            question_embedding,
            active_schema,
            static_context,
            items,
        )
        self._alignment_lru.put(key, alignment)
        return alignment

    def _get_described_schema_items(
        self,
        *,
        static_context: StaticSchemaContext,
        static_items: SchemaItemEmbeddings,
        visible_descriptions: Mapping[str, Mapping[str, str]],
        visible_description_sha256: str,
    ) -> SchemaItemEmbeddings:
        key = DescribedSchemaCacheKey(
            source_fingerprint=static_items.source_fingerprint,
            visible_description_sha256=visible_description_sha256,
        )
        hit = self._described_schema_lru.get(key)
        if hit is not None:
            return hit
        specs = _schema_item_specs(static_context, visible_descriptions)
        encoded = self.encoder.encode_texts(tuple(spec.text for spec in specs))
        items = _items_from_specs(
            specs,
            encoded,
            source_fingerprint=_sha256_text(_canonical_json(list(key))),
        )
        self._described_schema_lru.put(key, items)
        return items

    def _get_question_embedding(self, question: str) -> np.ndarray:
        _, model_fingerprint = self._model_source()
        provider = self._a3_question_embedding_provider
        if provider is not None:
            assert self._a3_question_data_sha256 is not None
            return provider.get(
                question,
                model_fingerprint=model_fingerprint,
                question_data_sha256=self._a3_question_data_sha256,
            )
        if model_fingerprint != self._question_model_fingerprint:
            self._question_embedding_lru.clear()
            self._question_model_fingerprint = model_fingerprint
        key = QuestionEmbeddingCacheKey(
            model_fingerprint=model_fingerprint,
            question_sha256=_sha256_text(question),
        )
        hit = self._question_embedding_lru.get(key)
        if hit is not None:
            return hit
        embedding = np.asarray(
            self.encoder.encode_texts([question])[0], dtype=np.float32
        )
        if embedding.shape != (EMBEDDING_DIM,) or not np.isfinite(embedding).all():
            raise ValueError("question embedding must be finite and 1024D")
        embedding = np.ascontiguousarray(embedding)
        embedding.setflags(write=False)
        self._question_embedding_lru.put(key, embedding)
        return embedding

    def encode_candidate_sql(
        self, static_context: StaticSchemaContext, sql: str
    ) -> np.ndarray:
        """Reuse candidate SQL embeddings by database content and SQL only."""
        sql = str(sql)
        _, model_fingerprint = self._model_source()
        if model_fingerprint != self._candidate_model_fingerprint:
            self._candidate_sql_lru.clear()
            self._candidate_model_fingerprint = model_fingerprint
        key = (self.database_fingerprint(static_context), _sha256_text(sql))
        hit = self._candidate_sql_lru.get(key)
        if hit is not None:
            return hit
        embedding = np.asarray(self.encoder.encode_texts([sql])[0], dtype=np.float32)
        if embedding.shape != (EMBEDDING_DIM,) or not np.isfinite(embedding).all():
            raise ValueError("candidate SQL embedding must be finite and 1024D")
        embedding = np.ascontiguousarray(embedding)
        embedding.setflags(write=False)
        self._candidate_sql_lru.put(key, embedding)
        return embedding


class ReadOnlySchemaSemanticCache(SchemaSemanticCache):
    """Validated semantic-cache hits only; misses never reach the write path."""

    read_only = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.miss_count = 0
        self.write_attempts = 0

    @classmethod
    def from_cache(
        cls,
        cache: SchemaSemanticCache,
    ) -> "ReadOnlySchemaSemanticCache":
        if not isinstance(cache, SchemaSemanticCache):
            raise TypeError("semantic cache must be a SchemaSemanticCache")
        instance = cls.__new__(cls)
        instance.__dict__ = dict(cache.__dict__)
        instance.miss_count = 0
        instance.write_attempts = 0
        return instance

    def _cache_miss_error(self, cache_path: Path) -> BaseException:
        return RuntimeError(f"read-only State V3 semantic cache miss: {cache_path}")

    def get_or_build(
        self, static_context: StaticSchemaContext
    ) -> SchemaItemEmbeddings:
        specs = _schema_item_specs(static_context)
        metadata = self._source_metadata(static_context, specs)
        source_fingerprint = str(metadata["source_fingerprint"])
        db_id = str(static_context.db_id)
        previous = self._source_by_db.get(db_id)
        if previous is not None and previous != source_fingerprint:
            self._alignment_lru.discard_where(lambda key: key.db_id == db_id)
            self._described_schema_lru.discard_where(
                lambda key: key.source_fingerprint == previous
            )
        self._source_by_db[db_id] = source_fingerprint

        memory_hit = self._schema_memory.get(source_fingerprint)
        if memory_hit is not None:
            return memory_hit
        cache_path = self._cache_path(
            static_context,
            str(metadata["model_files_sha256"]),
        )
        disk_hit = _load_npz(cache_path, specs, source_fingerprint)
        if disk_hit is None:
            self.miss_count += 1
            self.write_attempts += 1
            raise self._cache_miss_error(cache_path)
        self._schema_memory[source_fingerprint] = disk_hit
        return disk_hit


def compute_schema_alignment(
    question: str,
    question_embedding: np.ndarray,
    active_schema: Mapping[str, Sequence[str]],
    full_schema: StaticSchemaContext,
    item_embeddings: SchemaItemEmbeddings,
    top_k: int = DEFAULT_TOP_K,
) -> SchemaAlignment:
    """Compute finite query-schema metrics and question-conditioned pooling."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    question_vector = np.asarray(question_embedding, dtype=np.float32)
    if question_vector.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"question_embedding must have shape ({EMBEDDING_DIM},)"
        )
    if not np.isfinite(question_vector).all():
        raise ValueError("question_embedding must be finite")
    question_norm = float(np.linalg.norm(question_vector))
    if question_norm > 0.0:
        question_vector = question_vector / question_norm
    else:
        question_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    selected_schema = _resolve_active_schema(active_schema, full_schema)
    table_indices, column_indices = _active_item_indices(
        selected_schema, item_embeddings
    )
    active_indices = table_indices + column_indices
    if not active_indices and item_embeddings.names:
        active_indices = list(range(len(item_embeddings.names)))
        table_indices = [
            index
            for index in active_indices
            if item_embeddings.kinds[index] == "table"
        ]
        column_indices = [
            index
            for index in active_indices
            if item_embeddings.kinds[index] == "column"
        ]

    raw_scores = np.clip(
        item_embeddings.embeddings @ question_vector,
        -1.0,
        1.0,
    )
    table_metrics = _alignment_metrics(raw_scores, table_indices, top_k)
    column_metrics = _alignment_metrics(raw_scores, column_indices, top_k)
    full_column_indices = [
        index
        for index, kind in enumerate(item_embeddings.kinds)
        if kind == "column"
    ]
    full_column_metrics = _alignment_metrics(
        raw_scores, full_column_indices, top_k
    )
    delta = column_metrics[1] - full_column_metrics[1]

    pooled, available = _pooled_schema_embedding(
        raw_scores,
        item_embeddings.embeddings,
        active_indices,
        top_k,
    )
    full_table_count = len(full_schema.tables)
    full_column_count = sum(len(columns) for columns in full_schema.tables.values())
    active_table_count = len(selected_schema)
    active_column_count = sum(len(columns) for columns in selected_schema.values())

    return SchemaAlignment(
        table_top1=table_metrics[0],
        table_top3_mean=table_metrics[1],
        table_margin=table_metrics[2],
        table_entropy=table_metrics[3],
        column_top1=column_metrics[0],
        column_top3_mean=column_metrics[1],
        column_margin=column_metrics[2],
        column_entropy=column_metrics[3],
        token_coverage=_token_coverage(question, item_embeddings, active_indices),
        active_table_ratio=(
            active_table_count / full_table_count if full_table_count else 0.0
        ),
        active_column_ratio=(
            active_column_count / full_column_count if full_column_count else 0.0
        ),
        active_alignment_delta_from_full=float(np.clip(delta, -1.0, 1.0)),
        schema_embedding=pooled,
        available=available,
    )


def _validate_embedding_matrix(
    embeddings: np.ndarray, *, expected_rows: int, name: str
) -> None:
    if embeddings.shape != (expected_rows, EMBEDDING_DIM):
        raise ValueError(
            f"{name} must have shape ({expected_rows}, {EMBEDDING_DIM}), "
            f"got {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{name} must be finite")
    if expected_rows:
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
            raise ValueError(f"{name} must be L2 normalized")


def _schema_item_specs(
    static_context: StaticSchemaContext,
    descriptions: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[_SchemaItemSpec, ...]:
    column_contexts = {
        (column.table.casefold(), column.name.casefold()): column
        for column in static_context.columns
    }
    visible = _description_lookup(descriptions)
    specs: list[_SchemaItemSpec] = []
    for table, columns in static_context.tables.items():
        specs.append(
            _SchemaItemSpec(
                name=table,
                kind="table",
                table=table,
                column="",
                text=f"table: {table}; columns: {','.join(columns)}",
            )
        )
        for column in columns:
            context = column_contexts.get((table.casefold(), column.casefold()))
            data_type = context.data_type if context is not None else ""
            static_description = context.description if context is not None else ""
            description = visible.get(
                (table.casefold(), column.casefold()), static_description
            )
            specs.append(
                _SchemaItemSpec(
                    name=column,
                    kind="column",
                    table=table,
                    column=column,
                    text=(
                        f"table: {table}; column: {column}; type: {data_type}; "
                        f"description: {description}"
                    ),
                )
            )
    return tuple(specs)


def _description_lookup(
    descriptions: Mapping[str, Mapping[str, str]] | None,
) -> dict[tuple[str, str], str]:
    if not isinstance(descriptions, Mapping):
        return {}
    lookup: dict[tuple[str, str], str] = {}
    for table, columns in descriptions.items():
        if not isinstance(columns, Mapping):
            continue
        for column, description in columns.items():
            if isinstance(description, str):
                lookup[(str(table).casefold(), str(column).casefold())] = description
    return lookup


def _items_from_specs(
    specs: Sequence[_SchemaItemSpec],
    embeddings: np.ndarray,
    source_fingerprint: str,
) -> SchemaItemEmbeddings:
    return SchemaItemEmbeddings(
        names=tuple(spec.name for spec in specs),
        kinds=tuple(spec.kind for spec in specs),
        tables=tuple(spec.table for spec in specs),
        columns=tuple(spec.column for spec in specs),
        texts=tuple(spec.text for spec in specs),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        source_fingerprint=source_fingerprint,
    )


def _load_npz(
    cache_path: Path,
    specs: Sequence[_SchemaItemSpec],
    expected_fingerprint: str,
) -> SchemaItemEmbeddings | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            required = {
                "names",
                "kinds",
                "tables",
                "columns",
                "texts",
                "embeddings",
                "source_fingerprint",
                "payload_sha256",
                "metadata_json",
            }
            if not required.issubset(payload.files):
                return None
            source_fingerprint = str(payload["source_fingerprint"].item())
            if source_fingerprint != expected_fingerprint:
                return None
            names = tuple(str(value) for value in payload["names"].tolist())
            kinds = tuple(str(value) for value in payload["kinds"].tolist())
            tables = tuple(str(value) for value in payload["tables"].tolist())
            columns = tuple(str(value) for value in payload["columns"].tolist())
            texts = tuple(str(value) for value in payload["texts"].tolist())
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            expected_payload_hash = str(payload["payload_sha256"].item())
            actual_payload_hash = _embedding_payload_hash(
                names, kinds, tables, columns, texts, embeddings
            )
            if actual_payload_hash != expected_payload_hash:
                return None
            expected_fields = (
                tuple(spec.name for spec in specs),
                tuple(spec.kind for spec in specs),
                tuple(spec.table for spec in specs),
                tuple(spec.column for spec in specs),
                tuple(spec.text for spec in specs),
            )
            if (names, kinds, tables, columns, texts) != expected_fields:
                return None
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata.get("source_fingerprint") != expected_fingerprint:
                return None
        return SchemaItemEmbeddings(
            names=names,
            kinds=kinds,
            tables=tables,
            columns=columns,
            texts=texts,
            embeddings=embeddings,
            source_fingerprint=source_fingerprint,
        )
    except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError):
        return None


def _write_npz_atomic(
    cache_path: Path,
    items: SchemaItemEmbeddings,
    metadata: Mapping[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload_hash = _embedding_payload_hash(
        items.names,
        items.kinds,
        items.tables,
        items.columns,
        items.texts,
        items.embeddings,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                names=np.asarray(items.names, dtype=str),
                kinds=np.asarray(items.kinds, dtype=str),
                tables=np.asarray(items.tables, dtype=str),
                columns=np.asarray(items.columns, dtype=str),
                texts=np.asarray(items.texts, dtype=str),
                embeddings=np.asarray(items.embeddings, dtype=np.float32),
                source_fingerprint=np.asarray(items.source_fingerprint),
                payload_sha256=np.asarray(payload_hash),
                metadata_json=np.asarray(_canonical_json(dict(metadata))),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _embedding_payload_hash(
    names: Sequence[str],
    kinds: Sequence[str],
    tables: Sequence[str],
    columns: Sequence[str],
    texts: Sequence[str],
    embeddings: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            [list(names), list(kinds), list(tables), list(columns), list(texts)]
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(embeddings, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _resolve_active_schema(
    active_schema: Mapping[str, Sequence[str]],
    full_schema: StaticSchemaContext,
) -> dict[str, tuple[str, ...]]:
    table_lookup = {table.casefold(): table for table in full_schema.tables}
    selected: dict[str, tuple[str, ...]] = {}
    if isinstance(active_schema, Mapping):
        for raw_table, raw_columns in active_schema.items():
            table = table_lookup.get(str(raw_table).casefold())
            if table is None:
                continue
            if isinstance(raw_columns, str):
                candidates: Sequence[Any] = (raw_columns,)
            elif isinstance(raw_columns, Sequence):
                candidates = raw_columns
            else:
                candidates = ()
            column_lookup = {
                column.casefold(): column for column in full_schema.tables[table]
            }
            columns: list[str] = []
            for raw_column in candidates:
                column = column_lookup.get(str(raw_column).casefold())
                if column is not None and column not in columns:
                    columns.append(column)
            selected[table] = tuple(columns)
    if selected:
        return selected
    return {
        table: tuple(columns) for table, columns in full_schema.tables.items()
    }


def _active_item_indices(
    active_schema: Mapping[str, Sequence[str]],
    items: SchemaItemEmbeddings,
) -> tuple[list[int], list[int]]:
    table_keys = {str(table).casefold() for table in active_schema}
    column_keys = {
        (str(table).casefold(), str(column).casefold())
        for table, columns in active_schema.items()
        for column in columns
    }
    table_indices: list[int] = []
    column_indices: list[int] = []
    for index, kind in enumerate(items.kinds):
        if kind == "table" and items.tables[index].casefold() in table_keys:
            table_indices.append(index)
        elif kind == "column" and (
            items.tables[index].casefold(), items.columns[index].casefold()
        ) in column_keys:
            column_indices.append(index)
    return table_indices, column_indices


def _alignment_metrics(
    raw_scores: np.ndarray,
    indices: Sequence[int],
    top_k: int,
) -> tuple[float, float, float, float]:
    if not indices:
        return 0.0, 0.0, 0.0, 0.0
    selected_raw = np.asarray(raw_scores[list(indices)], dtype=np.float64)
    order = np.argsort(selected_raw, kind="stable")[::-1]
    ordered_raw = selected_raw[order]
    mapped = np.clip((ordered_raw + 1.0) / 2.0, 0.0, 1.0)
    top1 = float(mapped[0])
    top3 = float(mapped[: min(3, len(mapped))].mean())
    margin = float(top1 - mapped[1]) if len(mapped) >= 2 else 0.0
    count = min(int(top_k), len(ordered_raw))
    if count <= 1:
        entropy = 0.0
    else:
        probabilities = _softmax(ordered_raw[:count] / SOFTMAX_TEMPERATURE)
        entropy = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)))
            / math.log(count)
        )
    return (
        float(np.clip(top1, 0.0, 1.0)),
        float(np.clip(top3, 0.0, 1.0)),
        float(np.clip(margin, 0.0, 1.0)),
        float(np.clip(entropy, 0.0, 1.0)),
    )


def _pooled_schema_embedding(
    raw_scores: np.ndarray,
    embeddings: np.ndarray,
    active_indices: Sequence[int],
    top_k: int,
) -> tuple[np.ndarray, bool]:
    if not active_indices:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32), False
    indices = np.asarray(active_indices, dtype=np.int64)
    count = min(int(top_k), len(indices))
    order = np.argsort(raw_scores[indices], kind="stable")[-count:]
    top_indices = indices[order]
    weights = _softmax(
        np.asarray(raw_scores[top_indices], dtype=np.float64)
        / SOFTMAX_TEMPERATURE
    )
    pooled = (
        weights[:, None] * np.asarray(embeddings[top_indices], dtype=np.float64)
    ).sum(axis=0)
    norm = float(np.linalg.norm(pooled))
    if not math.isfinite(norm) or norm <= 0.0:
        fallback = np.asarray(embeddings[top_indices[-1]], dtype=np.float64)
        norm = float(np.linalg.norm(fallback))
        if not math.isfinite(norm) or norm <= 0.0:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32), False
        pooled = fallback
    return (pooled / norm).astype(np.float32), True


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def _token_coverage(
    question: str,
    items: SchemaItemEmbeddings,
    active_indices: Sequence[int],
) -> float:
    question_tokens = {
        token
        for token in _lexical_tokens(question)
        if token not in _STOP_WORDS
    }
    if not question_tokens:
        return 0.0
    schema_tokens: set[str] = set()
    for index in active_indices:
        for token in _lexical_tokens(items.texts[index]):
            schema_tokens.add(token)
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                schema_tokens.add(token[:-1])
    return float(len(question_tokens & schema_tokens) / len(question_tokens))


def _lexical_tokens(text: str) -> tuple[str, ...]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return tuple(
        token.casefold()
        for token in re.findall(r"[^\W_]+", separated, flags=re.UNICODE)
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _schema_for_hash(schema: Mapping[str, Sequence[str]]) -> list[list[Any]]:
    if not isinstance(schema, Mapping):
        return []
    rows: list[list[Any]] = []
    for table, columns in schema.items():
        if isinstance(columns, str):
            normalized_columns = [columns]
        elif isinstance(columns, Sequence):
            normalized_columns = [str(column) for column in columns]
        else:
            normalized_columns = []
        rows.append([str(table), sorted(normalized_columns)])
    return sorted(rows, key=lambda row: row[0])


def _descriptions_for_hash(
    descriptions: Mapping[str, Mapping[str, str]] | None,
) -> list[list[Any]]:
    if not isinstance(descriptions, Mapping):
        return []
    rows: list[list[Any]] = []
    for table, columns in descriptions.items():
        column_rows = []
        if isinstance(columns, Mapping):
            column_rows = sorted(
                [str(column), str(description)]
                for column, description in columns.items()
            )
        rows.append([str(table), column_rows])
    return sorted(rows, key=lambda row: row[0])


def _safe_component(value: str) -> str:
    text = str(value)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    if safe == text and safe:
        return safe
    prefix = safe or "value"
    return f"{prefix}-{_sha256_text(text)[:12]}"
