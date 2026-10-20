"""Persistent LlamaIndex + FAISS index for MultiHop-RAG."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .embedding import MODEL_NAME, TrackedOpenAIEmbedding

try:
    from RAG.env._shared.retriever import RetrievalItem, RetrievalResponse
except ImportError:  # Task 1 supplies these classes; keep Task 2 importable meanwhile.
    from dataclasses import field

    @dataclass(frozen=True)
    class RetrievalItem:  # type: ignore[no-redef]
        chunk_id: str
        text: str
        score: float
        rank: int
        metadata: dict[str, Any]

        def to_json_dict(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(frozen=True)
    class RetrievalResponse:  # type: ignore[no-redef]
        query: str
        results: list[RetrievalItem]
        embedding_input_tokens: int
        embedding_latency_ms: float
        index_identity: str


DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "data" / "raw" / "corpus.json"
DEFAULT_PERSIST_ROOT = Path(__file__).resolve().parents[1] / "cache" / "vector_index"


@dataclass(frozen=True)
class IndexConfig:
    corpus_path: Path | str = DEFAULT_CORPUS
    persist_root: Path | str = DEFAULT_PERSIST_ROOT
    chunk_size: int = 256
    chunk_overlap: int = 50
    top_k: int = 3
    embedding_model: str = MODEL_NAME

    def __post_init__(self) -> None:
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if int(self.chunk_overlap) < 0 or int(self.chunk_overlap) >= int(self.chunk_size):
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if int(self.top_k) < 1:
            raise ValueError("top_k must be at least one")
        if self.embedding_model != MODEL_NAME:
            raise ValueError(f"RAG retrieval model is fixed to {MODEL_NAME!r}")

    @property
    def corpus(self) -> Path:
        return Path(self.corpus_path).expanduser().resolve()

    @property
    def root(self) -> Path:
        return Path(self.persist_root).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).replace(os.sep, "/").encode())
        digest.update(_sha256_file(child).encode())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _manifest_identity(config: IndexConfig, corpus_sha: str) -> dict[str, Any]:
    return {
        "corpus_sha256": corpus_sha,
        "chunk_size": int(config.chunk_size),
        "chunk_overlap": int(config.chunk_overlap),
        "embedding_model": config.embedding_model,
        "vector_metric": "IndexFlatIP",
        "top_k": int(config.top_k),
        "llama_index_core": _package_version("llama-index-core"),
        "llama_index_vector_stores_faiss": _package_version("llama-index-vector-stores-faiss"),
    }


def _load_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import faiss
        from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.vector_stores.faiss import FaissVectorStore
    except ImportError as exc:  # pragma: no cover - depends on deployment install
        raise RuntimeError(
            "LlamaIndex retrieval dependencies are missing; install "
            "requirements.txt from the Github root first"
        ) from exc
    return (
        faiss,
        Document,
        Settings,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
        SentenceSplitter,
        FaissVectorStore,
    )


def _embedding(config: IndexConfig) -> TrackedOpenAIEmbedding:
    return TrackedOpenAIEmbedding(model_name=config.embedding_model, embed_batch_size=10)


def _documents(corpus_path: Path, Document: Any) -> list[Any]:
    with corpus_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Corpus must be a non-empty JSON list: {corpus_path}")
    docs = []
    for doc_index, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"Corpus row {doc_index} is not an object")
        docs.append(
            Document(
                id_=f"doc-{doc_index}",
                text=str(item.get("body") or ""),
                metadata={
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "published_at": item.get("published_at"),
                    "author": item.get("author"),
                    "category": item.get("category"),
                },
            )
        )
    return docs


def _build_nodes(docs: list[Any], splitter: Any) -> list[Any]:
    def stable_id(chunk_index: int, document: Any) -> str:
        return f"{document.id_}:chunk-{chunk_index}"

    # SentenceSplitter's id_func receives the local chunk index and parent doc.
    splitter.id_func = stable_id
    nodes = splitter.get_nodes_from_documents(docs, show_progress=False)
    if not nodes:
        raise ValueError("SentenceSplitter produced no index nodes")
    return nodes


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class MultiHopIndex:
    def __init__(
        self,
        index: Any,
        config: IndexConfig,
        manifest: dict[str, Any],
        embedding: TrackedOpenAIEmbedding,
    ) -> None:
        self.index = index
        self.config = config
        self.manifest = manifest
        self.embedding = embedding
        self.index_identity = str(manifest["index_identity"])

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResponse:
        query = str(query or "").strip()
        if not query:
            raise ValueError("retrieval query cannot be empty")
        k = int(top_k)
        if k < 1:
            raise ValueError("top_k must be at least one")
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        usage = self.embedding.pop_query_usage()
        results: list[RetrievalItem] = []
        for rank, node_with_score in enumerate(nodes, start=1):
            node = node_with_score.node
            text = str(node.get_content() or "").strip()
            if not text:
                continue
            score = node_with_score.get_score() if hasattr(node_with_score, "get_score") else 0.0
            try:
                score = float(score or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            metadata = dict(getattr(node, "metadata", {}) or {})
            results.append(
                RetrievalItem(
                    chunk_id=str(getattr(node, "node_id", getattr(node, "id_", ""))),
                    text=text,
                    score=score,
                    rank=rank,
                    metadata=metadata,
                )
            )
        return RetrievalResponse(
            query=query,
            results=results,
            embedding_input_tokens=int(usage.get("input_tokens", 0) or 0),
            embedding_latency_ms=float(usage.get("latency_ms", 0.0) or 0.0),
            index_identity=self.index_identity,
        )


def build_index(config: IndexConfig, *, rebuild: bool = False) -> dict[str, Any]:
    """Build and persist an index, refusing accidental identity replacement."""

    corpus_sha = _sha256_file(config.corpus)
    identity = _manifest_identity(config, corpus_sha)
    root = config.root
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("identity") == identity and (root / "faiss.index").exists() and (root / "llama_store").is_dir():
            return existing
        if not rebuild:
            raise RuntimeError(
                "Existing retrieval index identity differs; pass --rebuild explicitly to replace it"
            )
    elif root.exists() and any(root.iterdir()) and not rebuild:
        raise RuntimeError(
            f"Retrieval index directory is non-empty without a manifest: {root}; "
            "pass --rebuild explicitly"
        )

    faiss, Document, Settings, StorageContext, VectorStoreIndex, _load, SentenceSplitter, FaissVectorStore = _load_dependencies()
    embedding = _embedding(config)
    Settings.embed_model = embedding
    Settings.chunk_size = int(config.chunk_size)
    Settings.chunk_overlap = int(config.chunk_overlap)
    dimension = len(embedding.get_text_embedding("dim probe"))
    embedding.pop_query_usage()
    if dimension <= 0:
        raise RuntimeError("Embedding endpoint returned an invalid dimension")
    docs = _documents(config.corpus, Document)
    splitter = SentenceSplitter(chunk_size=int(config.chunk_size), chunk_overlap=int(config.chunk_overlap))
    nodes = _build_nodes(docs, splitter)

    target = root.parent / f".{root.name}.build-{os.getpid()}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(dimension))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)
        llama_store = target / "llama_store"
        storage_context.persist(persist_dir=str(llama_store))
        faiss.write_index(vector_store.client, str(target / "faiss.index"))
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "corpus_path": str(config.corpus),
            "persist_root": str(root),
            "dimension": int(dimension),
            "document_count": len(docs),
            "node_count": len(nodes),
            "faiss_index_sha256": _sha256_file(target / "faiss.index"),
            "llama_store_sha256": _sha256_tree(llama_store),
        }
        manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest["index_identity"] = hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest()
        _write_json_atomic(target / "manifest.json", manifest)

        if root.exists():
            if not rebuild:
                raise RuntimeError("Refusing to replace an existing retrieval index without --rebuild")
            shutil.rmtree(root)
        os.replace(target, root)
        return manifest
    finally:
        if target.exists():
            shutil.rmtree(target)


def load_index(config: IndexConfig) -> MultiHopIndex:
    """Load a persisted index; never rebuild implicitly."""

    root = config.root
    manifest_path = root / "manifest.json"
    faiss_path = root / "faiss.index"
    llama_store = root / "llama_store"
    if not (manifest_path.is_file() and faiss_path.is_file() and llama_store.is_dir()):
        raise FileNotFoundError(
            f"Persisted retrieval index is incomplete under {root}; run the explicit build command"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = _manifest_identity(config, _sha256_file(config.corpus))
    if manifest.get("identity") != expected:
        raise RuntimeError("Persisted retrieval index identity does not match current configuration")
    if manifest.get("faiss_index_sha256") != _sha256_file(faiss_path) or manifest.get("llama_store_sha256") != _sha256_tree(llama_store):
        raise RuntimeError("Persisted retrieval index artifact hash mismatch")

    faiss, _Document, Settings, StorageContext, _VectorStoreIndex, load_index_from_storage, _Splitter, FaissVectorStore = _load_dependencies()
    embedding = _embedding(config)
    Settings.embed_model = embedding
    Settings.chunk_size = int(config.chunk_size)
    Settings.chunk_overlap = int(config.chunk_overlap)
    vector_store = FaissVectorStore(faiss_index=faiss.read_index(str(faiss_path)))
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(llama_store))
    index = load_index_from_storage(storage_context)
    return MultiHopIndex(index, config, manifest, embedding)
