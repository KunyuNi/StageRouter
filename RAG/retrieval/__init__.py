"""Persistent LlamaIndex/FAISS retrieval for the MultiHop-RAG corpus."""

from .index import IndexConfig, MultiHopIndex, build_index, load_index

__all__ = ["IndexConfig", "MultiHopIndex", "build_index", "load_index"]
