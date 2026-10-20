"""Explicit CLI for building the persistent MultiHop-RAG FAISS index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from RAG.retrieval.index import DEFAULT_CORPUS, DEFAULT_PERSIST_ROOT, IndexConfig, build_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--persist-root", type=Path, default=DEFAULT_PERSIST_ROOT)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rebuild", action="store_true", help="replace an existing mismatched index")
    args = parser.parse_args()
    manifest = build_index(
        IndexConfig(
            corpus_path=args.corpus,
            persist_root=args.persist_root,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
        ),
        rebuild=args.rebuild,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
