"""Select RAG training requests before any route or LLM collection.

This is the RAG counterpart of Text-to-SQL's three-view HDBSCAN selector:
request semantics/structure, request-evidence alignment, and environment
structure. Gold evidence is used only for offline selection, never Router State.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SELECTION_FRACTION = 0.50
QUESTION_TYPES = ("comparison", "inference", "temporal")


def normalize_records(records: Sequence[Mapping]) -> list[dict]:
    result = []
    for row in records:
        item = dict(row)
        item["question_id"] = int(item["question_id"])
        if not str(item.get("query", "")).strip():
            raise ValueError(f"empty question for qid {item['question_id']}")
        result.append(item)
    if len({row["question_id"] for row in result}) != len(result):
        raise ValueError("duplicate question IDs")
    return result


def _question_structure(question: str) -> list[float]:
    words = question.split()
    lowered = question.lower()
    return [
        math.log1p(len(question)),
        math.log1p(len(words)),
        math.log1p(len(re.findall(r"\b\d{4}\b", question))),
        math.log1p(len(re.findall(r"\b\d+(?:\.\d+)?\b", question))),
        float(any(word in lowered for word in ("compare", "more than", "less than", "difference"))),
        float(any(word in lowered for word in ("when", "before", "after", "year", "date"))),
        float(any(word in lowered for word in ("why", "how", "because", "cause"))),
        math.log1p(question.count(",") + question.count(";")),
    ]


def build_feature_views(
    records: Sequence[Mapping], encode_texts: Callable[[Sequence[str]], np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build three dataset-only views; no route outcomes or API calls."""
    questions = [str(row["query"]).strip() for row in records]
    evidence = []
    for row in records:
        facts = row.get("evidence_list") or []
        evidence.append(" ".join(str(fact.get("fact", "")) for fact in facts if isinstance(fact, Mapping)))
    question_vec = np.asarray(encode_texts(questions), dtype=np.float64)
    evidence_vec = np.asarray(encode_texts([text or "[no evidence]" for text in evidence]), dtype=np.float64)
    if question_vec.ndim != 2 or question_vec.shape != evidence_vec.shape or len(question_vec) != len(records):
        raise ValueError("GTE question and evidence embeddings must have matching matrix shapes")
    structure = np.asarray([_question_structure(text) for text in questions], dtype=np.float64)
    request = np.concatenate((question_vec, structure), axis=1)
    denominator = np.linalg.norm(question_vec, axis=1) * np.linalg.norm(evidence_vec, axis=1)
    evidence_present = np.asarray([bool(text.strip()) for text in evidence], dtype=bool)
    alignment = np.divide(
        np.sum(question_vec * evidence_vec, axis=1), denominator,
        out=np.zeros(len(records), dtype=np.float64), where=(denominator > 0) & evidence_present,
    ).reshape(-1, 1)
    environment = []
    for row, text in zip(records, evidence):
        facts = [fact for fact in (row.get("evidence_list") or []) if isinstance(fact, Mapping)]
        sources = {str(fact.get("source", "")) for fact in facts if fact.get("source")}
        kind = str(row.get("question_type") or "").lower()
        environment.append([
            math.log1p(len(facts)), math.log1p(len(sources)), math.log1p(len(text)),
            *(float(kind == category) for category in QUESTION_TYPES),
        ])
    environment_view = np.asarray(environment, dtype=np.float64)
    if any(not np.isfinite(view).all() for view in (request, alignment, environment_view)):
        raise ValueError("non-finite selection feature")
    return request, alignment, environment_view


def _joint_features(request: np.ndarray, alignment: np.ndarray, environment: np.ndarray, seed: int) -> np.ndarray:
    if any(view.ndim != 2 for view in (request, alignment, environment)) or len({len(view) for view in (request, alignment, environment)}) != 1:
        raise ValueError("feature views must have equal row counts")
    if request.shape[1] <= 8 or alignment.shape[1] < 1 or environment.shape[1] < 1:
        raise ValueError("invalid feature-view dimensions")
    semantic = request[:, :-8]
    if semantic.shape[1] > 32 and len(request) > 33:
        semantic = PCA(n_components=32, svd_solver="randomized", random_state=seed).fit_transform(semantic)
    semantic = StandardScaler().fit_transform(semantic) / math.sqrt(semantic.shape[1])
    structure = StandardScaler().fit_transform(request[:, -8:]) / math.sqrt(8)
    request_view = np.concatenate((semantic, structure), axis=1) / math.sqrt(2)
    aligned = StandardScaler().fit_transform(alignment) / math.sqrt(alignment.shape[1])
    env = StandardScaler().fit_transform(environment) / math.sqrt(environment.shape[1])
    return np.ascontiguousarray(np.concatenate((request_view, aligned, env), axis=1) / math.sqrt(3))


def _spread(indices: Sequence[int], features: np.ndarray, count: int, qids: Sequence[int]) -> list[int]:
    if count == 0:
        return []
    remaining = np.asarray(indices, dtype=int)
    center = features[remaining].mean(axis=0)
    first = min(remaining, key=lambda i: (float(np.linalg.norm(features[i] - center)), qids[i]))
    selected = [int(first)]
    distances = np.sum((features[remaining] - features[first]) ** 2, axis=1)
    while len(selected) < count:
        chosen = set(selected)
        next_index = max((float(distances[j]), -qids[i], int(i)) for j, i in enumerate(remaining) if i not in chosen)[2]
        selected.append(next_index)
        distances = np.minimum(distances, np.sum((features[remaining] - features[next_index]) ** 2, axis=1))
    return selected


def select_requests(
    qids: Sequence[int], request: np.ndarray, alignment: np.ndarray, environment: np.ndarray,
    *, fraction: float = SELECTION_FRACTION, seed: int = 1145, min_cluster_size: int = 15,
    min_samples: int = 5, noise_fraction: float = 0.10,
) -> tuple[tuple[int, ...], dict]:
    """Return a deterministic representative subset at the requested budget."""
    if len(qids) < 2 or len(set(qids)) != len(qids) or not 0 < fraction <= 1 or not 0 <= noise_fraction <= 1:
        raise ValueError("invalid request IDs or fraction")
    if min_cluster_size < 2 or min_samples < 1:
        raise ValueError("invalid HDBSCAN parameters")
    features = _joint_features(request, alignment, environment, seed)
    if len(features) != len(qids):
        raise ValueError("request and feature counts differ")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(features)
    probabilities = np.asarray(clusterer.probabilities_, dtype=float)
    groups = {int(label): np.flatnonzero(labels == label).tolist() for label in sorted(set(labels)) if label >= 0}
    noise = np.flatnonzero(labels == -1).tolist()
    budget = max(1, round(len(qids) * fraction))
    noise_budget = min(len(noise), round(budget * noise_fraction)) if groups else budget
    if groups:
        noise_budget = min(noise_budget, max(0, budget - min(budget, len(groups))))
    ranked: dict[int, list[int]] = {}
    for label, members in groups.items():
        typical = sorted(members, key=lambda i: (-probabilities[i], qids[i]))
        boundary = sorted(members, key=lambda i: (probabilities[i], qids[i]))
        seen: set[int] = set()
        order = []
        for offset in range(len(members)):
            candidate = next(i for i in (typical if offset % 2 == 0 else boundary) if i not in seen)
            seen.add(candidate)
            order.append(candidate)
        ranked[label] = order
    quotas = Counter({label: 0 for label in ranked})
    selected: list[int] = []
    if ranked and budget - noise_budget >= len(ranked):
        for label in ranked:
            selected.append(ranked[label][0])
            quotas[label] = 1
    while len(selected) < budget - noise_budget:
        eligible = [label for label, members in ranked.items() if quotas[label] < len(members)]
        label = max(eligible, key=lambda key: (math.sqrt(len(ranked[key])) / (quotas[key] + 1), -key))
        selected.append(ranked[label][quotas[label]])
        quotas[label] += 1
    noise_order = _spread(noise, features, noise_budget, qids)
    if noise_order:
        # Interleave atypical requests with clustered representatives.
        merged: list[int] = []
        cluster_cursor = noise_cursor = 0
        while cluster_cursor < len(selected) or noise_cursor < len(noise_order):
            due_noise = noise_cursor < len(noise_order) and (noise_cursor + 1) / len(noise_order) <= (len(merged) + 1) / budget
            if due_noise or cluster_cursor == len(selected):
                merged.append(noise_order[noise_cursor]); noise_cursor += 1
            else:
                merged.append(selected[cluster_cursor]); cluster_cursor += 1
        selected = merged
    if len(selected) != budget or len(set(selected)) != budget:
        raise AssertionError("selection did not meet the exact budget")
    return tuple(int(qids[i]) for i in selected), {
        "cluster_sizes": {str(label): len(members) for label, members in ranked.items()},
        "noise_selected": sum(labels[i] == -1 for i in selected),
        "selection_budget": budget,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("RAG/data/train.json"))
    parser.add_argument("--gte-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("RAG/cache/data_selection"))
    parser.add_argument("--seed", type=int, default=1145)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--noise-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)
    from RAG.rl.semantic_config import SemanticModelConfig
    from RAG.rl.state_v1_semantic import LocalGTESemanticEncoder, SemanticEmbeddingCache

    raw = args.train.read_bytes()
    records = normalize_records(json.loads(raw))
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache = SemanticEmbeddingCache(args.output_root / "semantic_embeddings.db")
    try:
        encoder = LocalGTESemanticEncoder(SemanticModelConfig(model_path=str(args.gte_model)), cache=cache)
        request, alignment, environment = build_feature_views(records, encoder.encode_texts)
        ordered, diagnostics = select_requests(
            [row["question_id"] for row in records], request, alignment, environment,
            seed=args.seed, min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples, noise_fraction=args.noise_fraction,
        )
        by_qid = {row["question_id"]: row for row in records}
        manifest = {
            "method": "rag_gte_three_view_hdbscan_v1", "seed": args.seed,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "gte_identity": encoder.model_identity, "population": len(records),
            "selection_fraction": SELECTION_FRACTION,
            "selected_qids": list(ordered), "diagnostics": diagnostics,
            "uses_route_outcomes": False, "api_calls": 0,
        }
        manifest_path = args.output_root / "selection_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing Data Selection manifest differs; use a new output root")
        (args.output_root / "selected_qids.txt").write_text(
            "".join(f"{qid}\n" for qid in ordered), encoding="utf-8"
        )
        (args.output_root / "selected_train.json").write_text(
            json.dumps([by_qid[qid] for qid in ordered], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"population": len(records), "selected": len(ordered), "api_calls": 0}))
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
