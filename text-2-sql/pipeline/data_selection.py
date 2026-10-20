"""Select a compact BIRD training subset before any LLM execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlglot import exp, parse_one


def normalize_train_records(records: Sequence[Mapping]) -> list[dict]:
    normalized = []
    for index, raw in enumerate(records):
        row = dict(raw)
        row["question_id"] = int(row.get("question_id", row.get("id", index)))
        normalized.append(row)
    if len({row["question_id"] for row in normalized}) != len(normalized):
        raise ValueError("training records contain duplicate request IDs")
    return normalized


def sql_structure(sql: str) -> list[float]:
    tree = parse_one(sql, read="sqlite")
    if tree is None:
        raise ValueError("gold SQL could not be parsed")
    nodes = tuple(tree.walk())
    return [
        math.log1p(len(sql)),
        sum(isinstance(node, exp.Join) for node in nodes),
        sum(isinstance(node, exp.Select) for node in nodes),
        int(any(isinstance(node, exp.AggFunc) for node in nodes)),
        int(any(isinstance(node, exp.Group) for node in nodes)),
        int(any(isinstance(node, exp.Order) for node in nodes)),
        int(any(isinstance(node, exp.Having) for node in nodes)),
        int(any(isinstance(node, (exp.Union, exp.Intersect, exp.Except)) for node in nodes)),
    ]


@dataclass(frozen=True)
class SelectionResult:
    question_ids: tuple[int, ...]
    cluster_labels: tuple[int, ...]
    cluster_sizes: Mapping[int, int]
    noise_selected: int


def _schema_views(tables: Sequence[Mapping]) -> dict[str, tuple[str, tuple[float, ...]]]:
    views = {}
    for entry in tables:
        db_id = str(entry["db_id"])
        names = entry.get("table_names_original") or entry.get("table_names") or []
        columns = entry.get("column_names_original") or entry.get("column_names") or []
        if not names:
            raise ValueError(f"schema has no tables: {db_id}")
        lines = [f"Table: {name}" for name in names]
        counts = Counter()
        for table_index, column in columns:
            table_index = int(table_index)
            if table_index < 0 or str(column) == "*":
                continue
            if table_index >= len(names):
                raise ValueError(f"invalid column table index in {db_id}")
            counts[table_index] += 1
            lines.append(f"{names[table_index]}.{column}")
        views[db_id] = (
            "\n".join(lines),
            (float(len(names)), float(sum(counts.values())), float(max(counts.values(), default=0))),
        )
    return views


def build_feature_views(
    records: Sequence[Mapping],
    tables: Sequence[Mapping],
    encode: Callable[[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return request, question-schema alignment, and environment views.

    Database IDs are used only to look up schemas; they are not features.
    Gold SQL supplies offline structural attributes, never a router input.
    """
    schemas = _schema_views(tables)
    schema_vectors = {db: np.asarray(encode("database", text), dtype=np.float64)
                      for db, (text, _) in schemas.items()}
    semantic, structural, alignment, environment = [], [], [], []
    for row in records:
        db_id = str(row.get("db_id") or row.get("db") or "")
        if db_id not in schemas:
            raise ValueError(f"missing schema for request {row['question_id']}")
        question = str(row.get("question") or row.get("query") or "")
        evidence = str(row.get("evidence") or "")
        sql = str(row.get("SQL") or row.get("sql") or row.get("query_sql") or "")
        if not question or not sql:
            raise ValueError(f"request {row['question_id']} needs a question and gold SQL")
        question_vector = np.asarray(encode("task", question), dtype=np.float64)
        task_vector = np.asarray(encode("task", question + "\n" + evidence), dtype=np.float64)
        schema_vector = schema_vectors[db_id]
        if task_vector.shape != question_vector.shape or task_vector.shape != schema_vector.shape:
            raise ValueError("GTE views must have the same vector dimension")
        semantic.append(task_vector)
        structural.append(sql_structure(sql))
        denominator = np.linalg.norm(question_vector) * np.linalg.norm(schema_vector)
        alignment.append([float(np.dot(question_vector, schema_vector) / denominator)
                          if denominator > 0 else 0.0])
        environment.append(np.log1p(schemas[db_id][1]))
    views = tuple(np.asarray(view, dtype=np.float64)
                  for view in (semantic, structural, alignment, environment))
    if any(not np.isfinite(view).all() for view in views):
        raise ValueError("non-finite request-selection feature")
    return np.concatenate(views[:2], axis=1), views[2], views[3]


def _normalized_joint_views(request: np.ndarray, alignment: np.ndarray,
                            environment: np.ndarray, seed: int) -> np.ndarray:
    if request.ndim != 2 or alignment.ndim != 2 or environment.ndim != 2:
        raise ValueError("each feature view must be a matrix")
    if not (len(request) == len(alignment) == len(environment)) or len(request) < 2:
        raise ValueError("feature views need the same number of requests")
    if any(not np.isfinite(view).all() for view in (request, alignment, environment)):
        raise ValueError("feature views must be finite")
    semantic_dim = request.shape[1] - 8
    if semantic_dim < 1 or alignment.shape[1] < 1 or environment.shape[1] < 1:
        raise ValueError("invalid feature-view dimensions")
    semantic = request[:, :semantic_dim]
    if semantic_dim > 32 and len(request) > 33:
        semantic = PCA(n_components=32, svd_solver="randomized", random_state=seed).fit_transform(semantic)
    semantic = StandardScaler().fit_transform(semantic) / math.sqrt(semantic.shape[1])
    structure = StandardScaler().fit_transform(request[:, semantic_dim:]) / math.sqrt(8)
    request_view = np.concatenate((semantic, structure), axis=1) / math.sqrt(2)
    aligned = StandardScaler().fit_transform(alignment) / math.sqrt(alignment.shape[1])
    env = StandardScaler().fit_transform(environment) / math.sqrt(environment.shape[1])
    return np.ascontiguousarray(np.concatenate((request_view, aligned, env), axis=1) / math.sqrt(3))


def _allocate(groups: Mapping[int, Sequence[int]], budget: int) -> dict[int, int]:
    quotas = {label: 0 for label in groups}
    if budget >= len(groups):
        quotas = {label: 1 for label in groups}
        budget -= len(groups)
    for _ in range(budget):
        eligible = [label for label, members in groups.items() if quotas[label] < len(members)]
        if not eligible:
            raise ValueError("selection budget exceeds the available requests")
        chosen = max(eligible, key=lambda label: (
            math.sqrt(len(groups[label])) / (quotas[label] + 1), -label
        ))
        quotas[chosen] += 1
    return quotas


def _spread(indices: Sequence[int], features: np.ndarray, count: int,
            qids: Sequence[int]) -> list[int]:
    remaining = np.asarray(indices, dtype=int)
    if not count:
        return []
    center = features[remaining].mean(axis=0)
    first = min(remaining, key=lambda i: (float(np.linalg.norm(features[i] - center)), qids[i]))
    selected = [int(first)]
    selected_set = {int(first)}
    distances = np.sum((features[remaining] - features[first]) ** 2, axis=1)
    while len(selected) < count:
        candidates = [(float(distances[j]), -qids[i], int(i))
                      for j, i in enumerate(remaining) if i not in selected_set]
        next_index = max(candidates)[2]
        selected.append(next_index)
        selected_set.add(next_index)
        distances = np.minimum(distances, np.sum((features[remaining] - features[next_index]) ** 2, axis=1))
    return selected


def select_requests(
    qids: Sequence[int], request: np.ndarray, alignment: np.ndarray,
    environment: np.ndarray, *, fraction: float = 0.25, seed: int = 1145,
    min_cluster_size: int = 15, min_samples: int = 5, noise_fraction: float = 0.10,
) -> SelectionResult:
    if len(set(qids)) != len(qids) or not 0 < fraction <= 1 or not 0 <= noise_fraction <= 1:
        raise ValueError("invalid request IDs or sampling fractions")
    features = _normalized_joint_views(request, alignment, environment, seed)
    if len(qids) != len(features) or min_cluster_size < 2 or min_samples < 1:
        raise ValueError("invalid request count or clustering parameters")
    budget = max(1, round(len(qids) * fraction))
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(features)
    groups = {int(label): np.flatnonzero(labels == label).tolist()
              for label in sorted(set(labels)) if label >= 0}
    noise = np.flatnonzero(labels == -1).tolist()
    if not groups:
        selected = _spread(noise, features, budget, qids)
    else:
        noise_quota = min(len(noise), round(budget * noise_fraction)) if noise else 0
        noise_quota = min(noise_quota, max(0, budget - min(budget, len(groups))))
        quotas = _allocate(groups, budget - noise_quota)
        probabilities = np.asarray(clusterer.probabilities_, dtype=float)
        selected = []
        for label, members in groups.items():
            typical = sorted(members, key=lambda i: (-probabilities[i], qids[i]))
            boundary = sorted(members, key=lambda i: (probabilities[i], qids[i]))
            chosen = set()
            for index in range(quotas[label]):
                ranked = typical if index % 2 == 0 else boundary
                candidate = next(i for i in ranked if i not in chosen)
                chosen.add(candidate)
                selected.append(candidate)
        selected.extend(_spread(noise, features, noise_quota, qids))
    if len(selected) != budget or len(set(selected)) != budget:
        raise AssertionError("selection did not meet its exact budget")
    return SelectionResult(
        tuple(sorted(int(qids[i]) for i in selected)),
        tuple(int(label) for label in labels),
        {label: len(members) for label, members in groups.items()},
        sum(labels[i] == -1 for i in selected),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train/train.json"))
    parser.add_argument("--schemas", type=Path, default=Path("data/train/train_tables.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/train/selected/paper"))
    parser.add_argument("--gte", type=Path, default=Path(os.getenv("NL2SQL_SEMANTIC_MODEL", "models/gte-large-en-v1-5")))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1145)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--noise-fraction", type=float, default=0.10)
    args = parser.parse_args()
    from rl.state_v25_semantic import StateV25TextEncoder

    raw_train = args.train.read_bytes()
    records = normalize_train_records(json.loads(raw_train))
    tables = json.loads(args.schemas.read_text(encoding="utf-8"))
    encoder = StateV25TextEncoder(args.gte, device=args.device)
    request, alignment, environment = build_feature_views(
        records, tables, lambda role, text: encoder.encode(role, text)[0]
    )
    result = select_requests(
        [int(row["question_id"]) for row in records], request, alignment, environment,
        fraction=args.fraction, seed=args.seed, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples, noise_fraction=args.noise_fraction,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chosen = set(result.question_ids)
    outputs = {
        "selected_qids.json": list(result.question_ids),
        "selected_train.json": sorted((row for row in records if int(row["question_id"]) in chosen),
                                      key=lambda row: int(row["question_id"])),
        "selection_manifest.json": {
            "method": "gte_three_view_hdbscan_v1",
            "train_sha256": hashlib.sha256(raw_train).hexdigest(),
            "schema_sha256": hashlib.sha256(args.schemas.read_bytes()).hexdigest(),
            "gte_contract_sha256": encoder.contract_sha256,
            "requested_fraction": args.fraction,
            "selected_count": len(result.question_ids),
            "cluster_sizes": result.cluster_sizes,
            "noise_count": result.cluster_labels.count(-1),
            "noise_selected": result.noise_selected,
            "seed": args.seed,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "noise_fraction": args.noise_fraction,
            "features": "request GTE + offline SQL structure; question-schema cosine; schema statistics",
        },
    }
    for name, payload in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
