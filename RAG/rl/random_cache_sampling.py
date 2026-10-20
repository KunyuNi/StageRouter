"""Deterministic query sampling helpers for RAG cache ablations."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable


def _canonical_qids(qids: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(qid) for qid in qids)
    if not values:
        raise ValueError("qids must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("duplicate qid")
    return tuple(sorted(values))


def select_nested_qids(
    qids: Iterable[int], fraction: float, *, seed: int = 1145
) -> tuple[int, ...]:
    """Select a deterministic prefix of one canonical seeded qid shuffle."""

    population = list(_canonical_qids(qids))
    value = float(fraction)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("fraction must be in (0, 1]")
    random.Random(int(seed)).shuffle(population)
    count = int(math.floor(len(population) * value + 0.5))
    count = max(1, min(len(population), count))
    return tuple(sorted(population[:count]))


def qid_selection_sha256(qids: Iterable[int]) -> str:
    """Hash a qid set using canonical compact JSON serialization."""

    encoded = json.dumps(
        _canonical_qids(qids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["qid_selection_sha256", "select_nested_qids"]
