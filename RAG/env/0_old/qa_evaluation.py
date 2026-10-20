"""Deterministic normalized token-set F1 evaluation."""

from __future__ import annotations

import re
import string
import unicodedata

from .._shared.state_payload import StatePayload


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value).lower()
    normalized = "".join(ch for ch in normalized if ch not in string.punctuation)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def f1_score_str(prediction: str, label: str) -> float:
    predicted_tokens = set(normalize_text(prediction).split())
    label_tokens = set(normalize_text(label).split())
    overlap = len(predicted_tokens & label_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(label_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_evaluation(payload: StatePayload) -> StatePayload:
    if payload.answer is None:
        raise ValueError("qa_evaluation requires answer")
    if payload.ground_truth is None:
        raise ValueError("qa_evaluation requires ground_truth")
    output = payload.snapshot()
    output.evaluation = {
        "f1_score": f1_score_str(payload.answer, payload.ground_truth),
        "answer": payload.answer,
    }
    return output

