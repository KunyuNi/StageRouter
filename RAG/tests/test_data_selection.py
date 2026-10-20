"""Dataset-only RAG request selection; no route buffer or provider needed."""

from __future__ import annotations

import numpy as np
import pytest

from RAG.pipeline.data_selection import build_feature_views, normalize_records, select_requests


def test_three_views_use_dataset_only_and_select_half():
    records = normalize_records([
        {
            "question_id": qid,
            "query": f"When did entity {qid % 4} change in {2000 + qid}?",
            "question_type": "temporal" if qid % 2 else "comparison",
            "evidence_list": [{"fact": f"Entity {qid % 4} changed in {2000 + qid}.", "source": "wiki"}],
        }
        for qid in range(30)
    ])

    def fake_encode(texts):
        return np.asarray([
            [float(len(text) % 11), float(sum(map(ord, text)) % 17), 1.0]
            for text in texts
        ])

    request, alignment, environment = build_feature_views(records, fake_encode)
    assert request.shape == (30, 11)
    assert alignment.shape == (30, 1)
    assert environment.shape == (30, 6)
    qids = [row["question_id"] for row in records]
    first, diagnostics = select_requests(qids, request, alignment, environment, min_cluster_size=3, min_samples=2)
    second, _ = select_requests(qids, request, alignment, environment, min_cluster_size=3, min_samples=2)
    assert first == second
    assert len(first) == round(30 * 0.50) == diagnostics["selection_budget"]
    assert len(set(first)) == len(first)


def test_selection_rejects_duplicate_request_ids():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_records([{"question_id": 1, "query": "A"}, {"question_id": 1, "query": "B"}])


def test_missing_gold_evidence_has_zero_alignment():
    rows = normalize_records([{"question_id": 1, "query": "What happened?", "evidence_list": []}])
    _request, alignment, _environment = build_feature_views(
        rows, lambda texts: np.ones((len(texts), 3), dtype=float)
    )
    assert alignment[0, 0] == 0.0
