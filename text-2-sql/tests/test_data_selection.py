"""Offline checks for paper-aligned request selection."""

import numpy as np
import pytest

from pipeline.data_selection import build_feature_views, select_requests


def test_selection_uses_three_views_and_exact_budget():
    random = np.random.default_rng(8)
    qids = list(range(40))
    request = np.column_stack((random.normal(size=(40, 4)),
                               np.repeat([[1.0] * 8], 40, axis=0)))
    request[20:, :4] += 8
    alignment = np.r_[random.normal(0, 0.01, (20, 1)),
                      random.normal(1, 0.01, (20, 1))]
    environment = np.r_[random.normal(0, 0.01, (20, 3)),
                        random.normal(1, 0.01, (20, 3))]
    first = select_requests(qids, request, alignment, environment,
                            fraction=0.25, min_cluster_size=5, min_samples=2)
    second = select_requests(qids, request, alignment, environment,
                             fraction=0.25, min_cluster_size=5, min_samples=2)
    assert first == second
    assert len(first.question_ids) == 10
    assert set(first.question_ids).issubset(qids)
    assert any(qid < 20 for qid in first.question_ids)
    assert any(qid >= 20 for qid in first.question_ids)


def test_database_identifier_is_only_a_schema_lookup_key():
    records = [
        {"question_id": 1, "db_id": "first", "question": "How many?", "SQL": "SELECT count(*) FROM t"},
        {"question_id": 2, "db_id": "second", "question": "How many?", "SQL": "SELECT count(*) FROM t"},
    ]
    schemas = [
        {"db_id": db, "table_names_original": ["t"], "column_names_original": [[0, "id"]]}
        for db in ("first", "second")
    ]

    def embed(_role, text):
        return np.array([len(text), text.count("t"), 1.0, 2.0], dtype=float)

    views = build_feature_views(records, schemas, embed)
    assert all(np.array_equal(view[0], view[1]) for view in views)
    with pytest.raises(ValueError, match="gold SQL"):
        build_feature_views([{**records[0], "SQL": ""}], schemas, embed)
