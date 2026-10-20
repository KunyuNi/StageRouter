"""Deterministic, causal feature extraction for RAG State V1."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from RAG.env._shared.state_payload import StatePayload
from RAG.env._shared.stage_action_registry import SKIP_ACTION

from .decision_session import DecisionObservation
from .state_v1_schema import legal_action_mask


_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SENTENCE = re.compile(r"[.!?]+")
_CLAUSE = re.compile(r"[,;:]|\b(?:and|but|or|while|whereas|because|which|that)\b", re.I)
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?!\w)")
_YEAR = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_DATE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    re.I,
)
_CURRENCY_PERCENT = re.compile(r"(?:[$£€¥]\s*\d|\d\s*%)")
_QUOTED = re.compile(r"(?:\"[^\"]+\"|'[^']+')")
_ENTITY = re.compile(r"\b(?:[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*)\b")
_CONJUNCTION = re.compile(r"\b(?:and|or|but|nor|yet|so)\b", re.I)
_COMPARISON = re.compile(
    r"\b(?:more|less|fewer|greater|higher|lower|better|worse|largest|smallest|"
    r"earlier|later|before|after|compared|difference|versus|vs)\b",
    re.I,
)
_TEMPORAL = re.compile(
    r"\b(?:when|year|date|before|after|during|earlier|later|first|last|latest|"
    r"oldest|newest|century|decade)\b",
    re.I,
)
_CAUSAL = re.compile(r"\b(?:why|because|cause|caused|reason|result|led to|due to)\b", re.I)
_NEGATION = re.compile(r"\b(?:not|no|never|neither|without|except)\b", re.I)
_CONSTRAINT = re.compile(
    r"\b(?:before|after|during|between|only|except|not|least|most|more|less|"
    r"first|last|same|different)\b",
    re.I,
)


def _tokens(text: str | None) -> list[str]:
    return [item.lower() for item in _TOKEN.findall(text or "")]


def _token_set(text: str | None) -> set[str]:
    return set(_tokens(text))


def _jaccard(left: str | None, right: str | None) -> float:
    a, b = _token_set(left), _token_set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(1.0, float(denominator))


def _entities(text: str | None) -> list[str]:
    return [item.strip().lower() for item in _ENTITY.findall(text or "")]


def _number_dates(text: str | None) -> set[str]:
    text = text or ""
    return {item.lower() for item in _NUMBER.findall(text)} | {
        item.lower() for item in _DATE.findall(text)
    }


def _retention(source: Sequence[str] | set[str], target: Sequence[str] | set[str]) -> float:
    source_set, target_set = set(source), set(target)
    if not source_set:
        return 1.0
    return len(source_set & target_set) / len(source_set)


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or not a.size:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 0.0 else float(np.dot(a, b) / denominator)


def _semantic_cosine(
    left_text: str | None,
    right_text: str | None,
    embeddings: Mapping[str, np.ndarray] | None,
) -> float:
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    if embeddings is None:
        return 0.0
    return _cosine(embeddings.get(left_text), embeddings.get(right_text))


def question_structured_features(question: str | None) -> np.ndarray:
    text = question or ""
    lower = text.lower()
    entities = _entities(text)
    counts = [
        len(text),
        len(text.split()),
        max(1 if text.strip() else 0, len(_SENTENCE.findall(text))),
        len(_CLAUSE.findall(text)) + (1 if text.strip() else 0),
        len(_NUMBER.findall(text)),
        len(_YEAR.findall(text)),
        len(_DATE.findall(text)),
        len(_CURRENCY_PERCENT.findall(text)),
        len(_QUOTED.findall(text)),
        len(entities),
        len(set(entities)),
        len(_CONJUNCTION.findall(text)),
        len(_COMPARISON.findall(text)),
        len(_TEMPORAL.findall(text)),
        len(_CAUSAL.findall(text)),
        len(_NEGATION.findall(text)),
    ]
    yes_no = bool(re.match(r"\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would)\b", lower))
    flags = [
        yes_no,
        bool(_COMPARISON.search(text)),
        bool(_TEMPORAL.search(text)),
        bool(_CAUSAL.search(text)),
        bool(re.search(r"\b(?:how many|number of|count|quantity|total)\b", text, re.I)),
        bool(re.search(r"\b(?:who|whom|whose|person|people)\b", text, re.I)),
        bool(re.search(r"\b(?:where|city|country|location|place)\b", text, re.I)),
        len(set(entities)) > 1 or len(_CLAUSE.findall(text)) > 0,
    ]
    return np.asarray(
        [math.log1p(value) for value in counts] + [float(value) for value in flags],
        dtype=np.float32,
    )


_QUESTION_TYPES = ("inference", "comparison", "temporal", "null_query", "unknown")
_QUESTION_TYPE_ALIASES = {
    "inference_query": "inference",
    "comparison_query": "comparison",
    "temporal_query": "temporal",
    "inference": "inference",
    "comparison": "comparison",
    "temporal": "temporal",
    "null_query": "null_query",
}


def encode_question_type(raw: Any) -> np.ndarray:
    canonical = _QUESTION_TYPE_ALIASES.get(str(raw or "").strip().lower(), "unknown")
    values = np.zeros(5, dtype=np.float32)
    values[_QUESTION_TYPES.index(canonical)] = 1.0
    return values


@dataclass(frozen=True)
class PassageRecord:
    identity: str
    text: str
    hop: int
    retrieval_rank: int
    retrieval_score: float
    metadata: Mapping[str, Any]


def _passage_identity(item: Mapping[str, Any]) -> str:
    chunk_id = item.get("chunk_id")
    if chunk_id not in (None, ""):
        return f"chunk:{chunk_id}"
    text = str(item.get("text", ""))
    return "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_passages(payload: StatePayload) -> tuple[PassageRecord, ...]:
    records: list[PassageRecord] = []
    seen: set[str] = set()
    for hop, stage_name in ((1, "retrieval1"), (2, "retrieval2")):
        stage = payload.metadata.get(stage_name, {})
        results = stage.get("results", []) if isinstance(stage, Mapping) else []
        per_hop = 0
        for fallback_rank, raw in enumerate(results, start=1):
            if not isinstance(raw, Mapping):
                continue
            identity = _passage_identity(raw)
            if identity in seen:
                continue
            seen.add(identity)
            records.append(
                PassageRecord(
                    identity=identity,
                    text=str(raw.get("text", "")),
                    hop=hop,
                    retrieval_rank=int(raw.get("rank", fallback_rank) or fallback_rank),
                    retrieval_score=float(raw.get("score", 0.0) or 0.0),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
            per_hop += 1
            if per_hop == 3:
                break
    return tuple(records)


def _raw_hop_identities(payload: StatePayload, stage_name: str) -> set[str]:
    stage = payload.metadata.get(stage_name, {})
    results = stage.get("results", []) if isinstance(stage, Mapping) else []
    return {
        _passage_identity(item)
        for item in results[:3]
        if isinstance(item, Mapping)
    }


def query_feature_rows(
    observation: DecisionObservation,
    embeddings: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    payload = observation.payload
    question = payload.question or ""
    rows = np.zeros((2, 8), dtype=np.float32)
    previous = question
    for index, query in enumerate((payload.query_1, payload.query_2)):
        if not query:
            if index == 0:
                previous = question
            continue
        source_entities = _entities(question)
        query_entities = _entities(query)
        rows[index] = np.asarray(
            [
                1.0,
                math.log1p(len(query)),
                math.log1p(len(query.split())),
                _semantic_cosine(question, query, embeddings),
                _jaccard(question, query),
                _retention(source_entities, query_entities),
                _retention(_number_dates(question), _number_dates(query)),
                1.0 - _jaccard(previous, query),
            ],
            dtype=np.float32,
        )
        previous = query
    return rows


def _rerank_by_identity(payload: StatePayload) -> dict[str, Mapping[str, Any]]:
    stage = payload.metadata.get("rerank", {})
    results = stage.get("results", []) if isinstance(stage, Mapping) else []
    answer: dict[str, Mapping[str, Any]] = {}
    for item in results:
        if isinstance(item, Mapping):
            answer[_passage_identity(item)] = item
    return answer


def passage_feature_rows(
    observation: DecisionObservation,
    embeddings: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[PassageRecord, ...]]:
    payload = observation.payload
    passages = extract_passages(payload)
    features = np.zeros((6, 20), dtype=np.float32)
    mask = np.zeros(6, dtype=np.float32)
    rerank = _rerank_by_identity(payload)
    final_context = {text: rank for rank, text in enumerate(payload.context, start=1)}
    source_counts = Counter(str(item.metadata.get("source", "")) for item in passages)
    article_counts = Counter(
        str(item.metadata.get("url") or item.metadata.get("title") or item.identity)
        for item in passages
    )
    question = payload.question or ""
    for index, passage in enumerate(passages[:6]):
        mask[index] = 1.0
        rerank_item = rerank.get(passage.identity)
        metadata = passage.metadata
        merged_rank = index + 1
        final_rank = final_context.get(passage.text, 0)
        source = str(metadata.get("source", ""))
        article = str(metadata.get("url") or metadata.get("title") or passage.identity)
        features[index] = np.asarray(
            [
                1.0,
                float(passage.hop),
                passage.retrieval_rank / 3.0,
                passage.retrieval_score,
                merged_rank / max(1, len(passages)),
                float(rerank_item is not None),
                (float(rerank_item.get("rerank_rank", 0)) / max(1, len(passages))) if rerank_item else 0.0,
                (float(rerank_item.get("rerank_score", 0.0)) / 100.0) if rerank_item else 0.0,
                final_rank / max(1, len(payload.context)) if final_rank else 0.0,
                float(bool(final_rank)),
                _semantic_cosine(question, passage.text, embeddings),
                _semantic_cosine(payload.query_1, passage.text, embeddings),
                _semantic_cosine(payload.query_2, passage.text, embeddings),
                float(bool(metadata.get("title"))),
                float(bool(metadata.get("source"))),
                float(bool(metadata.get("published_at") or metadata.get("date"))),
                float(bool(metadata.get("author"))),
                float(bool(metadata.get("category"))),
                source_counts[source] / max(1, len(passages)) if source else 0.0,
                article_counts[article] / max(1, len(passages)),
            ],
            dtype=np.float32,
        )
    return features, mask, passages


def _stats(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    array = np.asarray(values, dtype=np.float32)
    return [float(array.min()), float(array.max()), float(array.mean()), float(array.std())]


def _entropy(values: Sequence[str]) -> float:
    filtered = [str(value) for value in values if str(value)]
    if not filtered:
        return 0.0
    counts = np.asarray(list(Counter(filtered).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / math.log(len(counts)) if len(counts) > 1 else 0.0


def retrieval_summary_features(
    observation: DecisionObservation,
    embeddings: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    payload = observation.payload
    passages = extract_passages(payload)
    r1 = [item for item in passages if item.hop == 1]
    r2 = [item for item in passages if item.hop == 2]
    raw_count = 0
    for stage_name in ("retrieval1", "retrieval2"):
        stage = payload.metadata.get(stage_name, {})
        if isinstance(stage, Mapping) and isinstance(stage.get("results"), list):
            raw_count += len(stage["results"])
    sources = [str(item.metadata.get("source", "")) for item in passages]
    dates = [str(item.metadata.get("published_at") or item.metadata.get("date") or "") for item in passages]
    categories = [str(item.metadata.get("category", "")) for item in passages]
    titles = [str(item.metadata.get("title", "")) for item in passages]
    authors = [str(item.metadata.get("author", "")) for item in passages]
    parsed_dates = []
    for value in dates:
        if not value:
            continue
        try:
            parsed_dates.append(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except ValueError:
            continue
    date_span = 0.0
    if len(parsed_dates) > 1:
        date_span = min(1.0, (max(parsed_dates) - min(parsed_dates)) / (100 * 365.25 * 86400))
    r1_sources = {str(item.metadata.get("source", "")) for item in r1 if item.metadata.get("source")}
    r2_sources = {str(item.metadata.get("source", "")) for item in r2 if item.metadata.get("source")}
    r1_ids = _raw_hop_identities(payload, "retrieval1")
    r2_ids = _raw_hop_identities(payload, "retrieval2")
    similarities_question = [
        _semantic_cosine(payload.question, item.text, embeddings) for item in passages
    ]
    similarities_q1 = [
        _semantic_cosine(payload.query_1, item.text, embeddings) for item in passages
    ]
    similarities_q2 = [
        _semantic_cosine(payload.query_2, item.text, embeddings) for item in passages
    ]
    top = sorted(similarities_question, reverse=True)
    semantic = [
        float(np.mean(similarities_question)) if similarities_question else 0.0,
        max(similarities_question, default=0.0),
        float(np.mean(similarities_q1)) if similarities_q1 else 0.0,
        max(similarities_q1, default=0.0),
        float(np.mean(similarities_q2)) if similarities_q2 else 0.0,
        max(similarities_q2, default=0.0),
        (top[0] - top[1]) if len(top) > 1 else (top[0] if top else 0.0),
        (max(top[:3]) - min(top[:3])) if len(top) > 1 else 0.0,
    ]
    groups = [
        [
            len(r1),
            len(r2),
            len(passages),
            len(payload.context),
            max(0, raw_count - len(passages)),
            len(r2_ids - r1_ids),
            len({value for value in sources if value}),
            len({value for value in dates if value}),
        ],
        _stats([item.retrieval_score for item in r1])
        + _stats([item.retrieval_score for item in r2]),
        [
            _entropy(sources),
            _entropy(categories),
            date_span,
            sum(not value for value in dates) / max(1, len(dates)),
            len({value for value in titles if value}) / max(1, len(passages)),
            len({value for value in authors if value}) / max(1, len(passages)),
            len(r1_sources & r2_sources) / max(1, len(r1_sources | r2_sources)),
            len(r1_ids & r2_ids) / max(1, len(r1_ids | r2_ids)),
        ],
        semantic,
    ]
    return np.asarray([value for group in groups for value in group], dtype=np.float32)


def transition_delta_features(
    observation: DecisionObservation,
    embeddings: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    values = np.zeros(24, dtype=np.float32)
    if observation.decision_idx == 0 or observation.previous_decision_payload is None:
        return values
    current = observation.payload
    previous = observation.previous_decision_payload
    previous_action = observation.action_history[-1] if observation.action_history else None
    values[-2] = float(previous_action == SKIP_ACTION)
    values[-1] = 1.0

    if observation.decision_idx == 1:
        question = current.question or ""
        effective = current.query_1 or question
        values[:6] = [
            _semantic_cosine(question, effective, embeddings),
            _ratio(len(effective), len(question)),
            _jaccard(question, effective),
            _retention(_entities(question), _entities(effective)) - 1.0,
            _retention(_number_dates(question), _number_dates(effective)),
            float(len(_CONSTRAINT.findall(effective)) - len(_CONSTRAINT.findall(question))),
        ]
    elif observation.decision_idx == 2:
        q1 = current.query_1 or current.question or ""
        q2 = current.query_2 or q1
        old_passages = extract_passages(previous)
        new_passages = extract_passages(current)
        old_ids = {item.identity for item in old_passages}
        new_ids = {item.identity for item in new_passages}
        old_sources = {str(item.metadata.get("source", "")) for item in old_passages if item.metadata.get("source")}
        new_sources = {str(item.metadata.get("source", "")) for item in new_passages if item.metadata.get("source")}
        old_dates = {str(item.metadata.get("published_at") or item.metadata.get("date")) for item in old_passages if item.metadata.get("published_at") or item.metadata.get("date")}
        new_dates = {str(item.metadata.get("published_at") or item.metadata.get("date")) for item in new_passages if item.metadata.get("published_at") or item.metadata.get("date")}
        added = new_ids - old_ids
        values[6:14] = [
            _semantic_cosine(q1, q2, embeddings),
            _ratio(len(q2), len(q1)),
            1.0 - _jaccard(q1, q2),
            len(added),
            len(added) / max(1, len(new_ids)),
            len(current.context) - len(previous.context),
            len(new_sources) - len(old_sources),
            len(new_dates) - len(old_dates),
        ]
    elif observation.decision_idx == 3:
        old_passages = extract_passages(previous)
        old_rank = {item.identity: index for index, item in enumerate(old_passages, 1)}
        reranked = _rerank_by_identity(current)
        displacements = [
            abs(old_rank[identity] - int(item.get("rerank_rank", old_rank[identity])))
            for identity, item in reranked.items()
            if identity in old_rank
        ]
        ordered = sorted(
            (
                (int(item.get("rerank_rank", 10**6)), identity, item)
                for identity, item in reranked.items()
            )
        )
        scores = [float(item.get("rerank_score", 0.0)) / 100.0 for _, _, item in ordered]
        old_top = old_passages[0].identity if old_passages else None
        new_top = ordered[0][1] if ordered else old_top
        correlation = 0.0
        pairs = [
            (old_rank[identity], rank)
            for rank, identity, _item in ordered
            if identity in old_rank
        ]
        if len(pairs) > 1:
            correlation = float(np.corrcoef(np.asarray(pairs).T)[0, 1])
            if not math.isfinite(correlation):
                correlation = 0.0
        old_similarity = float(
            np.mean(
                [
                    _semantic_cosine(previous.question, item.text, embeddings)
                    for item in old_passages
                ]
            )
        ) if old_passages else 0.0
        new_similarity = float(
            np.mean(
                [
                    _semantic_cosine(current.question, text, embeddings)
                    for text in current.context
                ]
            )
        ) if current.context else 0.0
        values[14:22] = [
            float(np.mean(displacements)) if displacements else 0.0,
            max(displacements, default=0.0),
            float(old_top != new_top),
            correlation,
            float(np.mean(scores)) if scores else 0.0,
            (max(scores) - min(scores)) if scores else 0.0,
            new_similarity - old_similarity,
            len(current.context) - len(previous.context),
        ]
    return values


def route_control_features(observation: DecisionObservation) -> np.ndarray:
    result = np.zeros(32, dtype=np.float32)
    result[observation.decision_idx] = 1.0
    offset = 4
    for slot in range(3):
        action = observation.action_history[slot] if slot < len(observation.action_history) else None
        category = 0 if action is None else int(action) + 1
        result[offset + slot * 5 + category] = 1.0
    offset += 15
    result[offset : offset + 4] = [
        float(index < observation.decision_idx) for index in range(4)
    ]
    offset += 4
    result[offset : offset + 4] = legal_action_mask(observation.decision_idx)
    offset += 4
    result[offset] = observation.decision_idx / 3.0
    offset += 1
    counts = Counter(observation.action_history)
    result[offset : offset + 4] = [float(counts.get(action, 0)) for action in range(4)]
    return result


@dataclass(frozen=True)
class DecisionCostPriors:
    """Train-only action priors in SKIP, T1, T2, T3 order for each decision."""

    means: tuple[tuple[float, float, float, float], ...]
    identity: str

    def __post_init__(self) -> None:
        if len(self.means) != 4 or any(len(row) != 4 for row in self.means):
            raise ValueError("DecisionCostPriors.means must have shape [4,4]")
        if not self.identity:
            raise ValueError("DecisionCostPriors.identity is required")


def build_cost_history(
    observation: DecisionObservation,
    priors: DecisionCostPriors,
    *,
    c_ref: float = 0.00333879,
) -> np.ndarray:
    if c_ref <= 0.0:
        raise ValueError("c_ref must be positive")
    values = np.zeros(17, dtype=np.float32)
    completed = tuple(float(value) for value in observation.bundled_cost_history)
    values[: min(4, len(completed))] = np.asarray(
        completed[:4], dtype=np.float32
    ) / float(c_ref)
    cumulative = sum(completed)
    values[4] = cumulative
    entries = observation.payload.cost_tracker.stage_costs
    llm_input = sum(
        int(item.get("input_tokens", 0) or 0)
        for item in entries
        if isinstance(item, Mapping) and item.get("call_type") == "llm"
    )
    llm_output = sum(
        int(item.get("output_tokens", 0) or 0)
        for item in entries
        if isinstance(item, Mapping) and item.get("call_type") == "llm"
    )
    embedding_input = sum(
        int(item.get("input_tokens", 0) or 0)
        for item in entries
        if isinstance(item, Mapping) and item.get("call_type") == "embedding"
    )
    if not entries:
        llm_input = int(observation.payload.cost_tracker.total_input_tokens)
        llm_output = int(observation.payload.cost_tracker.total_output_tokens)
    values[5:8] = np.log1p([llm_input, llm_output, embedding_input])
    candidate_costs = np.asarray(priors.means[observation.decision_idx], dtype=np.float32)
    values[8:12] = candidate_costs / float(c_ref)
    legal_in_candidate_order = np.asarray(
        [
            legal_action_mask(observation.decision_idx)[SKIP_ACTION],
            legal_action_mask(observation.decision_idx)[0],
            legal_action_mask(observation.decision_idx)[1],
            legal_action_mask(observation.decision_idx)[2],
        ],
        dtype=np.bool_,
    )
    cheapest = float(candidate_costs[legal_in_candidate_order].min())
    gaps = np.where(legal_in_candidate_order, candidate_costs - cheapest, 0.0)
    values[12:16] = gaps / float(c_ref)
    values[16] = cumulative / float(c_ref)
    return values


def availability_features(observation: DecisionObservation) -> np.ndarray:
    payload = observation.payload
    retrieval1 = payload.metadata.get("retrieval1")
    retrieval2 = payload.metadata.get("retrieval2")
    rerank = payload.metadata.get("rerank")
    return np.asarray(
        [
            bool(payload.query_1),
            isinstance(retrieval1, Mapping) and bool(retrieval1.get("results")),
            bool(payload.query_2),
            isinstance(retrieval2, Mapping) and bool(retrieval2.get("results")),
            bool(payload.context),
            isinstance(rerank, Mapping) and bool(rerank.get("results")),
            observation.decision_idx >= 3 and bool(payload.context),
            bool(observation.bundled_cost_history),
        ],
        dtype=np.float32,
    )


def difficulty_context_features(
    observation: DecisionObservation,
    embeddings: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    question = observation.payload.question or ""
    structured = question_structured_features(question)
    summary = retrieval_summary_features(observation, embeddings)
    passages = extract_passages(observation.payload)
    question_similarities = [
        _semantic_cosine(question, item.text, embeddings) for item in passages
    ]
    scores = [item.retrieval_score for item in passages]
    score_entropy = 0.0
    if scores:
        shifted = np.asarray(scores, dtype=np.float64)
        shifted = shifted - shifted.min() + 1e-8
        probabilities = shifted / shifted.sum()
        score_entropy = float(-(probabilities * np.log(probabilities)).sum())
        if len(scores) > 1:
            score_entropy /= math.log(len(scores))
    metadata_fields = ("title", "source", "published_at", "author", "category")
    missing = sum(not item.metadata.get(field) for item in passages for field in metadata_fields)
    missing_ratio = missing / max(1, len(passages) * len(metadata_fields))
    previous_action = observation.action_history[-1] if observation.action_history else None
    fallback = previous_action == SKIP_ACTION
    if observation.decision_idx == 1 and not observation.payload.query_1:
        fallback = True
    if observation.decision_idx == 2 and not observation.payload.query_2:
        fallback = True
    values = [
        structured[10],
        structured[3],
        structured[23],
        structured[17],
        structured[18],
        float(encode_question_type(observation.payload.metadata.get("question_type"))[3]),
        1.0 - (
            max(scores, default=0.0) - sorted(scores, reverse=True)[1]
            if len(scores) > 1
            else max(scores, default=0.0)
        ),
        score_entropy,
        summary[23],
        1.0 - min(1.0, summary[6] / max(1.0, summary[2])),
        float(np.mean(question_similarities)) if question_similarities else 0.0,
        max(question_similarities, default=0.0),
        sum(similarity > 0.25 for similarity in question_similarities) / max(1, len(question_similarities)),
        missing_ratio,
        len(passages) / 6.0,
        float(fallback),
    ]
    return np.asarray(values, dtype=np.float32)


__all__ = [
    "DecisionCostPriors",
    "PassageRecord",
    "availability_features",
    "build_cost_history",
    "difficulty_context_features",
    "encode_question_type",
    "extract_passages",
    "passage_feature_rows",
    "query_feature_rows",
    "question_structured_features",
    "retrieval_summary_features",
    "route_control_features",
    "transition_delta_features",
]
