from __future__ import annotations

from pathlib import Path

import numpy as np

from RAG.rl.semantic_config import SemanticModelConfig
from RAG.rl.state_v1_semantic import (
    LocalGTESemanticEncoder,
    SemanticEmbeddingCache,
    load_stage_prompt_embeddings,
)


class CountingFakeGTE:
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.encoded_texts: list[str] = []

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.encoded_texts.extend(texts)
        rows = []
        for text in texts:
            row = np.zeros(self.dimension, dtype=np.float32)
            row[sum(text.encode("utf-8")) % self.dimension] = 2.0
            rows.append(row)
        return np.stack(rows)


def _config(tmp_path: Path) -> SemanticModelConfig:
    model_dir = tmp_path / "gte"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    return SemanticModelConfig(model_path=str(model_dir))


def test_semantic_cache_avoids_second_model_call(tmp_path: Path):
    model = CountingFakeGTE(dimension=1024)
    cache = SemanticEmbeddingCache(tmp_path / "semantic.db")
    encoder = LocalGTESemanticEncoder(_config(tmp_path), model=model, cache=cache)
    first = encoder.encode_texts(["question", "passage"])
    second = encoder.encode_texts(["passage", "question"])

    assert model.encoded_texts == ["question", "passage"]
    np.testing.assert_allclose(first[0], second[1])
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    cache.close()


def test_stage_prompt_embeddings_use_four_decision_rows(tmp_path: Path):
    prompt_file = tmp_path / "stage_prompt.tsv"
    prompt_file.write_text(
        "S1\tfirst stage\nS2\tsecond stage\nS3\tthird stage\nS4\tfourth stage\n",
        encoding="utf-8",
    )
    cache = SemanticEmbeddingCache(tmp_path / "semantic.db")
    encoder = LocalGTESemanticEncoder(
        _config(tmp_path), model=CountingFakeGTE(), cache=cache
    )
    embeddings = load_stage_prompt_embeddings(prompt_file, encoder, cache)

    assert embeddings.shape == (4, 1024)
    assert embeddings.dtype == np.float32
    cache.close()


def test_semantic_identity_changes_with_tokenizer_contract(tmp_path: Path):
    model_dir = _config(tmp_path).model_path
    base = SemanticModelConfig(model_path=model_dir, max_length=512)
    changed = SemanticModelConfig(model_path=model_dir, max_length=256)
    assert base.identity(model_fingerprint="abc") != changed.identity(
        model_fingerprint="abc"
    )


def test_config_freezes_local_gte_contract(tmp_path: Path):
    config = _config(tmp_path)
    assert config.pooling == "cls"
    assert config.normalize is True
    assert config.dimension == 1024
    assert config.max_length == 512
    assert config.batch_size == 32
    assert config.trust_remote_code is True
    assert config.local_files_only is True
    assert config.output_dtype == "float32"
