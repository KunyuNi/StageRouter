"""Stage-FiLM legal Dueling DQN networks for RAG State V1."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .state_v1_schema import StateV1Schema


def legal_dueling(
    value: torch.Tensor,
    advantage: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    mask_illegal: bool = True,
) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError("value must have shape [batch,1]")
    if advantage.ndim != 2 or advantage.shape[1] != 4:
        raise ValueError("advantage must have shape [batch,4]")
    if legal_mask.shape != advantage.shape:
        raise ValueError("legal_mask must match advantage shape")
    legal = legal_mask.to(dtype=torch.bool, device=advantage.device)
    if not torch.all(legal.any(dim=1)):
        raise ValueError("every state must expose at least one legal action")
    weights = legal.to(advantage.dtype)
    legal_mean = (advantage * weights).sum(dim=-1, keepdim=True) / weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    q = value + advantage - legal_mean
    return q.masked_fill(~legal, -torch.inf) if mask_illegal else q


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


class _StageFiLMBackbone(nn.Module):
    def __init__(
        self,
        state_schema: StateV1Schema,
        stage_embeddings: torch.Tensor,
        *,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be positive")
        stage_embeddings = torch.as_tensor(stage_embeddings, dtype=torch.float32)
        if stage_embeddings.shape != (4, 1024):
            raise ValueError("stage_embeddings must have shape [4,1024]")
        self.schema = state_schema
        self.register_buffer("stage_embeddings", stage_embeddings.clone())

        self.task_encoder = _mlp(1024 + 24 + 5, 256, 128)
        self.query_encoder = _mlp(1024 + 8, 256, 128)
        self.passage_encoder = _mlp(1024 + 20, 256, 128)
        self.passage_attention_content = nn.Linear(128, 1)
        self.passage_attention_rank = nn.Linear(4, 1, bias=False)
        self.workspace_encoder = _mlp(128 + 128 + 32, 384, 256)
        self.control_encoder = _mlp(32, 128, 64)
        self.context_encoder = _mlp(24 + 17 + 8 + 16, 128, 64)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.final_fusion = _mlp(
            128 + 256 + 64 + 64, 384, self.fusion_hidden_dim
        )

        self.task_film = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.workspace_film = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        nn.init.zeros_(self.task_film[-1].weight)
        nn.init.zeros_(self.task_film[-1].bias)
        nn.init.zeros_(self.workspace_film[-1].weight)
        nn.init.zeros_(self.workspace_film[-1].bias)

    def _block(self, state: torch.Tensor, name: str) -> torch.Tensor:
        block_slice = self.schema.slices[name]
        shape = self.schema.shapes[name]
        return state[:, block_slice].reshape(state.shape[0], *shape)

    def apply_film(
        self,
        task: torch.Tensor,
        workspace: torch.Tensor,
        stages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stages.ndim == 2:
            if stages.shape[1] != 4:
                raise ValueError("stage one-hot must have width 4")
            stage_indices = stages.argmax(dim=1)
        else:
            stage_indices = stages.to(torch.long).reshape(-1)
        prompts = self.stage_embeddings[stage_indices]
        task_delta_gamma, task_beta = self.task_film(prompts).chunk(2, dim=-1)
        workspace_delta_gamma, workspace_beta = self.workspace_film(prompts).chunk(
            2, dim=-1
        )
        return (
            task * (1.0 + task_delta_gamma) + task_beta,
            workspace * (1.0 + workspace_delta_gamma) + workspace_beta,
        )

    def _masked_query_pool(
        self, semantics: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.query_encoder(torch.cat([semantics, features], dim=-1))
        mask = features[:, :, 0:1].clamp(0.0, 1.0)
        return (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _masked_passage_pool(
        self,
        semantics: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.passage_encoder(torch.cat([semantics, features], dim=-1))
        logits = self.passage_attention_content(encoded).squeeze(-1)
        logits = logits + self.passage_attention_rank(features[:, :, 1:5]).squeeze(-1)
        valid = mask.to(torch.bool)
        masked_logits = logits.masked_fill(~valid, -torch.inf)
        has_passage = valid.any(dim=1, keepdim=True)
        safe_logits = torch.where(has_passage, masked_logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return torch.where(has_passage, pooled, torch.zeros_like(pooled))

    def encode_backbone(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != sum(self.schema.flat_sizes.values()):
            raise ValueError("state tensor does not match State V1 schema")
        question = torch.cat(
            [
                self._block(state, "question_semantic"),
                self._block(state, "question_structured"),
                self._block(state, "question_type"),
            ],
            dim=-1,
        )
        task = self.task_encoder(question)

        query_pool = self._masked_query_pool(
            self._block(state, "query_semantics"),
            self._block(state, "query_features"),
        )
        passage_pool = self._masked_passage_pool(
            self._block(state, "passage_semantics"),
            self._block(state, "passage_features"),
            self._block(state, "passage_mask"),
        )
        workspace = self.workspace_encoder(
            torch.cat(
                [query_pool, passage_pool, self._block(state, "retrieval_summary")],
                dim=-1,
            )
        )
        route_control = self._block(state, "route_control")
        control = self.control_encoder(route_control)
        context = self.context_encoder(
            torch.cat(
                [
                    self._block(state, "transition_delta"),
                    self._block(state, "cost_history"),
                    self._block(state, "availability"),
                    self._block(state, "difficulty_context"),
                ],
                dim=-1,
            )
        )
        task, workspace = self.apply_film(task, workspace, route_control[:, :4])
        return self.final_fusion(torch.cat([task, workspace, control, context], dim=-1))


class _DuelingHead(nn.Module):
    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        legal_mask: torch.Tensor,
        *,
        mask_illegal: bool = True,
    ) -> torch.Tensor:
        return legal_dueling(
            self.value(hidden),
            self.advantage(hidden),
            legal_mask,
            mask_illegal=mask_illegal,
        )


class ScalarStageFiLMDuelingDQN(_StageFiLMBackbone):
    def __init__(
        self,
        state_schema: StateV1Schema,
        stage_embeddings: torch.Tensor,
        *,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            state_schema, stage_embeddings, fusion_hidden_dim=fusion_hidden_dim
        )
        self.scalar_head = _DuelingHead(fusion_hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor,
        *,
        mask_illegal: bool = True,
    ) -> torch.Tensor:
        return self.scalar_head(
            self.encode_backbone(state), legal_mask, mask_illegal=mask_illegal
        )


__all__ = [
    "ScalarStageFiLMDuelingDQN",
    "legal_dueling",
]
