"""Supervised local packet-to-token generator, before lossy patch compression.

The CPU experiment owns the learning-rate/freeze policy. The generator owns
local convolutions, within-segment attention, and the final token projection.
Run/window extraction and the downstream hierarchical classifier are unchanged.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hierarchical_transformer import HierarchicalViewTransformer
from .transformer_proto import GeneratorTokenBatch


class LocalPacketGenerator(nn.Module):
    def __init__(self, *, pool: str = "attention", packet_budget: int = 5000,
                 d_model: int = 52) -> None:
        super().__init__()
        if pool not in ("mean", "attention") or packet_budget <= 0 or packet_budget % 50:
            raise ValueError("mean/attention pooling and a positive multiple-of-50 budget required")
        self.pool = pool
        self.packet_budget = packet_budget
        self.conv1 = nn.Conv1d(2, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)
        self.score = nn.Linear(16, 1, bias=False)
        self.projection = nn.Linear(87, d_model)
        nn.init.zeros_(self.score.weight)

    def forward(self, directions: Tensor, observed: Tensor, summaries: Tensor) -> Tensor:
        if directions.ndim != 2 or directions.shape[1] != self.packet_budget:
            raise ValueError("directions must be [B, packet_budget]")
        if observed.shape != directions.shape or observed.dtype != torch.bool:
            raise ValueError("observed must be a bool mask matching directions")
        b, n = directions.shape
        if summaries.shape != (b, n // 50, 2):
            raise ValueError("original two-field patch summaries required")
        mask = observed[:, None].to(summaries.dtype)
        signed = directions.to(summaries.dtype).masked_fill(~observed, 0)
        values = torch.stack((signed, observed.to(summaries.dtype)), dim=1)
        hidden = F.gelu(self.conv1(values)) * mask
        hidden = F.gelu(self.conv2(hidden)) * mask
        # [batch, patch, ordered subbin, position, channel]
        segments = hidden.transpose(1, 2).reshape(b, n // 50, 5, 10, 16)
        valid = observed.reshape(b, n // 50, 5, 10)
        scores = self.score(segments).squeeze(-1) if self.pool == "attention" else torch.zeros_like(valid, dtype=summaries.dtype)
        # Finite masked softmax also for all-padding subbins. Empty outputs = 0.
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        pooled = (segments * weights[..., None]).sum(-2).flatten(-2)
        fractions = valid.to(summaries.dtype).mean(-1)
        tokens = self.projection(torch.cat((pooled, summaries, fractions), dim=-1))
        return tokens * valid.any(-1).any(-1)[..., None]


class _TokenIdentity(nn.Identity):
    # Preserve the existing classifier's explicit feature-width validation.
    def __init__(self, width: int) -> None:
        super().__init__()
        self.in_features = width


class GeneratorClassifier(nn.Module):
    def __init__(self, *, pool: str = "attention", generator_trainable: bool = True,
                 packet_budget: int = 5000, max_runs: int = 128,
                 window_tokens: int = 120, dropout: float = 0.1) -> None:
        super().__init__()
        self.generator = LocalPacketGenerator(pool=pool, packet_budget=packet_budget)
        self.classifier = HierarchicalViewTransformer(
            packet_dim=52, max_lengths=(packet_budget // 50, max_runs, window_tokens), dropout=dropout,
        )
        self.classifier.projections["packet"] = _TokenIdentity(52)
        self.generator.requires_grad_(generator_trainable)
        if pool == "mean":
            self.generator.score.requires_grad_(False)

    def forward(self, original: GeneratorTokenBatch, directions: Tensor, observed: Tensor) -> Tensor:
        packet = self.generator(directions, observed, original.packet)
        return self.classifier(replace(original, packet=packet))
