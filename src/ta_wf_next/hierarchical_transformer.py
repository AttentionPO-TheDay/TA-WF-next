"""Hierarchical classifier for the project's generator views.

Inspired by CipherSight's local-to-global aggregation, but these views are
packet/run/window summaries, not TLS records or network flows.  No resource
annotations, privileged teacher, distillation, or TTA are used here.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .transformer_proto import GeneratorTokenBatch


class HierarchicalViewTransformer(nn.Module):
    """Encode each view locally, then classify from three view summaries."""

    VIEW_ORDER = ("packet", "runs", "windows")

    def __init__(
        self,
        *,
        d_model: int = 52,
        nhead: int = 4,
        dim_feedforward: int = 104,
        num_classes: int = 102,
        dropout: float = 0.1,
        max_lengths: tuple[int, int, int] = (100, 128, 120),
    ) -> None:
        super().__init__()
        if d_model % nhead or any(length <= 0 for length in max_lengths):
            raise ValueError("invalid hierarchical dimensions")
        self.max_lengths = dict(zip(self.VIEW_ORDER, max_lengths))
        widths = {"packet": 2, "runs": 4, "windows": 4}
        self.projections = nn.ModuleDict({name: nn.Linear(widths[name], d_model) for name in self.VIEW_ORDER})
        self.local_cls = nn.ParameterDict({name: nn.Parameter(torch.zeros(1, 1, d_model)) for name in self.VIEW_ORDER})
        self.local_position = nn.ParameterDict({
            name: nn.Parameter(torch.empty(1, length + 1, d_model)) for name, length in self.max_lengths.items()
        })
        self.local_encoders = nn.ModuleDict({
            name: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
                ),
                num_layers=1,
            )
            for name in self.VIEW_ORDER
        })
        self.page_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.view_type = nn.Embedding(4, d_model)
        self.global_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=1,
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        for position in self.local_position.values():
            nn.init.normal_(position, std=0.02)
        for cls in self.local_cls.values():
            nn.init.normal_(cls, std=0.02)
        nn.init.normal_(self.page_cls, std=0.02)

    def forward(self, batch: GeneratorTokenBatch) -> Tensor:
        batch_size = batch.packet.shape[0]
        summaries: list[Tensor] = []
        present: list[Tensor] = []
        for name in self.VIEW_ORDER:
            value = getattr(batch, name)
            valid = getattr(batch, f"{name}_mask")
            if value.shape[1] > self.max_lengths[name]:
                raise ValueError(f"{name} exceeds configured token budget")
            if value.shape[-1] != self.projections[name].in_features:
                raise ValueError(f"{name} feature width mismatch")
            cls = self.local_cls[name].expand(batch_size, -1, -1)
            tokens = torch.cat((cls, self.projections[name](value)), dim=1)
            tokens = tokens + self.local_position[name][:, : tokens.shape[1]]
            mask = torch.cat((torch.ones(batch_size, 1, device=valid.device, dtype=torch.bool), valid), dim=1)
            hidden = self.local_encoders[name](tokens, src_key_padding_mask=~mask)
            summaries.append(hidden[:, 0])
            present.append(valid.any(dim=1))
        page = self.page_cls.expand(batch_size, -1, -1)
        global_tokens = torch.cat((page, torch.stack(summaries, dim=1)), dim=1)
        type_ids = torch.arange(4, device=global_tokens.device)
        global_tokens = global_tokens + self.view_type(type_ids)[None]
        global_valid = torch.cat((
            torch.ones(batch_size, 1, device=global_tokens.device, dtype=torch.bool),
            torch.stack(present, dim=1),
        ), dim=1)
        page_hidden = self.global_encoder(global_tokens, src_key_padding_mask=~global_valid)[:, 0]
        return self.classifier(self.norm(page_hidden))
