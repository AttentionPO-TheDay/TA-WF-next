"""Matched-width packet inputs for the ordered-representation experiment.

Both conditions retain existing summaries and observed-position indicators.
The control repeats the patch mean; the candidate retains each signed direction.
Only the packet tensor changes: run/window values, masks and spans are preserved.
This is an information/representation ablation, not an equal effective-rank claim.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from .transformer_proto import GeneratorTokenBatch


def packet_patch_inputs(
    batch: GeneratorTokenBatch, directions: Tensor, *, condition: str, patch_size: int = 50,
) -> GeneratorTokenBatch:
    if condition not in ("summary", "ordered"):
        raise ValueError("condition must be summary or ordered")
    if patch_size <= 0 or directions.ndim != 2 or batch.packet.shape[-1] != 2:
        raise ValueError("expected positive patch size, [B,N] directions and original two-field summaries")
    if directions.shape != (batch.packet.shape[0], batch.packet.shape[1] * patch_size):
        raise ValueError("directions must cover exactly the configured packet budget")
    if not torch.all((directions == 0) | (directions == 1) | (directions == -1)):
        raise ValueError("only signed direction and zero padding are allowed")
    observed = directions != 0
    if ((~observed).cumsum(1).gt(0) & observed).any():
        raise ValueError("internal zero followed by an observed packet is not valid suffix padding")
    direction_patches = directions.to(dtype=batch.packet.dtype).reshape(*batch.packet.shape[:2], patch_size)
    position_mask = observed.reshape_as(direction_patches)
    if not torch.equal(position_mask.any(-1), batch.packet_mask):
        raise ValueError("direction and summary token masks disagree")
    counts = position_mask.sum(-1)
    means = direction_patches.sum(-1) / counts.clamp_min(1)
    transitions = ((direction_patches[..., 1:] != direction_patches[..., :-1]) &
                   position_mask[..., 1:] & position_mask[..., :-1]).sum(-1)
    transitions = transitions / (counts - 1).clamp_min(1)
    if not torch.allclose(batch.packet[..., 0], means, atol=1e-6, rtol=0):
        raise ValueError("raw directions do not match saved patch means")
    if not torch.allclose(batch.packet[..., 1], transitions, atol=1e-6, rtol=0):
        raise ValueError("raw directions do not match saved transition summaries")
    detail = direction_patches if condition == "ordered" else means[..., None] * position_mask
    values = torch.cat((batch.packet, position_mask.to(batch.packet.dtype), detail), dim=-1)
    return replace(batch, packet=values)
