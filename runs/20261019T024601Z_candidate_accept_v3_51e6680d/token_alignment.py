"""Reference-only Token alignment, before the 110 -> 128 projection.

All target statistics and gradients belong to an independent unlabeled
reference set. Forward never estimates statistics from query batches.
"""
from contextlib import contextmanager

import torch
from torch import nn


class Moments:
    def __init__(self, channels=110):
        self.count = 0
        self.total = torch.zeros(channels, dtype=torch.float64)
        self.square = torch.zeros_like(self.total)

    def add(self, tokens):
        values = tokens.detach().reshape(-1, tokens.shape[-1]).cpu().double()
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite source/reference tokens")
        self.count += len(values)
        self.total += values.sum(0)
        self.square += values.square().sum(0)

    def finish(self):
        if self.count == 0:
            raise ValueError("empty reference")
        mean = self.total / self.count
        variance = (self.square / self.count - mean.square()).clamp_min(0)
        return {"mean": mean.float(), "variance": variance.float(), "count": self.count}


class AdaINToken(nn.Module):
    def __init__(self, source, reference, eps=1e-5, ratio_bounds=(0.25, 4.0)):
        super().__init__()
        source_sd = source["variance"].clamp_min(eps).sqrt()
        reference_sd = reference["variance"].clamp_min(eps).sqrt()
        ratio = (source_sd / reference_sd).clamp(*ratio_bounds)
        self.register_buffer("ratio", ratio)
        self.register_buffer("offset", source["mean"] - reference["mean"] * ratio)

    def forward(self, tokens):
        return tokens * self.ratio + self.offset


class ResidualTokenProjector(nn.Module):
    def __init__(self, channels=110, bottleneck=16, scale=0.1):
        super().__init__()
        self.scale = scale
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Linear(channels, bottleneck)
        self.up = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        return tokens + self.scale * self.up(torch.nn.functional.gelu(self.down(self.norm(tokens))))


def moment_alignment_loss(before, after, source, eps=1e-5, residual_weight=1e-3):
    mean = after.mean((0, 1))
    variance = after.var((0, 1), unbiased=False)
    source_variance = source["variance"].clamp_min(eps)
    location = ((mean - source["mean"]).square() / source_variance).mean()
    scale = (torch.log(variance.clamp_min(eps)) - torch.log(source_variance)).square().mean()
    penalty = ((after - before).square() / source_variance).mean()
    return location + scale + residual_weight * penalty


@contextmanager
def aligned_generator(model, aligner):
    def hook(module, args, output):
        return {**output, "time": aligner(output["time"])}
    handle = model.generator.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
