"""Wide+BN Transformer with a source-only global TAM statistics branch."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from framework_base import Student, LocalStage, ProgressiveGenerator

CONDITIONS = ("zero_add", "stats_add", "zero_gate", "stats_gate")
SEGMENTS = (1, 3, 6, 12)


class NormStage(nn.Module):
    def __init__(self, previous):
        super().__init__()
        self.conv1, self.conv2, self.skip, self.act = previous.conv1, previous.conv2, previous.skip, previous.act
        channels = self.conv2.out_channels
        self.norm1 = nn.BatchNorm1d(channels, eps=1e-5, momentum=.1, affine=True, track_running_stats=True)
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-5, momentum=.1, affine=True, track_running_stats=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(x)) + residual)


class WideGenerator(ProgressiveGenerator):
    def __init__(self, previous):
        nn.Module.__init__(self)
        self.packet_conv = previous.packet_conv
        self.stages = nn.ModuleList([LocalStage(2, 64), LocalStage(64, 128), LocalStage(128, 256)])
        self.to_features = nn.Conv1d(256, 80, 1)


def global_stats(tam):
    """Return 180 fixed, label-free statistics from log-count TAM bins."""
    x = tam.float().log1p()
    b = x.shape[0]
    values = []
    for n in SEGMENTS:
        z = x.reshape(b, 2, n, 1800 // n)
        values += [z.mean(-1), z.std(-1, unbiased=False), z.amax(-1), (z > 0).float().mean(-1)]
    all_x = x.reshape(b, -1)
    values += [all_x.mean(-1, keepdim=True), all_x.std(-1, unbiased=False, keepdim=True),
               all_x.amax(-1, keepdim=True), (all_x > 0).float().mean(-1, keepdim=True)]
    return torch.cat([v.reshape(b, -1) for v in values], -1)


class StatsModel(Student):
    def __init__(self, condition, stats_mean, stats_std, num_classes=102, dropout=.1):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        super().__init__('progressive_transformer', num_classes, dropout)
        self.condition = condition
        with torch.random.fork_rng(devices=[]):
            self.generator = WideGenerator(self.generator)
            self.generator.stages = nn.ModuleList([NormStage(s) for s in self.generator.stages])
            self.stats_projection = nn.Sequential(nn.Linear(180, 64), nn.GELU(), nn.Linear(64, 128))
            self.stats_norm = nn.LayerNorm(128)
            self.stats_head = nn.Linear(128, num_classes)
            self.gate = nn.Linear(384, num_classes)
        nn.init.zeros_(self.stats_head.weight)
        nn.init.zeros_(self.stats_head.bias)
        self.register_buffer('stats_mean', stats_mean.detach().float().reshape(1, 180))
        self.register_buffer('stats_std', stats_std.detach().float().reshape(1, 180))
        if condition.startswith('zero_'):
            for p in self.stats_projection.parameters(): p.requires_grad = True

    def representation(self, signed_timestamps, tam):
        p, t, pv, tv = self.forward_tokens(signed_timestamps, tam)
        pm = (p * pv[..., None]).sum(1) / pv.sum(1).clamp_min(1)[..., None]
        tm = (t * tv[..., None]).sum(1) / tv.sum(1).clamp_min(1)[..., None]
        return torch.cat([pm, tm], -1)

    def forward(self, signed_timestamps, tam, return_features=False):
        rep = self.representation(signed_timestamps, tam)
        base_logits = self.classifier(rep)
        stats = global_stats(tam)
        if self.condition.startswith('zero_'):
            stats = torch.zeros_like(stats)
        stats_z = (stats - self.stats_mean) / self.stats_std.clamp_min(1e-6)
        stats_repr = self.stats_norm(self.stats_projection(stats_z))
        stats_logits = self.stats_head(stats_repr)
        if self.condition.endswith('_gate'):
            logits = base_logits + torch.sigmoid(self.gate(torch.cat([rep, stats_repr], -1))) * stats_logits
        else:
            logits = base_logits + stats_logits
        return (logits, rep) if return_features else logits


def span_mask(tam, generator):
    active = torch.rand(len(tam), generator=generator) < .5
    start = torch.randint(0, 1711, (len(tam),), generator=generator)
    bins = torch.arange(1800)[None, :]
    mask = active[:, None] & (bins >= start[:, None]) & (bins < start[:, None] + 90)
    return tam.masked_fill(mask.to(tam.device)[:, None, :], 0)


def learning_rate(step, total=12800):
    return .001 if step <= 6400 else (.0003 if step <= 9600 else .0001)


def make_optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8)
