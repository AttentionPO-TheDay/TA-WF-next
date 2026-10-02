"""Independent two-view generator/head factorial; no archived-code imports.

Inputs preserve packet order and the already audited RF TAM convention. Absolute
timestamps are clipped, never sorted or differenced. Fixed and learned modes
share all deterministic statistics; learned mode adds pre-compression CNN
features. Classifier projections belong to the classifier in both modes.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiViewGenerator(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        if mode not in {"fixed", "learned"}:
            raise ValueError(mode)
        self.mode = mode
        self.packet_conv = nn.ModuleList([nn.Conv1d(3, 16, 5, padding=2),
                                          nn.Conv1d(16, 16, 5, padding=2)])
        self.time_conv = nn.ModuleList([nn.Conv1d(2, 16, 5, padding=2),
                                        nn.Conv1d(16, 16, 5, padding=2)])
        self.act = nn.GELU()
        if mode == "fixed":
            self.requires_grad_(False)

    def generate(self, signed_timestamps: Tensor, tam: Tensor):
        if signed_timestamps.ndim != 2 or signed_timestamps.shape[1] != 5000:
            raise ValueError("signed_timestamps must be B x 5000")
        if tam.shape != (signed_timestamps.shape[0], 2, 1800):
            raise ValueError("tam must be B x 2 x 1800")
        x = signed_timestamps.float()
        counts = tam.float()
        valid = x.ne(0)
        weight = valid.to(x.dtype)
        direction = x.sign()
        time = x.abs().clamp(max=80.0) / 80.0
        b = x.shape[0]
        w = weight.reshape(b, 100, 5, 10)
        d = direction.reshape(b, 100, 5, 10)
        t = time.reshape(b, 100, 5, 10)
        n = w.sum(-1).clamp_min(1)
        adjacent = w[..., :-1] * w[..., 1:]
        switches = (d[..., :-1] != d[..., 1:]).to(x.dtype) * adjacent
        stats = torch.stack([(d * w).sum(-1) / n,
                             w.mean(-1), (t * w).sum(-1) / n,
                             switches.sum(-1) / adjacent.sum(-1).clamp_min(1)], -1)
        fixed_packet = stats.reshape(b, 100, 20)
        log_counts = counts.log1p()
        fixed_time = log_counts.reshape(b, 2, 120, 15).permute(0, 2, 1, 3).reshape(b, 120, 30)
        if self.mode == "learned":
            p = torch.stack([direction, weight, time], 1)
            for layer in self.packet_conv:
                p = self.act(layer(p)) * weight[:, None, :]
            p = p.reshape(b, 16, 100, 5, 10)
            p = (p * w[:, None, ...]).sum(-1) / n[:, None, ...]
            learned_packet = p.permute(0, 2, 3, 1).reshape(b, 100, 80)
            z = log_counts
            for layer in self.time_conv:
                z = self.act(layer(z))
            learned_time = z.reshape(b, 16, 120, 5, 3).mean(-1).permute(0, 2, 3, 1).reshape(b, 120, 80)
        else:
            learned_packet = x.new_zeros(b, 100, 80)
            learned_time = x.new_zeros(b, 120, 80)
        packet = torch.cat([fixed_packet, learned_packet], -1)
        temporal = torch.cat([fixed_time, learned_time], -1)
        packet_valid = valid.reshape(b, 100, 50).any(-1)
        # Empty temporal bins encode absence; they are deliberately visible.
        temporal_valid = torch.ones(b, 120, dtype=torch.bool, device=x.device)
        return {"packet": packet, "time": temporal,
                "packet_mask": packet_valid, "time_mask": temporal_valid}

    def forward(self, signed_timestamps: Tensor, tam: Tensor):
        return self.generate(signed_timestamps, tam)


class TokenMLP(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 512), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(512, 128), nn.Dropout(dropout))
            for _ in range(2)])

    def forward(self, x: Tensor, padding_mask: Tensor):
        for layer in self.layers:
            x = x + layer(x)
            x = x.masked_fill(padding_mask[..., None], 0)
        return x


class MultiViewModel(nn.Module):
    def __init__(self, generator_mode: str, head: str, num_classes: int = 102,
                 dropout: float = 0.1):
        super().__init__()
        if head not in {"mlp", "transformer"}:
            raise ValueError(head)
        self.generator_mode, self.head_name = generator_mode, head
        # Same construction order across all four cells: even fixed mode owns
        # inactive CNNs, so their initialization cannot shift shared weights.
        self.generator = MultiViewGenerator(generator_mode)
        self.packet_projection = nn.Linear(100, 128)
        self.time_projection = nn.Linear(110, 128)
        self.packet_positions = nn.Parameter(torch.empty(100, 128))
        self.time_positions = nn.Parameter(torch.empty(120, 128))
        self.view_types = nn.Parameter(torch.empty(2, 128))
        nn.init.normal_(self.packet_positions, std=0.02)
        nn.init.normal_(self.time_positions, std=0.02)
        nn.init.normal_(self.view_types, std=0.02)
        self.final_norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(256, num_classes)
        # Head initialization leaves the outer CPU RNG unchanged. Shared
        # modules are already built, and have identical weights for a seed.
        with torch.random.fork_rng(devices=[]):
            if head == "mlp":
                self.head = TokenMLP(dropout)
            else:
                self.head = nn.ModuleList([
                    nn.TransformerEncoderLayer(128, 4, 256, dropout=dropout,
                                               activation="gelu", batch_first=True,
                                               norm_first=True) for _ in range(2)])

    def forward_tokens(self, signed_timestamps: Tensor, tam: Tensor):
        generated = self.generator(signed_timestamps, tam)
        packet, temporal = generated["packet"], generated["time"]
        p_valid, t_valid = generated["packet_mask"], generated["time_mask"]
        p = self.packet_projection(packet) + self.packet_positions + self.view_types[0]
        t = self.time_projection(temporal) + self.time_positions + self.view_types[1]
        valid = torch.cat([p_valid, t_valid], 1)
        padding = ~valid
        z = torch.cat([p, t], 1).masked_fill(padding[..., None], 0)
        if self.head_name == "mlp":
            z = self.head(z, padding)
        else:
            for layer in self.head:
                z = layer(z, src_key_padding_mask=padding)
                z = z.masked_fill(padding[..., None], 0)
        z = self.final_norm(z).masked_fill(padding[..., None], 0)
        return z[:, :100], z[:, 100:], p_valid, t_valid

    def forward(self, signed_timestamps: Tensor, tam: Tensor):
        p, t, p_valid, t_valid = self.forward_tokens(signed_timestamps, tam)
        p_mean = (p * p_valid[..., None]).sum(1) / p_valid.sum(1).clamp_min(1)[..., None]
        t_mean = (t * t_valid[..., None]).sum(1) / t_valid.sum(1).clamp_min(1)[..., None]
        return self.classifier(torch.cat([p_mean, t_mean], -1))


def parameter_counts(model: nn.Module):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "generator_total": sum(p.numel() for p in model.generator.parameters()),
            "generator_trainable": sum(p.numel() for p in model.generator.parameters() if p.requires_grad)}
