"""Explicit inference-only extraction from v9 and v10; see SOURCES.md."""
import torch
from torch import nn
from torch.nn.utils import parametrize


class Stage(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv1 = nn.Conv1d(ci, co, 5, padding=2)
        self.conv2 = nn.Conv1d(co, co, 5, padding=2)
        self.skip = nn.Conv1d(ci, co, 1)
        self.act = nn.GELU()
        self.norm1 = nn.BatchNorm1d(co, eps=1e-5, momentum=.1)
        self.norm2 = nn.BatchNorm1d(co, eps=1e-5, momentum=.1)

    def forward(self, x):
        return self.act(self.norm2(self.conv2(self.act(self.norm1(self.conv1(x))))) + self.skip(x))


class WideGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.packet_conv = nn.ModuleList([nn.Conv1d(3, 16, 5, padding=2), nn.Conv1d(16, 16, 5, padding=2)])
        self.stages = nn.ModuleList([Stage(2, 64), Stage(64, 128), Stage(128, 256)])
        self.to_features = nn.Conv1d(256, 80, 1)

    def forward(self, tam):
        b = len(tam)
        counts = tam.float().log1p()
        fixed = counts.reshape(b, 2, 120, 15).permute(0, 2, 1, 3).reshape(b, 120, 30)
        z = counts
        for layer, pool in zip(self.stages, [3, 5, 1]):
            z = layer(z)
            if pool > 1:
                z = z.reshape(b, z.shape[1], z.shape[2] // pool, pool).mean(-1)
        return torch.cat([fixed, self.to_features(z).transpose(1, 2)], -1)


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = WideGenerator()
        self.packet_projection = nn.Linear(100, 128)
        self.time_projection = nn.Linear(110, 128)
        self.packet_positions = nn.Parameter(torch.empty(100, 128))
        self.time_positions = nn.Parameter(torch.empty(120, 128))
        self.view_types = nn.Parameter(torch.empty(2, 128))
        self.final_norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(256, 102)
        self.head = nn.ModuleList([nn.TransformerEncoderLayer(128, 4, 256, dropout=.1, activation='gelu', batch_first=True, norm_first=True) for _ in range(2)])

    def forward(self, tam):
        # Operation order follows v9 methods.encode, including masked readout.
        t = self.generator(tam)
        b = len(t)
        packet = self.packet_projection(t.new_zeros(b, 100, 100)) + self.packet_positions + self.view_types[0]
        temporal = self.time_projection(t) + self.time_positions + self.view_types[1]
        padding = torch.cat([torch.ones(b, 100, dtype=torch.bool, device=t.device), torch.zeros(b, 120, dtype=torch.bool, device=t.device)], 1)
        z = torch.cat([packet, temporal], 1).masked_fill(padding[..., None], 0)
        for layer in self.head:
            z = layer(z, src_key_padding_mask=padding).masked_fill(padding[..., None], 0)
        z = self.final_norm(z).masked_fill(padding[..., None], 0)
        pv, tv = ~padding[:, :100], ~padding[:, 100:]
        p = (z[:, :100] * pv[..., None]).sum(1) / pv.sum(1).clamp_min(1)[..., None]
        e = (z[:, 100:] * tv[..., None]).sum(1) / tv.sum(1).clamp_min(1)[..., None]
        return self.classifier(torch.cat([p, e], -1))


class QVLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.aq = nn.Parameter(torch.randn(4, 128) * .02)
        self.bq = nn.Parameter(torch.zeros(128, 4))
        self.av = nn.Parameter(torch.randn(4, 128) * .02)
        self.bv = nn.Parameter(torch.zeros(128, 4))

    def forward(self, w):
        q, v = self.bq @ self.aq, self.bv @ self.av
        return w + torch.cat([q, torch.zeros_like(q), v], 0)


def attach(model, state):
    modules = []
    for layer in model.head:
        a = QVLoRA().cuda()
        parametrize.register_parametrization(layer.self_attn, 'in_proj_weight', a)
        modules.append(a)
    adapter = nn.ModuleList(modules)
    adapter.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)


class Pool(nn.Module):
    def forward(self, x):
        n = x.shape[-1]
        return torch.stack([x[..., i*n//8:((i+1)*n+7)//8].mean(-1) for i in range(8)], -1)


class ViewNet(nn.Module):
    def __init__(self):
        super().__init__()
        blocks, ci = [], 2
        for co in [32, 64, 96, 128]:
            blocks.extend([nn.Conv1d(ci, co, 7, stride=2, padding=3, bias=False), nn.GroupNorm(8, co), nn.ELU(), nn.Conv1d(co, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.ELU(), nn.MaxPool1d(2, ceil_mode=True)])
            ci = co
        self.encoder = nn.Sequential(*blocks)
        self.pool = Pool()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.2), nn.Linear(1024, 256), nn.ELU(), nn.Dropout(.2), nn.Linear(256, 102))

    def forward(self, x):
        return self.head(self.pool(self.encoder(x)))


def input_features(t):
    t = t.float()
    other = torch.zeros_like(t)
    dt = (t[:, 1:].abs() - t[:, :-1].abs()).clamp_min(0)
    valid = (t[:, 1:] != 0) & (t[:, :-1] != 0)
    other[:, 1:] = ((dt * 1000).log1p() / 10).clamp(0, 1) * valid
    return torch.stack((t.sign(), other), 1)
