"""Wide+BN model for single-network symmetric view-consistency training."""
import torch
from torch import nn
from framework_base import Student, LocalStage, ProgressiveGenerator

CONDITIONS = ("ce_only", "dual_mask_ce", "mask_peer", "dual_cross_ce", "cross_peer")


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


class MultiViewModel(Student):
    def __init__(self, condition, num_classes=102, dropout=.1):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        super().__init__('progressive_transformer', num_classes, dropout)
        self.condition = condition
        with torch.random.fork_rng(devices=[]):
            self.generator = WideGenerator(self.generator)
            self.generator.stages = nn.ModuleList([NormStage(s) for s in self.generator.stages])


def span_mask(tam, generator):
    active = torch.rand(len(tam), generator=generator) < .5
    start = torch.randint(0, 1711, (len(tam),), generator=generator)
    bins = torch.arange(1800)[None, :]
    mask = active[:, None] & (bins >= start[:, None]) & (bins < start[:, None] + 90)
    return tam.masked_fill(mask.to(tam.device)[:, None, :], 0)


def learning_rate(step, total=12800):
    return .001 if step <= 6400 else (.0003 if step <= 9600 else .0001)


def consistency_weight(step, max_weight=.25, ramp_steps=3200):
    return max_weight * min(step / ramp_steps, 1.0)


def symmetric_peer_loss(la, lb, temperature):
    pa = torch.softmax(la.detach() / temperature, dim=1)
    pb = torch.softmax(lb.detach() / temperature, dim=1)
    one = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(la / temperature, dim=1), pb, reduction='batchmean')
    two = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(lb / temperature, dim=1), pa, reduction='batchmean')
    return 0.5 * (one + two) * (temperature * temperature)


def make_optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0001,
                             betas=(.9, .999), eps=1e-8)
