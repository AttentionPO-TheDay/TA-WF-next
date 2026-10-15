"""Wide+BN model with a fixed relative-time structure residual."""
import torch
from torch import nn
from framework_base import Student, LocalStage, ProgressiveGenerator
from candidate_base import span_mask

def learning_rate(step, total=12800):
    return .001 if step <= 6400 else (.0003 if step <= 9600 else .0001)

CONDITIONS = ("ce_control", "relative_time")

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
        self.stages = nn.ModuleList([LocalStage(2,64), LocalStage(64,128), LocalStage(128,256)])
        self.to_features = nn.Conv1d(256,80,1)

class MultiViewModel(Student):
    def __init__(self, condition, num_classes=102, dropout=.1):
        if condition not in CONDITIONS: raise ValueError(condition)
        super().__init__('progressive_transformer', num_classes, dropout)
        self.condition = condition
        with torch.random.fork_rng(devices=[]):
            self.generator = WideGenerator(self.generator)
            self.generator.stages = nn.ModuleList([NormStage(s) for s in self.generator.stages])
        self.relative_projection = nn.Linear(4, 128)
        self.relative_projection_norm = nn.LayerNorm(128)

    @staticmethod
    def relative_time_features(timestamps):
        b, n = timestamps.shape
        if n != 5000: raise ValueError('timestamps must be B x 5000')
        device = timestamps.device
        pos = torch.arange(n, device=device)
        bins = (pos * 120 // n).long()
        lengths = torch.bincount(bins, minlength=120).to(timestamps.dtype)
        valid = timestamps.ne(0)
        w = valid.to(timestamps.dtype)
        signed = timestamps.float().clamp(-80,80) / 80.0
        absolute = timestamps.float().abs().clamp(max=80).log1p() / torch.log(torch.tensor(81.0, device=device))
        def aggregate(value):
            out = torch.zeros(b, 120, device=device, dtype=torch.float32)
            return out.scatter_add(1, bins[None,:].expand(b,-1), value.float())
        count = aggregate(w)
        signed_mean = aggregate(signed*w) / count.clamp_min(1.0)
        abs_mean = aggregate(absolute*w) / count.clamp_min(1.0)
        fraction = count / lengths[None,:].clamp_min(1.0)
        pair_valid = (valid[:,:-1] & valid[:,1:]).to(torch.float32)
        pair_switch = (timestamps[:,:-1].sign() != timestamps[:,1:].sign()).to(torch.float32) * pair_valid
        pair_bins = bins[:-1]
        transitions = torch.zeros(b, 120, device=device, dtype=torch.float32).scatter_add(1, pair_bins[None,:].expand(b,-1), pair_switch)
        transition_rate = transitions / torch.zeros(b,120,device=device).scatter_add(1, pair_bins[None,:].expand(b,-1), pair_valid).clamp_min(1.0)
        return torch.stack([signed_mean, fraction, abs_mean, transition_rate], -1)

    def forward(self, signed_timestamps, tam):
        generated = self.generator(signed_timestamps, tam)
        packet, temporal = generated['packet'], generated['time']
        p_valid, t_valid = generated['packet_mask'], generated['time_mask']
        p = self.packet_projection(packet) + self.packet_positions + self.view_types[0]
        t = self.time_projection(temporal) + self.time_positions + self.view_types[1]
        if self.condition == 'relative_time':
            rel = self.relative_time_features(signed_timestamps)
        else:
            rel = torch.zeros(signed_timestamps.shape[0], 120, 4, device=signed_timestamps.device)
        t = t + self.relative_projection_norm(self.relative_projection(rel))
        valid = torch.cat([p_valid, t_valid], 1)
        padding = ~valid
        z = torch.cat([p,t],1).masked_fill(padding[...,None],0)
        for layer in self.head:
            z = layer(z, src_key_padding_mask=padding)
            z = z.masked_fill(padding[...,None],0)
        z = self.final_norm(z).masked_fill(padding[...,None],0)
        p,t = z[:,:100], z[:,100:]
        p_mean=(p*p_valid[...,None]).sum(1)/p_valid.sum(1).clamp_min(1)[...,None]
        t_mean=(t*t_valid[...,None]).sum(1)/t_valid.sum(1).clamp_min(1)[...,None]
        return self.classifier(torch.cat([p_mean,t_mean],-1))
