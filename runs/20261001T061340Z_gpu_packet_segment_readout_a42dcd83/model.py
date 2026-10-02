"""Run-local direction-only progressive CNN models, implemented independently.

No legacy model import or pretrained weights. A matched generator and readout
compare attention against pointwise residual MLP token processing.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MaskedConvStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, 7, padding=3),
            nn.Conv1d(out_channels, out_channels, 7, padding=3),
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(2)])

    def forward(self, x, observed):
        for conv, norm in zip(self.convs, self.norms):
            x = x.masked_fill(~observed[:, None, :], 0)
            x = conv(x)
            x = F.gelu(norm(x.transpose(1, 2))).transpose(1, 2)
            x = x.masked_fill(~observed[:, None, :], 0)
        pooled_mask = F.max_pool1d(observed[:, None, :].to(x.dtype), 4, 4, ceil_mode=True).bool()[:, 0]
        x = F.max_pool1d(x.masked_fill(~observed[:, None, :], -torch.inf), 4, 4, ceil_mode=True)
        x = x.masked_fill(~pooled_mask[:, None, :], 0)
        return x, pooled_mask


class PacketGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([MaskedConvStage(1, 32), MaskedConvStage(32, 64), MaskedConvStage(64, 128)])

    def forward(self, directions, observed):
        x = directions.masked_fill(~observed, 0).unsqueeze(1)
        for stage in self.stages:
            x, observed = stage(x, observed)
        return x.transpose(1, 2), observed


class TokenMLPBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.ff = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.ff(self.norm(x))


class PacketModel(nn.Module):
    def __init__(self, head='transformer', num_classes=102, dropout=0.1, pooling='segments'):
        super().__init__()
        assert pooling in ['segments','global_repeat']
        self.pooling=pooling
        if head not in ('transformer', 'mlp'):
            raise ValueError('head must be transformer or mlp')
        self.head_type = head
        # Construct ALL shared parameters before the condition-specific head.
        # Resetting torch.manual_seed(seed) before either constructor therefore
        # yields exactly matching generator, positional, norm and readout states.
        self.generator = PacketGenerator()
        self.position = nn.Parameter(torch.empty(1, 79, 128))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.final_norm = nn.LayerNorm(128)
        self.readout = nn.Linear(128, num_classes)
        # Head construction does not alter the caller's random stream either.
        with torch.random.fork_rng(devices=[]):
            if head == 'transformer':
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(128, 4, 256, dropout=dropout,
                        activation='gelu', batch_first=True, norm_first=True)
                    for _ in range(2)
                ])
            else:
                self.blocks = nn.ModuleList([TokenMLPBlock(dropout) for _ in range(2)])

        # Expand only after original shared components were initialized.
        original_readout=self.readout
        with torch.random.fork_rng(devices=[]):
            self.readout=nn.Linear(512,num_classes)
        with torch.no_grad():
            self.readout.weight.copy_(original_readout.weight.repeat(1,4)/4)
            self.readout.bias.copy_(original_readout.bias)

    def forward(self, directions, observed=None):
        if directions.ndim != 2 or directions.shape[1] != 5000:
            raise ValueError('directions must have shape [batch, 5000]')
        if observed is None:
            observed = directions != 0
        if observed.shape != directions.shape:
            raise ValueError('observed must match directions shape')
        observed = observed.to(device=directions.device, dtype=torch.bool)
        directions = directions.to(dtype=self.position.dtype)
        x, mask = self.generator(directions, observed)
        x = (x + self.position).masked_fill(~mask[:, :, None], 0)
        # PyTorch attention needs at least one unmasked key. This sentinel has
        # no effect on the output because the original mask governs readout.
        key_mask = mask.clone()
        key_mask[~key_mask.any(dim=1), 0] = True
        for block in self.blocks:
            if self.head_type == 'transformer':
                x = block(x, src_key_padding_mask=~key_mask)
            else:
                x = block(x)
            x = x.masked_fill(~mask[:, :, None], 0)
        x = self.final_norm(x).masked_fill(~mask[:, :, None], 0)
        pooled = pool_tokens(x,mask,self.pooling)
        return self.readout(pooled)

def pool_tokens(x,mask,mode):
    x=x.masked_fill(~mask[:,:,None],0)
    if mode=='global_repeat':
        return (x.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).repeat(1,4)
    # Relative quartiles of the observed token sequence, no boundary changes upstream.
    rank=mask.long().cumsum(1)-1
    bins=(rank*4//mask.sum(1,keepdim=True).clamp_min(1)).clamp(0,3)
    result=[]
    for j in range(4):
        selected=mask&(bins==j)
        result.append((x*selected[:,:,None]).sum(1)/selected.sum(1,keepdim=True).clamp_min(1))
    return torch.cat(result,dim=1)
