"""Fixed-source-statistics TENT with a small Transformer LayerNorm scope."""
from copy import deepcopy
import torch
from torch import nn
import torch.jit

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, tau_low=0.20, tau_high=0.40):
    outputs = model(x)
    ent = softmax_entropy(outputs)
    keep = (ent >= tau_low) & (ent <= tau_high)
    optimizer.zero_grad(set_to_none=True)
    if bool(keep.any()):
        ent[keep].mean().backward()
        optimizer.step()
    return outputs, ent.detach(), keep.detach()

def collect_params(model):
    params=[]; names=[]
    for nm,m in model.named_modules():
        if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)) or (isinstance(m,nn.LayerNorm) and (nm.endswith('norm1') or nm.endswith('norm2') or nm.endswith('final_norm'))):
            for pn,p in m.named_parameters(recurse=False):
                if pn in ('weight','bias'):
                    params.append(p); names.append(f'{nm}.{pn}')
    return params,names

def configure_model(model):
    model.train(); model.requires_grad_(False)
    for nm,m in model.named_modules():
        if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats=True
            m.eval()
        elif isinstance(m,nn.LayerNorm) and (nm.endswith('norm1') or nm.endswith('norm2') or nm.endswith('final_norm')):
            m.requires_grad_(True)
    return model
