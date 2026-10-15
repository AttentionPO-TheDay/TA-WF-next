"""Target-only full-parameter conditional entropy; loss mask, tracked BN."""
import torch
from torch import nn

def configure_model(model,momentum=0.10):
    model.train()
    model.requires_grad_(True)
    for m in model.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            assert m.running_mean is not None and m.running_var is not None
            m.track_running_stats=True
            m.momentum=float(momentum)
            m.train()
    return model

def collect_params(model):
    items=list(model.named_parameters())
    assert items and all(p.requires_grad for _,p in items)
    return [p for _,p in items],[n for n,_ in items]

def forward_and_adapt(x,model,optimizer,tau_low=.20,tau_high=.40):
    optimizer.zero_grad(set_to_none=True)
    outputs=model(x)
    assert torch.isfinite(outputs).all(),'nonfinite logits'
    ent=-(outputs.softmax(1)*outputs.log_softmax(1)).sum(1)
    assert torch.isfinite(ent).all(),'nonfinite entropy'
    keep=(ent>=tau_low)&(ent<=tau_high)
    if bool(keep.any()):
        loss=ent[keep].mean()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(),'nonfinite gradient'
        optimizer.step()
    return outputs,ent.detach(),keep.detach()
