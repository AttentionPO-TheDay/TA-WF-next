"""Matched reference entropy updates, with explicit query statistics policies."""
import time
import torch
from runtime import C,configure,entropy,load_model
from methods import streams


def train_tent(m,reference):
    assert set(reference)=={'timestamps','tam','rows'}
    configure(m,'tent_bn');opt=torch.optim.Adam([v for v in m.parameters() if v.requires_grad],lr=C['adaptation']['lr']);history=[];start=time.monotonic()
    for step,ix in enumerate(streams(len(reference['rows']),C['adaptation']['steps'],C['adaptation']['batch_size'])):
        idx=ix.cpu();opt.zero_grad(set_to_none=True)
        z=m(reference['timestamps'][idx].cuda(),reference['tam'][idx].cuda());loss=entropy(z).mean()
        assert torch.isfinite(loss);loss.backward();opt.step()
        history.append({'step':step+1,'loss':float(loss.detach())})
        if time.monotonic()-start>C['budget']['max_cell_seconds']:raise TimeoutError('tent training budget')
    return history


def reference_statistics(m,reference,seed):
    # Copy only updated BN affines into a fresh model. No query in this API.
    out=load_model(seed);previous=dict(m.named_parameters())
    names=[]
    for name,mod in out.named_modules():
        if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):
            names.append(name)
            with torch.no_grad():
                mod.weight.copy_(previous[name+'.weight']);mod.bias.copy_(previous[name+'.bias'])
    modules=dict(out.named_modules());notes=[]
    # In source architecture named BN order equals actual six-stage forward order.
    # Calibrate each layer using upstream already-frozen reference population stats.
    for name in names:
        mod=modules[name];sums=[];squares=[];counts=[]
        def collect(module,args):
            x=args[0].detach().double();axes=(0,)+tuple(range(2,x.ndim))
            sums.append(x.sum(axes));squares.append(x.square().sum(axes));counts.append(x.numel()//x.shape[1])
        h=mod.register_forward_pre_hook(collect)
        with torch.no_grad():
            for i in range(0,len(reference['rows']),128):out(reference['timestamps'][i:i+128].cuda(),reference['tam'][i:i+128].cuda())
        h.remove();n=sum(counts);mean=torch.stack(sums).sum(0)/n;var=(torch.stack(squares).sum(0)/n-mean.square()).clamp_min(0)
        with torch.no_grad():mod.running_mean.copy_(mean.float());mod.running_var.copy_(var.float());mod.num_batches_tracked.zero_()
        notes.append({'layer':name,'activation_count_per_channel':n,'finite':bool(torch.isfinite(var).all())})
    return out,notes


def check_tent_scope(m,original,mode):
    affine={n for n,p in m.named_parameters() if n.endswith(('.weight','.bias')) and any(n.startswith(k+'.') for k,v in m.named_modules() if isinstance(v,torch.nn.modules.batchnorm._BatchNorm))}
    for name,v in m.state_dict().items():
        assert torch.isfinite(v).all(),name
        if name in affine:continue
        if mode=='tent_bn_reference' and name.endswith(('running_mean','running_var','num_batches_tracked')):continue
        assert torch.equal(v.cpu(),original[name]),name
    return sorted(affine)
