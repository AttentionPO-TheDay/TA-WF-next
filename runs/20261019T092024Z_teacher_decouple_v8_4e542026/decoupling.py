"""Direct KL ablations only. Frozen-source BN and all other v6 state transitions."""
import time
import torch
from torch import nn
from runtime import C,entropy
from methods import attach,streams
from refinement import RefinedState
from candidate import target_distribution
from isolation import loss_parts,gradient_probe


def grads(loss,params):
    g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,g)])


def assemble(parts,per_kl,w,gate,condition,params,ema):
    geo=.1*parts['pull']+parts['relation'];kl=parts['kl'];weight=1.;diag={}
    if condition=='no_teacher_kl':target_kl=kl*0
    elif condition=='confident_teacher_kl':target_kl=(per_kl*w*gate).mean()
    else:target_kl=kl
    if condition=='geometry_grad_balance':
        gk=grads(kl,params);gg=grads(geo,params);kn=float(gk.norm());gn=float(gg.norm())
        if ema is None:ema=[kn,gn]
        else:ema=[.9*ema[0]+.1*kn,.9*ema[1]+.1*gn]
        weight=max(1.,min(50.,ema[0]/(ema[1]+1e-8))) if gn>1e-10 else 1.
        diag=dict(kl_gradient_norm=kn,geometry_gradient_norm=gn,kl_geometry_cosine=float(gk@gg/(gk.norm()*gg.norm())) if kn*gn>1e-15 else None,balance_ema_kl=ema[0],balance_ema_geometry=ema[1])
    assert condition in ['full_kl_control']+C['conditions']
    return target_kl+weight*geo+parts['identity'],ema,dict(geometry_weight=weight,active_kl=float(target_kl.detach()),**diag)


def train(m,t,e,z,votes,source_t,source_e,p,gate,condition):
    cfg=C['adaptation'];a=attach(m,'lora');state=RefinedState(e,z,p,'sharp_target_only');qfixed=target_distribution(z,votes,'soft_neighbor');params=list(a.parameters());opt=torch.optim.Adam(params,lr=cfg['lr']);src=streams(len(source_t),cfg['steps'],cfg['source_batch_size']);hist=[];probes=[];ema=None;begin=time.monotonic()
    for step,ix in enumerate(streams(len(t),cfg['steps'],cfg['batch_size'])):
        opt.zero_grad(set_to_none=True);si=next(src);q=state.target(qfixed,ix);w,wd=state.weight(ix,True)
        legacy,x,zz,parts,n=loss_parts(m,a,t[ix],q,w,source_t[si],source_e[si],p)
        temp=cfg['kl_temperature'];per_kl=nn.functional.kl_div((zz/temp).log_softmax(-1),q,reduction='none').sum(-1)*temp**2
        loss,ema,detail=assemble(parts,per_kl,w,gate[ix],condition,params,ema)
        if step+1 in C['v8']['gradient_probes']:probes.append(dict(step=step+1,**gradient_probe(parts,a),geometry_weight=detail['geometry_weight']))
        assert torch.isfinite(loss) and not q.requires_grad;loss.backward();norm=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();state.observe(ix,x,zz,q,w)
        with torch.no_grad():
            hist.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),**{k:float(v.detach()) for k,v in parts.items()},**wd,**detail,kl_gate_fraction=float(gate[ix].mean()),prototype_drift_mean=float((state.anchored()-state.source).norm(dim=-1).mean()),student_entropy=float(entropy(zz).mean())))
        if time.monotonic()-begin>C['budget']['max_cell_seconds']:raise TimeoutError('adaptation cell')
    return a,state,hist,probes
