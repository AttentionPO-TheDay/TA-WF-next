"""Retained KL, reference-only auxiliary view and explicitly rate-limited weights."""
import math,time
import numpy as np
import torch
from torch import nn
from runtime import C
from methods import attach,streams
from refinement import RefinedState
from candidate import target_distribution
from isolation import loss_parts,gradient_probe


def grads(loss,params):
    vs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,vs)])


def schedule(kn,gn,step,ema,ell,smooth):
    ema=[kn,gn] if ema is None else [.9*ema[0]+.1*kn,.9*ema[1]+.1*gn];ratio=max(1.,min(50.,ema[0]/(ema[1]+1e-8))) if gn>1e-10 else 1.;ramp=math.sin(math.pi*min(step/100,1)/2)**2
    if smooth:
        requested=.9*ell+.1*math.log(ratio);cap=.02 if step<=200 else .005;ell+=max(-cap,min(cap,requested-ell));effective=math.exp(ell)
    else:effective=ratio
    return 1+ramp*(effective-1),ema,ell,ratio


def train(m,t,e,z,votes,st,se,p,viewq,enabled,condition):
    cfg=C['adaptation'];a=attach(m,'lora');state=RefinedState(e,z,p,'sharp_target_only');qfixed=target_distribution(z,votes,'soft_neighbor');params=list(a.parameters());opt=torch.optim.Adam(params,lr=cfg['lr']);ss=streams(len(st),400,32);hist=[];probes=[];ema=None;ell=0.;previous_grad=None;previous_update=None;start=time.monotonic();rng=np.random.default_rng(20261019)
    conf=(1+(viewq*viewq.clamp_min(1e-12).log()).sum(-1)/math.log(102)).clamp(0,1).detach()
    for n,ix in enumerate(streams(len(t),400,64),1):
        opt.zero_grad(set_to_none=True);si=next(ss);q=state.target(qfixed,ix);w,wd=state.weight(ix,True);_,x,zz,parts,_=loss_parts(m,a,t[ix],q,w,st[si],se[si],p);geo=.1*parts['pull']+parts['relation'];gk=grads(parts['kl'],params);gg=grads(geo,params);kn=float(gk.norm());gn=float(gg.norm());weight,ema,ell,ratio=schedule(kn,gn,n,ema,ell,condition!='ramp_balance')
        vq=viewq[ix];vc=conf[ix];original_mass=(w*vc).sum();aux_enabled=enabled and condition in ['smooth_view','smooth_shuffled_view']
        if condition=='smooth_shuffled_view':
            perm=torch.as_tensor(rng.permutation(len(ix)),device='cuda');vq=vq[perm];vc=vc[perm];vc=vc*original_mass/(w*vc).sum().clamp_min(1e-12)
        temp=2.;per=nn.functional.kl_div((zz/temp).log_softmax(-1),vq,reduction='none').sum(-1)*temp**2;aux=.25*(w*vc*per).mean() if aux_enabled else zz.sum()*0
        loss=parts['kl']+weight*geo+parts['identity']+aux
        ga=grads(aux,params) if aux_enabled else torch.zeros_like(gk)
        if n in C['v9']['gradient_probes']:probes.append(dict(step=n,**gradient_probe(parts,a),geometry_weight=weight,aux_gradient_norm=float(ga.norm())))
        loss.backward();g=torch.cat([p.grad.flatten() for p in params]);grad_norm=float(g.norm());gc=float(g@previous_grad/(g.norm()*previous_grad.norm())) if previous_grad is not None and float(g.norm()*previous_grad.norm())>1e-15 else None;previous_grad=g.detach().clone();before=torch.cat([p.detach().flatten() for p in params]);nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();after=torch.cat([p.detach().flatten() for p in params]);delta=after-before;uc=float(delta@previous_update/(delta.norm()*previous_update.norm())) if previous_update is not None and float(delta.norm()*previous_update.norm())>1e-15 else None;previous_update=delta.detach().clone();state.observe(ix,x,zz,q,w)
        with torch.no_grad():
            mass=float((w*vc).sum());orig=float(original_mass)
            if condition=='smooth_shuffled_view':assert abs(mass-orig)<1e-4
            hist.append(dict(step=n,loss=float(loss),gradient_norm=grad_norm,geometry_weight=weight,raw_ratio=ratio,ell=ell,ema_kl=ema[0],ema_geometry=ema[1],kl_gradient_norm=kn,geometry_gradient_norm=gn,kl_geometry_cosine=float(gk@gg/(gk.norm()*gg.norm())) if kn*gn>1e-15 else None,aux_gradient_norm=float(ga.norm()),kl_aux_cosine=float(gk@ga/(gk.norm()*ga.norm())) if float(gk.norm()*ga.norm())>1e-15 else None,gradient_previous_cosine=gc,update_previous_cosine=uc,update_norm=float(delta.norm()),aux=float(aux),aux_enabled=bool(aux_enabled),aux_weighted_mass=mass,unshuffled_aux_mass=orig,**{k:float(v) for k,v in parts.items()},**wd))
        if time.monotonic()-start>C['budget']['max_cell_seconds']:raise TimeoutError('cell')
    return a,state,hist,probes
