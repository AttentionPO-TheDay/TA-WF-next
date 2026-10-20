"""Reference-only momentum states, detached dynamic targets and batch soft contrast."""
import time
import numpy as np
import torch
from torch import nn
from runtime import C,entropy
from methods import attach,encode,standardize,streams,relational
from candidate import target_distribution


def soft_contrast(x,q,w):
    # No self-positive, true labels or discrete pseudo-label assignments.
    n=len(x)
    if n<2:return x.sum()*0,{'pair_mass':0.,'positive_pair_ess':0.}
    u=nn.functional.normalize(x,dim=-1);diag=torch.eye(n,device=x.device,dtype=torch.bool)
    pair=(q.detach()@q.detach().T).masked_fill(diag,0)
    target=pair/pair.sum(-1,keepdim=True).clamp_min(1e-12)
    logp=(u@u.T/C['momentum_v5']['contrast_temperature']).masked_fill(diag,-torch.inf).log_softmax(-1)
    per=-(target*logp.masked_fill(diag,0)).sum(-1)
    loss=(w*per).mean()
    with torch.no_grad():
        ess=target.sum(-1).square()/target.square().sum(-1).clamp_min(1e-12)
        return loss,{'pair_mass':float(pair.sum()),'positive_pair_ess':float(ess.mean())}

class MomentumState:
    def __init__(self,e,z,p):
        self.source=p['centers'][:,0].detach().clone();self.prototype=self.source.clone()
        self.features=standardize(e,p).detach().clone();self.probabilities=z.softmax(-1).detach().clone()
        h=-(self.probabilities*self.probabilities.clamp_min(1e-12).log()).sum(-1)
        self.mean=h.mean();self.second=h.square().mean();self.steps=0
    def anchored(self):
        a=C['momentum_v5']['prototype_source_anchor'];return a*self.source+(1-a)*self.prototype
    @torch.no_grad()
    def target(self,qfixed,ix,dynamic):
        if not dynamic:return qfixed[ix].detach()
        cfg=C['momentum_v5'];u=nn.functional.normalize(self.features[ix],dim=-1);v=nn.functional.normalize(self.anchored(),dim=-1)
        qgeo=(u@v.T/cfg['geometry_temperature']).softmax(-1)
        return ((1-cfg['prototype_target_mix'])*qfixed[ix]+cfg['prototype_target_mix']*qgeo).detach()
    @torch.no_grad()
    def weight(self,ix,enabled):
        cfg=C['momentum_v5'];prob=self.probabilities[ix];h=-(prob*prob.clamp_min(1e-12).log()).sum(-1)
        rho=cfg['batch_momentum'];self.mean=rho*self.mean+(1-rho)*h.mean();self.second=rho*self.second+(1-rho)*h.square().mean()
        variance=(self.second-self.mean.square()).clamp_min(cfg['entropy_variance_floor'])
        w=cfg['weight_floor']+(1-cfg['weight_floor'])*torch.exp(-.5*(h-self.mean).square()/variance) if enabled else torch.ones_like(h)
        assert not w.requires_grad and torch.isfinite(w).all() and (w>=cfg['weight_floor']).all()
        return w,{'entropy_mean_ema':float(self.mean),'entropy_variance_ema':float(variance),'weight_mean':float(w.mean()),'weight_min':float(w.min()),'weight_max':float(w.max()),'effective_samples':float(w.sum().square()/w.square().sum())}
    @torch.no_grad()
    def update(self,ix,x,z,q,w,dynamic):
        cfg=C['momentum_v5'];f=cfg['feature_memory_momentum'];b=cfg['sample_probability_momentum']
        self.features[ix]=f*self.features[ix]+(1-f)*x.detach();self.probabilities[ix]=b*self.probabilities[ix]+(1-b)*z.detach().softmax(-1)
        if dynamic:
            mass=(q*w[:,None]).sum(0);observed=(q*w[:,None]).T@self.features[ix]/mass[:,None].clamp_min(1e-12)
            alpha=(1-cfg['prototype_momentum'])*mass/(mass+1)
            self.prototype+=alpha[:,None]*(observed-self.prototype)
        self.steps+=1
    def dump(self):
        return {k:getattr(self,k).detach().cpu() for k in ['source','prototype','features','probabilities','mean','second']}|{'steps':self.steps}


def objective(m,a,t,q,w,prototypes,source_t,source_e,p,contrast):
    e,z=encode(m,t,a);x=standardize(e,p);cfg=C['adaptation'];temp=cfg['kl_temperature'];anchors=q@prototypes
    kl=nn.functional.kl_div((z/temp).log_softmax(-1),q,reduction='none').sum(-1)*temp**2
    cost=(x[:,None]-prototypes[None]).square().mean(-1);pull=(cost*q).sum(-1)
    relation,n=relational(x,anchors,w,q.argmax(-1));se,_=encode(m,source_t,a);identity=(standardize(se,p)-standardize(source_e,p)).square().mean()
    cl,cd=soft_contrast(x,q,w) if contrast else (x.sum()*0,{'pair_mass':0.,'positive_pair_ess':0.})
    kl=(w*kl).mean();pull=(w*pull).mean();relation=relation*w.mean()
    loss=cfg['kl_weight']*kl+cfg['pull_weight']*pull+cfg['topology_weight']*relation+cfg['source_identity_weight']*identity+C['momentum_v5']['contrast_weight']*cl
    detail={'kl':float(kl.detach()),'pull':float(pull.detach()),'relation':float(relation.detach()),'identity':float(identity.detach()),'contrast':float(cl.detach()),'classes':n,**cd}
    return loss,x,z,detail


def train(m,t,e,z,votes,source_t,source_e,p,condition):
    cfg=C['adaptation'];a=attach(m,'lora');state=MomentumState(e,z,p);qfixed=target_distribution(z,votes,'soft_neighbor');opt=torch.optim.Adam(a.parameters(),lr=cfg['lr'])
    dynamic=condition in ['momentum_proto','momentum_joint'];contrast=condition in ['momentum_contrast','momentum_joint'];weighted=condition!='full_static'
    src=streams(len(source_t),cfg['steps'],cfg['source_batch_size']);history=[];begin=time.monotonic()
    for step,ix in enumerate(streams(len(t),cfg['steps'],cfg['batch_size'])):
        opt.zero_grad(set_to_none=True);si=next(src);q=state.target(qfixed,ix,dynamic);w,wd=state.weight(ix,weighted);prototypes=state.anchored() if dynamic else state.source
        loss,x,zz,detail=objective(m,a,t[ix],q,w,prototypes,source_t[si],source_e[si],p,contrast)
        assert not q.requires_grad and not prototypes.requires_grad and torch.isfinite(loss)
        loss.backward();grad=nn.utils.clip_grad_norm_(a.parameters(),cfg['grad_clip'],error_if_nonfinite=True);opt.step()
        state.update(ix,x,zz,q,w,dynamic)
        with torch.no_grad():
            drift=(state.anchored()-state.source).norm(dim=-1);qent=-(q*q.clamp_min(1e-12).log()).sum(-1)
            history.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(grad),'prototype_drift_mean':float(drift.mean()),'prototype_drift_max':float(drift.max()),'target_entropy':float(qent.mean()),'target_teacher_disagreement':float((q.argmax(-1)!=z[ix].argmax(-1)).float().mean()),'student_entropy':float(entropy(zz).mean()),**wd,**detail})
        if time.monotonic()-begin>C['budget']['max_cell_seconds']:raise TimeoutError('adaptation cell')
    assert state.steps==cfg['steps'];return a,state,history
