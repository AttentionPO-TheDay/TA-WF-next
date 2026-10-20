"""Frozen routing/evidence; labels are absent from optimization and score APIs."""
import time
import numpy as np
import torch
from torch import nn
from runtime import C,entropy
from methods import standardize,encode,attach,streams,relational
from candidate import nearest_distance,target_distribution

@torch.no_grad()
def geometry(e,p):
    x=nn.functional.normalize(standardize(e,p),dim=-1)
    centers=nn.functional.normalize(p['centers'][:,0],dim=-1)
    d=(1-x@centers.T).clamp_min(0);v,ix=d.topk(2,largest=False)
    return {'cosine':d,'pmr':(v[:,1]-v[:,0])/(v[:,1]+1e-8),'nearest':ix[:,0],'maha':nearest_distance(e,p)}

@torch.no_grad()
def calibration(g):
    return {'pmr25':float(torch.quantile(g['pmr'],.25)),'distance_scale':max(float(g['maha'].median()),1e-6),'distance_cutoff':float(torch.quantile(g['maha'],.99))}

@torch.no_grad()
def weights(h,g,cal,condition):
    if condition=='none':return torch.ones_like(h),{}
    lo,hi=torch.quantile(h,torch.tensor(C['routing_v4']['entropy_quantiles'],device=h.device))
    s=torch.maximum(.1*(hi-lo),h.new_tensor(.02))
    soft=torch.sigmoid((h-lo)/s)*torch.sigmoid((hi-h)/s)
    if condition=='hard_band':w=((h>lo)&(h<=hi)).float()
    elif condition=='soft_band':w=soft
    else:
        assert condition in ('soft_geometry','shuffle_soft_geometry','shuffle_soft_band')
        w=soft if condition.endswith('soft_band') else soft*torch.sigmoid((g['pmr']-cal['pmr25'])/.05)*torch.exp(-g['maha']/cal['distance_scale'])*(g['maha']<=cal['distance_cutoff'])
    assert not w.requires_grad and torch.isfinite(w).all() and (w>=0).all() and (w<=1).all()
    return w.detach(),{'entropy_low':float(lo),'entropy_high':float(hi),'soft_temperature':float(s),'weight_mean':float(w.mean()),'weight_zero_fraction':float((w==0).float().mean()),'effective_samples':float(w.sum().square()/w.square().sum().clamp_min(1e-12)),'samples':len(w)}

def objective(m,adapter,t,q,anchors,labels,w,source_t,source_e,p):
    e,z=encode(m,t,adapter);x=standardize(e,p);temp=C['adaptation']['kl_temperature']
    kl=nn.functional.kl_div((z/temp).log_softmax(-1),q,reduction='none').sum(-1)*temp**2
    cost=(x[:,None]-p['centers'][:,0][None]).square().mean(-1)
    pull=(cost*q).sum(-1)
    relation,n=relational(x,anchors,w,labels)
    se,_=encode(m,source_t,adapter);identity=(standardize(se,p)-standardize(source_e,p)).square().mean()
    kl=(kl*w).mean();pull=(pull*w).mean();relation=relation*w.mean()
    loss=C['adaptation']['kl_weight']*kl+C['adaptation']['pull_weight']*pull+C['adaptation']['topology_weight']*relation+C['adaptation']['source_identity_weight']*identity
    return loss,{'kl':float(kl.detach()),'pull':float(pull.detach()),'relation':float(relation.detach()),'identity':float(identity.detach()),'classes':n}

def train(m,t,teacher_z,votes,source_t,source_e,p,w,condition):
    cfg=C['adaptation'];a=attach(m,'lora');q=target_distribution(teacher_z,votes,'soft_neighbor');labels=q.argmax(-1);anchors=q@p['centers'][:,0]
    opt=torch.optim.Adam(a.parameters(),lr=cfg['lr']);source_stream=streams(len(source_t),cfg['steps'],cfg['source_batch_size']);hist=[];begin=time.monotonic()
    rng=np.random.default_rng(99117)
    for step,ix in enumerate(streams(len(t),cfg['steps'],cfg['batch_size'])):
        opt.zero_grad(set_to_none=True);si=next(source_stream);ww=w[ix]
        if condition.startswith('shuffle_'):ww=ww[torch.as_tensor(rng.permutation(len(ww)),device='cuda')]
        loss,detail=objective(m,a,t[ix],q[ix],anchors[ix],labels[ix],ww,source_t[si],source_e[si],p)
        assert torch.isfinite(loss);loss.backward();grad=nn.utils.clip_grad_norm_(a.parameters(),cfg['grad_clip'],error_if_nonfinite=True);opt.step()
        hist.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(grad),'weight_mean':float(ww.mean()),'effective_samples':float(ww.sum().square()/ww.square().sum().clamp_min(1e-12)),**detail})
        if time.monotonic()-begin>C['budget']['max_cell_seconds']:raise TimeoutError('training cell budget')
    return a,hist

def normalized_margin(z,cp):
    z=z/z.square().mean(-1,keepdim=True).add(1e-12).sqrt()
    v=z.gather(1,cp[:,None]).squeeze(1);other=z.clone().scatter_(1,cp[:,None],-torch.inf).max(-1).values
    return v-other

@torch.no_grad()
def scores(base_z,z,g,cal):
    cp=z.argmax(-1);d=g['cosine'];dc=d.gather(1,cp[:,None]).squeeze(1)
    other=d.clone().scatter_(1,cp[:,None],torch.inf).min(-1).values
    margin=(other-dc)/(other+1e-8);gain=normalized_margin(z,cp)-normalized_margin(base_z,cp)
    combined=torch.sigmoid(margin/.05)*torch.sigmoid(gain/.05)*torch.exp(-g['maha']/cal['distance_scale'])
    combined=torch.where(g['maha']<=cal['distance_cutoff'],combined,combined.new_full(combined.shape,-1.))
    return {'combined':combined,'negative_frozen_entropy':-entropy(base_z),'candidate_geometric_margin':margin,'negative_frozen_maha':-g['maha'],'normalized_margin_gain':gain}

def select_threshold(y,bp,cp,score,budget):
    # Source valid labels are explicitly authorized ONLY here and in terminal metrics.
    correct=bp==y;wrong=~correct;trials=[];feasible=[]
    for th in C['acceptance_v4']['source_grid']:
        keep=(cp!=bp)&(score>=th);damage=int((keep&correct).sum());recover=int((keep&wrong&(cp==y)).sum())
        row={'threshold':th,'changes':int(keep.sum()),'damaged':damage,'recovered':recover,'nar':damage/max(int(correct.sum()),1)}
        row['feasible']=row['changes']>=10 and row['nar']<=budget and recover>damage
        trials.append(row)
        if row['feasible']:feasible.append(row)
    best=max(feasible,key=lambda r:(r['recovered'],-r['damaged'],r['threshold'])) if feasible else None
    return {'enabled':best is not None,'threshold':best['threshold'] if best else 1.01,'budget':budget,'trials':trials}
