"""Detached candidate targets and source-calibrated acceptance; no query fitting."""
import hashlib,time
import numpy as np
import torch
from torch import nn
from runtime import C,entropy
from methods import standardize,encode,attach,streams,relational


def gpu_dict(p):return {k:v.cuda() if torch.is_tensor(v) else v for k,v in p.items()}

def banks(artifact,p):
    x=nn.functional.normalize(standardize(artifact['source_embeddings'].cuda(),p),dim=-1)
    labels=artifact['source_labels'].cuda()
    ids=[int(hashlib.sha256(str(int(row)).encode()).hexdigest(),16)%2 for row in artifact['source_rows']]
    index=torch.tensor(ids,device='cuda')
    return [(x[index==i],labels[index==i]) for i in range(2)]

@torch.no_grad()
def neighbors(e,p,bank_data,temperature):
    x=nn.functional.normalize(standardize(e,p),dim=-1);probs=[];ds=[]
    for bx,by in bank_data:
        ps=[];dd=[]
        for i in range(0,len(x),128):
            distance=(1-x[i:i+128]@bx.T).clamp_min(0)
            d,ix=distance.topk(C['adaptation']['neighbor_count_per_bank'],largest=False)
            weight=(-d/temperature).softmax(-1)
            v=torch.zeros(len(ix),102,device=e.device).scatter_add_(1,by[ix],weight)
            alpha=C['adaptation']['neighbor_smoothing'];ps.append((v+alpha/102)/(1+alpha));dd.append(d[:,0])
        probs.append(torch.cat(ps));ds.append(torch.cat(dd))
    return torch.stack(probs),torch.stack(ds)

@torch.no_grad()
def nearest_distance(e,p):
    x=standardize(e,p);out=[]
    mu=p['centers'][:,0];cov=p['covariances'][:,0]
    for i in range(0,len(x),128):out.append(((x[i:i+128,None]-mu[None]).square()/cov[None]).mean(-1).min(1).values)
    return torch.cat(out)

def target_distribution(teacher_z,votes,target):
    cfg=C['adaptation'];top=teacher_z.topk(cfg['candidate_topk'],dim=-1).indices
    mask=torch.zeros_like(teacher_z,dtype=torch.bool).scatter_(1,top,True)
    if target=='hard':q=nn.functional.one_hot(teacher_z.argmax(-1),102).float()
    else:
        values=teacher_z/cfg['kl_temperature']
        if target=='soft_neighbor':values=values+cfg['neighbor_exponent']*votes.mean(0).clamp_min(1e-12).log()
        else:assert target=='soft_teacher'
        q=values.masked_fill(~mask,-torch.inf).softmax(-1)
    assert torch.isfinite(q).all() and torch.allclose(q.sum(-1),torch.ones(len(q),device=q.device))
    assert (q.masked_select(~mask)==0).all()
    return q.detach()

@torch.no_grad()
def acceptance_inputs(e,base_z,p,bank_data,calibration):
    votes,_=neighbors(e,p,bank_data,calibration['neighbor_temperature'])
    return {'votes':votes,'distance':nearest_distance(e,p),'entropy':entropy(base_z),'top5':base_z.topk(C['adaptation']['candidate_topk'],dim=-1).indices}


def policy_logits(raw_z,base_z,info,calibration,policy):
    bp=base_z.argmax(-1);cp=raw_z.argmax(-1);changed=cp!=bp
    if policy=='raw':return raw_z,torch.ones_like(changed)
    if policy=='valid_entropy':keep=(info['entropy']<=calibration['entropy_cutoff']) & (info['distance']<=calibration['distance_cutoff'])
    else:
        assert policy=='evidence_accept'
        if not calibration['acceptance_enabled']:keep=torch.zeros_like(changed)
        else:
            v=info['votes'];candidate=v.gather(2,cp[None,:,None].expand(2,-1,1)).squeeze(-1)
            baseline=v.gather(2,bp[None,:,None].expand(2,-1,1)).squeeze(-1)
            gain=(candidate.clamp_min(1e-12).log()-baseline.clamp_min(1e-12).log()).min(0).values
            keep=changed & (gain>calibration['acceptance_threshold']) & (info['top5']==cp[:,None]).any(-1) & (info['distance']<=calibration['distance_cutoff'])
    # Unchanged predictions are returned as canonical frozen logits too.
    keep=keep & changed
    return torch.where(keep[:,None],raw_z,base_z),keep


def train_candidate(m,t,teacher_e,teacher_z,votes,source_t,source_e,p,condition):
    target,kind=condition.rsplit('_',1);cfg=C['adaptation'];adapter=attach(m,kind)
    q=target_distribution(teacher_z,votes,target);labels=q.argmax(-1)
    anchors=q@p['centers'][:,0]
    optimizer=torch.optim.Adam(adapter.parameters(),lr=cfg['lr'])
    source_stream=streams(len(source_t),cfg['steps'],cfg['source_batch_size']);history=[];start=time.monotonic()
    for step,ix in enumerate(streams(len(t),cfg['steps'],cfg['batch_size'])):
        optimizer.zero_grad(set_to_none=True);e,z=encode(m,t[ix],adapter);x=standardize(e,p)
        temp=cfg['kl_temperature'];kl=nn.functional.kl_div((z/temp).log_softmax(-1),q[ix],reduction='batchmean')*temp**2
        cost=(x[:,None]-p['centers'][:,0][None]).square().mean(-1)
        pull=(cost*q[ix]).sum(-1).mean()
        relation,n=relational(x,anchors[ix],torch.ones(len(ix),device='cuda'),labels[ix])
        si=next(source_stream);se,_=encode(m,source_t[si],adapter)
        identity=(standardize(se,p)-standardize(source_e[si],p)).square().mean()
        loss=cfg['kl_weight']*kl+cfg['pull_weight']*pull+cfg['topology_weight']*relation+cfg['source_identity_weight']*identity
        assert torch.isfinite(loss)
        loss.backward();g=nn.utils.clip_grad_norm_(adapter.parameters(),cfg['grad_clip'],error_if_nonfinite=True);optimizer.step()
        history.append({'step':step+1,'loss':float(loss.detach()),'kl':float(kl.detach()),'pull':float(pull.detach()),'relation':float(relation.detach()),'identity':float(identity.detach()),'classes':n,'gradient_norm':float(g)})
        if time.monotonic()-start>C['budget']['max_cell_seconds']:raise TimeoutError('candidate training budget')
    detail={'target':target,'teacher_disagreement_fraction':float((q.argmax(-1)!=teacher_z.argmax(-1)).float().mean()),'target_entropy_mean':float(-(q*q.clamp_min(1e-12).log()).sum(-1).mean())}
    return adapter,history,detail
