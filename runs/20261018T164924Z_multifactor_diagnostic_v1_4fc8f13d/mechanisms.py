"""Prototype anchored objectives. All teacher grouping uses unlabeled reference only."""
import time
import numpy as np
import torch
from torch import nn
from runtime import C,configure,entropy
from token_alignment import Moments,AdaINToken,ResidualTokenProjector,moment_alignment_loss,aligned_generator

def from_tokens(m,t):
    # WideGenerator has 100 permanently invalid packet slots and 120 valid time slots.
    b=len(t);packet=m.packet_projection(t.new_zeros(b,100,100))+m.packet_positions+m.view_types[0]
    temporal=m.time_projection(t)+m.time_positions+m.view_types[1]
    padding=torch.cat([torch.ones(b,100,dtype=torch.bool,device=t.device),torch.zeros(b,120,dtype=torch.bool,device=t.device)],1)
    z=torch.cat([packet,temporal],1).masked_fill(padding[...,None],0)
    for layer in m.head:z=layer(z,src_key_padding_mask=padding).masked_fill(padding[...,None],0)
    z=m.final_norm(z).masked_fill(padding[...,None],0)
    pv=~padding[:,:100];tv=~padding[:,100:]
    p=(z[:,:100]*pv[...,None]).sum(1)/pv.sum(1).clamp_min(1)[...,None]
    temporal=(z[:,100:]*tv[...,None]).sum(1)/tv.sum(1).clamp_min(1)[...,None]
    return m.classifier(torch.cat([p,temporal],-1))

@torch.no_grad()
def get_tokens(m,data):
    return torch.cat([m.generator(data['timestamps'][i:i+128].cuda(),data['tam'][i:i+128].cuda())['time'].detach()
                      for i in range(0,len(data['rows']),128)])

def indices(n,steps):
    rng=np.random.default_rng(20261019);order=np.empty(0,dtype=np.int64);offset=0
    for _ in range(steps):
        if offset+128>len(order):order=rng.permutation(n);offset=0
        yield torch.as_tensor(order[offset:offset+128],device='cuda');offset+=128

def topology(centers,anchors):
    a=nn.functional.normalize(centers,dim=-1);b=nn.functional.normalize(anchors,dim=-1)
    mask=~torch.eye(len(a),device=a.device,dtype=torch.bool)
    return ((a@a.T-b@b.T)[mask]).square().mean() if len(a)>1 else a.sum()*0

def anchor_losses(after,before,teacher,ref,source_tokens,aligner):
    cfg=C['adaptation'];mu=ref['moments']['mean'].cuda();sd=ref['moments']['variance'].cuda().clamp_min(1e-5).sqrt()
    prototypes=(ref['class_prototypes'].cuda()-mu)/sd
    h=entropy(teacher);prob,labels=teacher.softmax(1).max(1)
    reliable=(h<=cfg['prototype_teacher_entropy_max']) & (prob>=cfg['prototype_teacher_probability_min'])
    pooled=(after.mean(1)-mu)/sd
    pull=pooled.sum()*0; centers=[];source_centers=[]
    if reliable.any():
        pull=(pooled[reliable]-prototypes[labels[reliable]]).square().mean()
        for k in labels[reliable].unique():
            selected=reliable & (labels==k)
            if int(selected.sum())>=cfg['prototype_class_min_count']:
                centers.append(pooled[selected].mean(0));source_centers.append(prototypes[k])
    relation=topology(torch.stack(centers),torch.stack(source_centers)) if len(centers)>1 else pooled.sum()*0
    source_identity=((aligner(source_tokens)-source_tokens)/sd).square().mean()
    residual=((after-before)/sd).square().mean()
    return pull,relation,source_identity,residual,int(reliable.sum()),len(centers)

def adapt(m,support,condition,ref,source_anchors,steps=None):
    assert set(support)=={'timestamps','tam','rows'}
    cfg=C['adaptation'];steps=cfg['steps'] if steps is None else steps
    configure(m,condition); initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    trainable=[n for n,p in m.named_parameters() if p.requires_grad]
    history=[];aligner=None;start=time.monotonic()
    if condition=='frozen':return None,history,trainable
    if condition in ('tent_bn','tent_entropy_gate'):
        optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=cfg['lr'])
        for step,idx in enumerate(indices(len(support['rows']),steps)):
            m.eval();optimizer.zero_grad(set_to_none=True)
            z=m(support['timestamps'][idx.cpu()].cuda(),support['tam'][idx.cpu()].cuda());h=entropy(z)
            selected=(h>=cfg['gate'][0]) & (h<=cfg['gate'][1]) if condition=='tent_entropy_gate' else torch.ones_like(h,dtype=torch.bool)
            loss=h[selected].mean() if selected.any() else h.sum()*0
            assert torch.isfinite(loss)
            if selected.any():loss.backward();optimizer.step()
            history.append({'step':step+1,'loss':float(loss.detach()),'retained':int(selected.sum())})
            if time.monotonic()-start>C['budget']['max_cell_seconds']:raise TimeoutError(condition)
    else:
        m.eval().requires_grad_(False);tokens=get_tokens(m,support)
        if condition=='token_adain':
            moments=Moments();moments.add(tokens)
            aligner=AdaINToken(ref['moments'],moments.finish()).cuda()
        else:
            torch.manual_seed(20261019)
            aligner=ResidualTokenProjector(110,cfg['bottleneck'],cfg['scale']).cuda()
            optimizer=torch.optim.Adam(aligner.parameters(),lr=cfg['lr'])
            teacher=torch.cat([from_tokens(m,tokens[i:i+128]).detach() for i in range(0,len(tokens),128)]) if condition.startswith('prototype_') else None
            source_tokens=get_tokens(m,source_anchors) if condition.startswith('prototype_') else None
            source_iter=iter(indices(len(source_tokens),steps)) if source_tokens is not None else None
            stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in ref['moments'].items()}
            for step,idx in enumerate(indices(len(tokens),steps)):
                before=tokens[idx];optimizer.zero_grad(set_to_none=True);after=aligner(before)
                details={}
                if condition=='token_projector':loss=moment_alignment_loss(before,after,stats)
                else:
                    pull,relation,identity,residual,n,classes=anchor_losses(after,before,teacher[idx],ref,source_tokens[next(source_iter)],aligner)
                    h=entropy(from_tokens(m,after));keep=(h>=cfg['gate'][0]) & (h<=cfg['gate'][1])
                    ent=h[keep].mean() if keep.any() else h.sum()*0
                    weight=cfg['topology_weight'] if condition=='prototype_anchor' else 0.
                    loss=cfg['entropy_weight']*ent+cfg['centroid_weight']*pull+weight*relation+cfg['source_identity_weight']*identity+cfg['residual_weight']*residual
                    details={'entropy':float(ent.detach()),'centroid':float(pull.detach()),'topology':float(relation.detach()),'source_identity':float(identity.detach()),'reliable':n,'topology_classes':classes}
                assert torch.isfinite(loss)
                loss.backward()
                if condition.startswith('prototype_'):nn.utils.clip_grad_norm_(aligner.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
                optimizer.step()
                history.append({'step':step+1,'loss':float(loss.detach()),**details})
                if time.monotonic()-start>C['budget']['max_cell_seconds']:raise TimeoutError(condition)
    for k,v in m.state_dict().items():
        if k not in trainable:assert torch.equal(v.detach().cpu(),initial[k]),k
        assert torch.isfinite(v).all(),k
    if aligner is not None:assert all(torch.isfinite(v).all() for v in aligner.state_dict().values())
    return aligner,history,trainable

def restore_aligner(state,condition):
    if state is None:return None
    if condition=='token_adain':
        dummy={'mean':torch.zeros(110),'variance':torch.ones(110)}
        a=AdaINToken(dummy,dummy)
    else:a=ResidualTokenProjector(110,C['adaptation']['bottleneck'],C['adaptation']['scale'])
    a.load_state_dict(state,strict=True);return a.cuda().eval()
