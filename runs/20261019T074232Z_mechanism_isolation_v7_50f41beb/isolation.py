"""Reference-only BN calibration, prototype write isolation, layer/gradient probes."""
import time
import numpy as np
import torch
from torch import nn
from runtime import C,entropy,load_model
from methods import attach,encode,standardize,streams,relational
from candidate import target_distribution
from refinement import RefinedState


def copy_affines(m,state):
    for name,mod in m.named_modules():
        if isinstance(mod,nn.modules.batchnorm._BatchNorm):
            with torch.no_grad():
                mod.weight.copy_(state[name+'.weight']);mod.bias.copy_(state[name+'.bias'])


@torch.no_grad()
def calibrate(m,reference):
    assert set(reference)=={'timestamps','tam','rows'}
    modules=dict(m.named_modules());notes=[]
    for name,mod in modules.items():
        if not isinstance(mod,nn.modules.batchnorm._BatchNorm):continue
        sums=[];squares=[];counts=[]
        def collect(module,args):
            x=args[0].detach().double();axes=(0,)+tuple(range(2,x.ndim))
            sums.append(x.sum(axes));squares.append(x.square().sum(axes));counts.append(x.numel()//x.shape[1])
        handle=mod.register_forward_pre_hook(collect)
        try:
            for i in range(0,len(reference['rows']),128):
                m(reference['timestamps'][i:i+128].cuda(),reference['tam'][i:i+128].cuda())
        finally:handle.remove()
        n=sum(counts);mu=torch.stack(sums).sum(0)/n;var=(torch.stack(squares).sum(0)/n-mu.square()).clamp_min(0)
        mod.running_mean.copy_(mu.float());mod.running_var.copy_(var.float());mod.num_batches_tracked.zero_()
        assert torch.isfinite(var).all();notes.append(dict(layer=name,count=n,mean_norm=float(mu.norm()),variance_mean=float(var.mean())))
    assert len(notes)==6
    return notes


@torch.no_grad()
def reliability(z,votes,valid_logits):
    p=z.softmax(-1);b=votes.mean(0);mid=(p+b)/2
    js=.5*((p*(p.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)+(b*(b.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1))
    agree=(votes[0].argmax(-1)==z.argmax(-1))&(votes[1].argmax(-1)==z.argmax(-1))
    cutoff=torch.quantile(entropy(valid_logits),.75)
    r=agree*torch.sigmoid((cutoff-entropy(z))/.1)*torch.sigmoid((.05-js)/.01)
    assert not r.requires_grad and torch.isfinite(r).all() and (r>=0).all() and (r<=1).all()
    return r,dict(entropy_cutoff=float(cutoff),mean=float(r.mean()),nonzero=int((r>0).sum()),agreement=int(agree.sum()),js_mean=float(js.mean()))


def loss_parts(m,a,t,q,w,source_t,source_e,p):
    e,z=encode(m,t,a);x=standardize(e,p);cfg=C['adaptation'];temp=cfg['kl_temperature'];proto=p['centers'][:,0]
    kl=nn.functional.kl_div((z/temp).log_softmax(-1),q,reduction='none').sum(-1)*temp**2
    cost=(x[:,None]-proto[None]).square().mean(-1);pull=(cost*q).sum(-1)
    rel,n=relational(x,q@proto,w,q.argmax(-1));se,_=encode(m,source_t,a)
    identity=(standardize(se,p)-standardize(source_e,p)).square().mean()
    parts=dict(kl=(w*kl).mean(),pull=(w*pull).mean(),relation=rel*w.mean(),identity=identity)
    loss=cfg['kl_weight']*parts['kl']+cfg['pull_weight']*parts['pull']+cfg['topology_weight']*parts['relation']+cfg['source_identity_weight']*parts['identity']
    return loss,x,z,parts,n


def gradient_probe(parts,a):
    weights=dict(kl=1.,pull=.1,relation=1.,identity=1.);grads={};out={}
    params=list(a.parameters())
    for key,v in parts.items():
        gs=torch.autograd.grad(weights[key]*v,params,retain_graph=True,allow_unused=True)
        grads[key]=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,gs)])
        out[key+'_norm']=float(grads[key].norm())
    for i,k in enumerate(grads):
        for l in list(grads)[i+1:]:
            den=grads[k].norm()*grads[l].norm();out[k+'_'+l+'_cosine']=float(grads[k]@grads[l]/den) if den>1e-15 else None
    return out


def train(m,t,e,teacher_z,votes,source_t,source_e,p,condition,r):
    cfg=C['adaptation'];a=attach(m,'lora');state=RefinedState(e,teacher_z,p,'sharp_target_only')
    qfixed=target_distribution(teacher_z,votes,'soft_neighbor');opt=torch.optim.Adam(a.parameters(),lr=cfg['lr'])
    src=streams(len(source_t),cfg['steps'],cfg['source_batch_size']);history=[];gp=[];start=time.monotonic()
    for step,ix in enumerate(streams(len(t),cfg['steps'],cfg['batch_size'])):
        opt.zero_grad(set_to_none=True);si=next(src);q=state.target(qfixed,ix);w,wd=state.weight(ix,True)
        loss,x,z,parts,n=loss_parts(m,a,t[ix],q,w,source_t[si],source_e[si],p)
        if step+1 in C['v7']['gradient_probes']:gp.append(dict(step=step+1,**gradient_probe(parts,a)))
        assert torch.isfinite(loss);loss.backward();grad=nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            raw=r[ix]
            if condition=='memory_mass_control':writer=torch.ones_like(raw)*(w*raw).sum()/w.sum().clamp_min(1e-12)
            elif condition in ['memory_route','stats_memory_route']:writer=raw
            else:writer=torch.ones_like(raw)
            # Features/probabilities remain readable for ALL reference rows; writer only affects centroid mass.
            state.update(ix,x,z,q,w*writer,True)
            drift=(state.anchored()-state.source).norm(dim=-1)
            hist=dict(step=step+1,loss=float(loss),gradient_norm=float(grad),classes=n,
                prototype_drift_mean=float(drift.mean()),writer_mean=float(writer.mean()),writer_weighted_mass=float((w*writer).sum()),
                routed_reference_mass=float((w*raw).sum()),writer_nonzero=int((writer>0).sum()),
                target_entropy=float(-(q*q.clamp_min(1e-12).log()).sum(-1).mean()),
                target_teacher_disagreement=float((q.argmax(-1)!=teacher_z[ix].argmax(-1)).float().mean()),
                student_entropy=float(entropy(z).mean()),**wd,**{k:float(v) for k,v in parts.items()})
            if condition=='memory_mass_control':assert abs(hist['writer_weighted_mass']-hist['routed_reference_mass'])<1e-4
            history.append(hist)
        if time.monotonic()-start>C['budget']['max_cell_seconds']:raise TimeoutError('cell')
    return a,state,history,gp


@torch.no_grad()
def capture(m,data):
    results={name:[] for name in C['v7']['probe_layers']};handles=[];mods=dict(m.named_modules())
    for name in results:
        def hook(module,args,out,name=name):
            x=out.detach()
            if name.startswith('generator.stages'):v=torch.cat([x.mean(-1),x.std(-1,unbiased=False)],-1)
            else:
                if name.startswith('head.'):x=x[:,100:]
                v=torch.cat([x.mean(1),x.std(1,unbiased=False)],-1)
            results[name].append(v.cpu())
        handles.append(mods[name].register_forward_hook(hook))
    try:
        for i in range(0,len(data['rows']),128):m(data['timestamps'][i:i+128].cuda(),data['tam'][i:i+128].cuda())
    finally:
        for h in handles:h.remove()
    return {k:torch.cat(v).numpy() for k,v in results.items()}


def probe_summary(source,reference,baseline_reference):
    out={}
    for name,s in source.items():
        r=reference[name];b=baseline_reference[name];sd=np.maximum(s.std(0),1e-4)
        w2=np.mean(((s.mean(0)-r.mean(0))/sd)**2+((s.std(0)-r.std(0))/sd)**2)
        x=r-r.mean(0);y=b-b.mean(0);den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
        out[name]=dict(channel_gaussian_w2=float(w2),paired_reference_cka=float(np.linalg.norm(x.T@y)**2/den) if den>0 else None)
    return out
