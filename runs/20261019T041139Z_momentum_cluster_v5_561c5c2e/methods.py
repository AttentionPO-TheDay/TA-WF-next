"""Source-only Gaussian anchors, exact inference routing, matched adapters."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from token_alignment import ResidualTokenProjector
from runtime import C,entropy

class QVLoRA(nn.Module):
    def __init__(self,width=128,rank=4):
        super().__init__()
        self.aq=nn.Parameter(torch.randn(rank,width)*.02)
        self.bq=nn.Parameter(torch.zeros(width,rank))
        self.av=nn.Parameter(torch.randn(rank,width)*.02)
        self.bv=nn.Parameter(torch.zeros(width,rank))
    def forward(self,w):
        q=self.bq@self.aq;v=self.bv@self.av
        return w+C['adaptation']['lora_scale']*torch.cat([q,torch.zeros_like(q),v],0)

def attach(m,kind):
    m.eval().requires_grad_(False)
    torch.manual_seed(20261019)
    if kind=='residual':return ResidualTokenProjector(110,16,.1).cuda()
    assert kind=='lora'
    blocks=[]
    for layer in m.head:
        a=QVLoRA(rank=C['adaptation']['lora_rank']).cuda()
        parametrize.register_parametrization(layer.self_attn,'in_proj_weight',a)
        blocks.append(a)
    return nn.ModuleList(blocks)

def encode(m,t,adapter=None):
    if isinstance(adapter,ResidualTokenProjector):t=adapter(t)
    b=len(t)
    packet=m.packet_projection(t.new_zeros(b,100,100))+m.packet_positions+m.view_types[0]
    temporal=m.time_projection(t)+m.time_positions+m.view_types[1]
    padding=torch.cat([torch.ones(b,100,dtype=torch.bool,device=t.device),torch.zeros(b,120,dtype=torch.bool,device=t.device)],1)
    z=torch.cat([packet,temporal],1).masked_fill(padding[...,None],0)
    for layer in m.head:z=layer(z,src_key_padding_mask=padding).masked_fill(padding[...,None],0)
    z=m.final_norm(z).masked_fill(padding[...,None],0)
    pv=~padding[:,:100];tv=~padding[:,100:]
    p=(z[:,:100]*pv[...,None]).sum(1)/pv.sum(1).clamp_min(1)[...,None]
    e=(z[:,100:]*tv[...,None]).sum(1)/tv.sum(1).clamp_min(1)[...,None]
    return e,m.classifier(torch.cat([p,e],-1))

@torch.no_grad()
def tokenize(m,d):
    return torch.cat([m.generator(d['timestamps'][i:i+128].cuda(),d['tam'][i:i+128].cuda())['time'] for i in range(0,len(d['rows']),128)])

@torch.no_grad()
def evaluate(m,t,adapter=None):
    es=[];zs=[]
    for i in range(0,len(t),128):
        e,z=encode(m,t[i:i+128],adapter);es.append(e);zs.append(z)
    return torch.cat(es),torch.cat(zs)

def standardize(e,p):return (e-p['mean'])/p['sd']

def distances(x,p):
    # B,C,K,D intermediate is bounded by chunks, no query-derived statistics.
    dif=(x[:,None,None,:]-p['centers'][None]).square()
    w2=(dif+p['covariances'][None]).mean(-1)
    maha=(dif/p['covariances'][None]).mean(-1)
    return w2,maha

def route(e,z,p):
    x=standardize(e,p);ws=[];ds=[]
    for i in range(0,len(x),128):
        _,d=distances(x[i:i+128],p);ds.append(d.flatten(1).min(1).values)
    d=torch.cat(ds);h=entropy(z)
    keep=(h<=p['entropy_cutoff']) & (d<=p['distance_cutoff'])
    w=torch.exp(-d/p['distance_scale'])*keep
    return w.detach(),keep,d.detach(),h.detach()

def assignments(e,z,p):
    x=standardize(e,p);label=z.argmax(-1)
    mu=p['centers'][label];var=p['covariances'][label]
    w2=((x[:,None]-mu).square()+var).mean(-1)
    q=(p['priors'][label].clamp_min(1e-12).log()-w2/p['temperature']).softmax(-1)
    anchor=(q[...,None]*mu).sum(1)
    return q.detach(),anchor.detach(),label

def relational(x,a,w,labels):
    centers=[];anchors=[]
    for label in labels.unique():
        sel=(labels==label)&(w>0)
        if int(sel.sum())<2:continue
        v=w[sel];den=v.sum().clamp_min(1e-12)
        centers.append((x[sel]*v[:,None]).sum(0)/den)
        anchors.append((a[sel]*v[:,None]).sum(0)/den)
    if len(centers)<2:return x.sum()*0,0
    u=nn.functional.normalize(torch.stack(centers),dim=-1)
    v=nn.functional.normalize(torch.stack(anchors),dim=-1)
    mask=~torch.eye(len(u),device=u.device,dtype=torch.bool)
    return (u@u.T-v@v.T)[mask].square().mean(),len(centers)

def streams(n,steps,batch):
    rng=np.random.default_rng(20261019);order=np.empty(0,dtype=np.int64);off=0
    for _ in range(steps):
        if off+batch>len(order):order=rng.permutation(n);off=0
        yield torch.as_tensor(order[off:off+batch],device='cuda');off+=batch

def assert_base_locked(m,initial):
    for name,v in m.state_dict().items():
        if '.parametrizations.in_proj_weight.0.' in name:continue
        canonical=name.replace('.parametrizations.in_proj_weight.original','.in_proj_weight')
        assert canonical in initial,name
        assert torch.equal(v.detach().cpu(),initial[canonical]),name
        assert torch.isfinite(v).all(),name
