import torch
from model import features
from runtime import C

def tam(t):
    nz=t!=0;bins=(t.abs().double()*1799/80).floor().long().clamp_max(1799)+(t<0).long()*1800
    bins=bins+torch.arange(len(t))[:,None]*3600
    return torch.bincount(bins[nz],minlength=len(t)*3600).reshape(len(t),2,1800).float()

def transform(d,view):
    if view=='original':return d
    t=d['timestamps'].clone();valid=t!=0;idx=torch.arange(t.shape[1])[None,:].expand(len(t),-1)
    if view in ['time095','time105']:t=t*(.95 if view=='time095' else 1.05)
    elif view=='thin01':
        h=(d['rows'][:,None]*1000003+idx*9176+1103515245)%2147483647;h=((h^(h>>13))*1274126177)%2147483647
        keep=valid&((h%100!=0)|(valid.long().cumsum(-1)==1));order=torch.argsort(torch.where(keep,idx,idx+t.shape[1]),dim=-1);n=keep.sum(-1);t=t.gather(1,order)*(idx<n[:,None])
    elif view=='tail98':
        count=(valid.sum(-1)*98+99)//100;t=t*(valid.long().cumsum(-1)<=count[:,None])
    else:raise ValueError(view)
    return dict(timestamps=t,tam=tam(t),rows=d['rows'])

@torch.no_grad()
def predict(m,d,arm,view):
    out=[];m.eval()
    for start in range(0,len(d['rows']),C['training']['eval_batch_size']):
        sl=slice(start,start+C['training']['eval_batch_size']);b={k:d[k][sl] for k in ['timestamps','tam','rows']};x=features(transform(b,view),arm);out.append(m(x.cuda()).cpu())
    z=torch.cat(out);assert torch.isfinite(z).all() and z.shape==(len(d['rows']),102);return z.numpy()
