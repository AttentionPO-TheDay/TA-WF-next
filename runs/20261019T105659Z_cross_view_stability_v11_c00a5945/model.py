import torch
from torch import nn
from runtime import C

class DeterministicPool(nn.Module):
    """AdaptiveAvgPool1d(8) bins, using deterministic slice means for CUDA backward."""
    def forward(self,x):
        n=x.shape[-1]
        return torch.stack([x[...,i*n//8:((i+1)*n+7)//8].mean(-1) for i in range(8)],-1)

def features(d,arm):
    if arm=='tam_cnn':return d['tam'].float().log1p()
    t=d['timestamps'].float();sgn=t.sign()
    other=torch.zeros_like(t)
    if arm=='direction_iat':
        dt=(t[:,1:].abs()-t[:,:-1].abs()).clamp_min(0)
        valid=(t[:,1:]!=0)&(t[:,:-1]!=0)
        other[:,1:]=((dt*1000).log1p()/10).clamp(0,1)*valid
    else:assert arm=='direction_order'
    return torch.stack((sgn,other),1)

class ViewNet(nn.Module):
    def __init__(self):
        super().__init__();blocks=[];ci=2
        for co in C['architecture']['channels']:
            blocks.extend([nn.Conv1d(ci,co,7,stride=2,padding=3,bias=False),nn.GroupNorm(8,co),nn.ELU(),nn.Conv1d(co,co,3,padding=1,bias=False),nn.GroupNorm(8,co),nn.ELU(),nn.MaxPool1d(2,ceil_mode=True)])
            ci=co
        self.encoder=nn.Sequential(*blocks);self.pool=DeterministicPool()
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(ci*8,256),nn.ELU(),nn.Dropout(.2),nn.Linear(256,102))
    def forward(self,x):return self.head(self.pool(self.encoder(x)))

@torch.no_grad()
def predict(m,x):
    m.eval();z=[]
    for b in x.split(C['training']['eval_batch_size']):z.append(m(b.cuda()).cpu())
    z=torch.cat(z);assert torch.isfinite(z).all() and z.shape==(len(x),102)
    return z
