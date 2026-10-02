"""Single-factor candidates built on explicit run-local TAM multiscale code."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from encoder_base import MultiViewModel as Baseline, TAMGenerator

KINDS=('baseline','scale_concat','attention_readout','relative_time','span_mask',
       'supcon','cosine_lr','cpu_baseline','cpu_temporal_cnn')

class ScaleConcatGenerator(TAMGenerator):
    def __init__(self, previous):
        super().__init__(previous,'multiscale_d139')
        with torch.random.fork_rng(devices=[]):
            self.scale_projection=nn.Conv1d(48,16,1)
            nn.init.zeros_(self.scale_projection.weight)
            nn.init.zeros_(self.scale_projection.bias)
            with torch.no_grad():
                for i in range(3):
                    self.scale_projection.weight[:,i*16:(i+1)*16,0].copy_(torch.eye(16)/3)

    def forward(self,signed_timestamps,tam):
        b=len(tam)
        if signed_timestamps.shape!=(b,5000) or tam.shape!=(b,2,1800):
            raise ValueError('input shape')
        counts=tam.float().log1p()
        fixed=counts.reshape(b,2,120,15).permute(0,2,1,3).reshape(b,120,30)
        z=self.scale_projection(torch.cat([self.encode_scale(counts,d) for d in self.dilations],1))
        learned=z.reshape(b,16,120,5,3).mean(-1).permute(0,2,3,1).reshape(b,120,80)
        return {'packet':tam.new_zeros(b,100,100),'time':torch.cat([fixed,learned],-1),
                'packet_mask':torch.zeros(b,100,dtype=torch.bool,device=tam.device),
                'time_mask':torch.ones(b,120,dtype=torch.bool,device=tam.device)}

class TemporalBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.norm=nn.LayerNorm(128)
        self.conv1=nn.Conv1d(128,128,3,padding=1)
        self.conv2=nn.Conv1d(128,128,3,padding=1)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        y=self.norm(x).transpose(1,2)
        y=self.dropout(F.gelu(self.conv1(y)))
        return x+self.dropout(self.conv2(y)).transpose(1,2)

class MultiViewModel(Baseline):
    def __init__(self,condition,num_classes=102,dropout=.1):
        if condition not in KINDS:raise ValueError(condition)
        super().__init__('multiscale_d139',num_classes,dropout)
        self.condition=condition
        with torch.random.fork_rng(devices=[]):
            if condition=='scale_concat':self.generator=ScaleConcatGenerator(self.generator)
            if condition=='attention_readout':self.readout_query=nn.Parameter(torch.zeros(128))
            if condition=='cpu_temporal_cnn':
                self.head=nn.ModuleList([TemporalBlock(dropout) for _ in range(2)])

    def forward_tokens(self,signed_timestamps,tam):
        if self.condition!='cpu_temporal_cnn':return super().forward_tokens(signed_timestamps,tam)
        generated=self.generator(signed_timestamps,tam)
        t=self.time_projection(generated['time'])+self.time_positions+self.view_types[1]
        for block in self.head:t=block(t)
        t=self.final_norm(t)
        return t.new_zeros(len(t),100,128),t,generated['packet_mask'],generated['time_mask']

    def representation(self,signed_timestamps,tam):
        p,t,pv,tv=self.forward_tokens(signed_timestamps,tam)
        pm=(p*pv[...,None]).sum(1)/pv.sum(1).clamp_min(1)[...,None]
        if self.condition=='attention_readout':
            scores=(t*self.readout_query).sum(-1)/math.sqrt(128)
            weights=scores.masked_fill(~tv,float('-inf')).softmax(1)
            tm=(t*weights[...,None]).sum(1)
        else:tm=(t*tv[...,None]).sum(1)/tv.sum(1).clamp_min(1)[...,None]
        return torch.cat([pm,tm],-1),tm

    def forward(self,signed_timestamps,tam,return_features=False):
        rep,features=self.representation(signed_timestamps,tam)
        logits=self.classifier(rep)
        return (logits,features) if return_features else logits

def supervised_contrastive(features,labels,tau=.1):
    z=F.normalize(features,dim=-1)
    scores=z@z.T/tau
    eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
    positives=labels[:,None].eq(labels[None,:])&~eye
    good=positives.any(1)
    if not good.any():return features.sum()*0
    denominator=torch.logsumexp(scores.masked_fill(eye,float('-inf')),dim=1)
    logp=scores-denominator[:,None]
    means=logp.masked_fill(~positives,0).sum(1)/positives.sum(1).clamp_min(1)
    return -means[good].mean()

def span_mask(tam,generator):
    """One source-only common interval for both direction channels."""
    active=torch.rand(len(tam),generator=generator)<.5
    start=torch.randint(0,1711,(len(tam),),generator=generator)
    bins=torch.arange(1800)[None,:]
    mask=active[:,None]&(bins>=start[:,None])&(bins<start[:,None]+90)
    return tam.masked_fill(mask.to(tam.device)[:,None,:],0)

def learning_rate(kind,step,total=12800):
    if kind=='cosine_lr':return .0001+.00045*(1+math.cos(math.pi*(step-1)/(total-1)))
    return .001 if step<=6400 else (.0003 if step<=9600 else .0001)
