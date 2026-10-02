"""Generator-only factorial; raw summaries and Transformer interface unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from input_base import MultiViewModel as ParentModel
from encoder_base import TAMGenerator
from candidate_base import span_mask

FACTORS={'flat_none':(False,False),'flat_norm':(True,False),
         'hier_none':(False,True),'hier_norm':(True,True)}

class Generator(TAMGenerator):
    def __init__(self,previous,normalization,hierarchical):
        super().__init__(previous,'multiscale_d139')
        self.normalization=normalization;self.hierarchical=hierarchical
        with torch.random.fork_rng(devices=[]):
            self.channel_norms=nn.ModuleList([nn.LayerNorm(16,eps=1e-5) for _ in range(2)] if normalization else [])

    def stage(self,z,layer,index,dilation):
        z=F.conv1d(z,layer.weight,layer.bias,padding=2*dilation,dilation=dilation)
        if self.normalization:z=self.channel_norms[index](z.transpose(1,2)).transpose(1,2)
        return self.act(z)

    def encode_scale(self,counts,dilation):
        z=self.stage(counts,self.time_conv[0],0,dilation)
        if self.hierarchical:z=z.reshape(len(z),16,600,3).mean(-1)
        return self.stage(z,self.time_conv[1],1,dilation)

    def forward(self,signed_timestamps,tam):
        b=len(tam)
        if tam.shape!=(b,2,1800) or signed_timestamps.shape!=(b,5000):raise ValueError('input shape')
        counts=tam.float().log1p()
        fixed=counts.reshape(b,2,120,15).permute(0,2,1,3).reshape(b,120,30)
        z=sum(self.encode_scale(counts,d) for d in self.dilations)/len(self.dilations)
        if not self.hierarchical:z=z.reshape(b,16,600,3).mean(-1)
        learned=z.reshape(b,16,120,5).permute(0,2,3,1).reshape(b,120,80)
        return {'packet':tam.new_zeros(b,100,100),'time':torch.cat([fixed,learned],-1),
                'packet_mask':torch.zeros(b,100,dtype=torch.bool,device=tam.device),
                'time_mask':torch.ones(b,120,dtype=torch.bool,device=tam.device)}

class MultiViewModel(ParentModel):
    def __init__(self,condition,num_classes=102,dropout=.1):
        if condition not in FACTORS:raise ValueError(condition)
        super().__init__('log_current',num_classes,dropout)
        self.condition=condition
        self.generator=Generator(self.generator,*FACTORS[condition])

def learning_rate(kind,step,total=12800):
    return .001 if step<=6400 else (.0003 if step<=9600 else .0001)

def make_optimizer(model,kind):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
