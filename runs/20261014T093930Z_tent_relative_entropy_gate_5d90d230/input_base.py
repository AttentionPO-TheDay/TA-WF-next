"""Identical learnable generator/Transformer; explicit input scaling and recipe factor."""
import torch
from torch import nn
from encoder_base import TAMGenerator
from candidate_base import MultiViewModel as ParentModel,span_mask

FACTORS={'log_current':('log1p','current'),'raw_current':('raw','current'),
         'log_rfstyle':('log1p','rfstyle'),'raw_rfstyle':('raw','rfstyle')}

class InputGenerator(TAMGenerator):
    def __init__(self,previous,input_scale):
        super().__init__(previous,'multiscale_d139');self.input_scale=input_scale

    def forward(self,signed_timestamps,tam):
        b=len(tam)
        if tam.shape!=(b,2,1800) or signed_timestamps.shape!=(b,5000):raise ValueError('input shape')
        counts=tam.float().log1p() if self.input_scale=='log1p' else tam.float()
        fixed=counts.reshape(b,2,120,15).permute(0,2,1,3).reshape(b,120,30)
        z=sum(self.encode_scale(counts,d) for d in self.dilations)/len(self.dilations)
        learned=z.reshape(b,16,120,5,3).mean(-1).permute(0,2,3,1).reshape(b,120,80)
        return {'packet':tam.new_zeros(b,100,100),'time':torch.cat([fixed,learned],-1),
                'packet_mask':torch.zeros(b,100,dtype=torch.bool,device=tam.device),
                'time_mask':torch.ones(b,120,dtype=torch.bool,device=tam.device)}

class MultiViewModel(ParentModel):
    def __init__(self,condition,num_classes=102,dropout=.1):
        if condition not in FACTORS:raise ValueError(condition)
        super().__init__('span_mask',num_classes,dropout)
        self.condition=condition
        self.generator=InputGenerator(self.generator,FACTORS[condition][0])

def learning_rate(kind,step,total=12800):
    if FACTORS[kind][1]=='rfstyle':return .0005*(.2**((step-1)/total))
    return .001 if step<=6400 else (.0003 if step<=9600 else .0001)

def make_optimizer(model,kind):
    parameters=[p for p in model.parameters() if p.requires_grad]
    if FACTORS[kind][1]=='rfstyle':return torch.optim.Adam(parameters,lr=.0005,weight_decay=.001,betas=(.9,.999),eps=1e-8)
    return torch.optim.AdamW(parameters,lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
