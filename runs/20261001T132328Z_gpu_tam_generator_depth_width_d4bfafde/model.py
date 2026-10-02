"""Depth/width factorial, fixed dilation schedules and unchanged classifier.

encoder_base.py is the preceding run's explicit model.py copy. Additional
pointwise residual layers do not expand the spatial receptive field.
"""
import torch
from torch import nn
from torch.nn import functional as F
from encoder_base import MultiViewModel as PreviousModel, TAMGenerator
from base_model import parameter_counts

CONDITIONS={'d2_w16':(2,16),'d2_w32':(2,32),'d4_w16':(4,16),'d4_w32':(4,32)}

class CapacityGenerator(TAMGenerator):
    def __init__(self,previous,depth,width):
        super().__init__(previous,'multiscale_d139')
        self.depth,self.width=depth,width
        with torch.random.fork_rng(devices=[]):
            if width!=16:
                self.time_conv=nn.ModuleList([nn.Conv1d(2,width,5,padding=2),nn.Conv1d(width,16,5,padding=2)])
            self.depth_mix=nn.ModuleList([nn.Conv1d(width,width,1) for _ in range(depth-2)])
            # Identity at initialization, but branch derivatives are nonzero:
            # GELU'(0)=0.5 and the residual input remains visible.
            for layer in self.depth_mix:
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def encode_scale(self,log_counts,dilation):
        first,last=self.time_conv
        z=self.act(F.conv1d(log_counts,first.weight,first.bias,padding=2*dilation,dilation=dilation))
        for layer in self.depth_mix:
            z=z+self.act(layer(z))
        return self.act(F.conv1d(z,last.weight,last.bias,padding=2*dilation,dilation=dilation))

class MultiViewModel(PreviousModel):
    def __init__(self,condition,num_classes=102,dropout=.1):
        if condition not in CONDITIONS:raise ValueError(condition)
        super().__init__('multiscale_d139',num_classes,dropout)
        self.condition=condition
        self.generator=CapacityGenerator(self.generator,*CONDITIONS[condition])
