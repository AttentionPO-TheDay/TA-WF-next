"""Parameter-matched TAM encoders; ordered tokens and Transformer unchanged.

Explicit local copies: view_base.py from the preceding view-ablation model.py,
base_model.py from that run's base_model.py. No archived project dependencies.
"""
import torch
from torch import nn
from torch.nn import functional as F
from view_base import MultiViewModel as ViewModel
from base_model import parameter_counts

ENCODERS={'local_d1':(1,), 'context_d9':(9,), 'multiscale_d139':(1,3,9)}

class TAMGenerator(nn.Module):
    def __init__(self,previous_generator,encoder):
        super().__init__()
        self.packet_conv=previous_generator.packet_conv
        self.time_conv=previous_generator.time_conv
        self.act=previous_generator.act
        self.encoder=encoder
        self.dilations=ENCODERS[encoder]
        self.active_token_keys=('time',)

    def encode_scale(self,log_counts,dilation):
        z=log_counts
        for layer in self.time_conv:
            z=self.act(F.conv1d(z,layer.weight,layer.bias,padding=2*dilation,dilation=dilation))
        return z

    def forward(self,signed_timestamps,tam):
        if signed_timestamps.ndim!=2 or signed_timestamps.shape[1]!=5000:
            raise ValueError('signed_timestamps must be B x 5000')
        b=signed_timestamps.shape[0]
        if tam.shape!=(b,2,1800):raise ValueError('tam must be B x 2 x 1800')
        counts=tam.float().log1p()
        fixed=counts.reshape(b,2,120,15).permute(0,2,1,3).reshape(b,120,30)
        scales=[self.encode_scale(counts,d) for d in self.dilations]
        # No average across time or direction here: mix only receptive-field
        # scales at the same absolute bin, with equal predeclared weights.
        z=scales[0] if len(scales)==1 else sum(scales)/len(scales)
        learned=z.reshape(b,16,120,5,3).mean(-1).permute(0,2,3,1).reshape(b,120,80)
        return {'packet':tam.new_zeros(b,100,100),'time':torch.cat([fixed,learned],-1),
                'packet_mask':torch.zeros(b,100,dtype=torch.bool,device=tam.device),
                'time_mask':torch.ones(b,120,dtype=torch.bool,device=tam.device)}

class MultiViewModel(ViewModel):
    def __init__(self,encoder,num_classes=102,dropout=.1):
        if encoder not in ENCODERS:raise ValueError(encoder)
        super().__init__('tam_only',num_classes,dropout)
        self.encoder=encoder
        self.generator=TAMGenerator(self.generator,encoder)
