"""WF tokenizer/head architecture factorial and explicit RF recipe control."""
import torch
from torch import nn
from generator_base import MultiViewModel as ParentModel,span_mask,learning_rate,make_optimizer
from base_model import TokenMLP,MultiViewModel as OriginalModel
from rf_native import RFNative
KINDS=['shallow_transformer','shallow_mlp','progressive_transformer','progressive_mlp','rf_matched']
class LocalStage(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.conv1=nn.Conv1d(cin,cout,5,padding=2);self.conv2=nn.Conv1d(cout,cout,5,padding=2);self.skip=nn.Conv1d(cin,cout,1);self.act=nn.GELU()
    def forward(self,x):return self.act(self.conv2(self.act(self.conv1(x)))+self.skip(x))
class ProgressiveGenerator(nn.Module):
    def __init__(self,previous):
        super().__init__();self.packet_conv=previous.packet_conv
        self.stages=nn.ModuleList([LocalStage(2,32),LocalStage(32,64),LocalStage(64,128)])
        self.to_features=nn.Conv1d(128,80,1)
    def forward(self,signed_timestamps,tam):
        b=len(tam)
        if tam.shape!=(b,2,1800) or signed_timestamps.shape!=(b,5000):raise ValueError('input shape')
        counts=tam.float().log1p();fixed=counts.reshape(b,2,120,15).permute(0,2,1,3).reshape(b,120,30)
        z=counts
        for layer,pool in zip(self.stages,[3,5,1]):
            z=layer(z)
            if pool>1:z=z.reshape(b,z.shape[1],z.shape[2]//pool,pool).mean(-1)
        learned=self.to_features(z).transpose(1,2)
        return {'packet':tam.new_zeros(b,100,100),'time':torch.cat([fixed,learned],-1),'packet_mask':torch.zeros(b,100,dtype=torch.bool,device=tam.device),'time_mask':torch.ones(b,120,dtype=torch.bool,device=tam.device)}
class Student(ParentModel):
    def __init__(self,kind,num_classes=102,dropout=.1):
        super().__init__('flat_none',num_classes,dropout);self.condition=kind
        if kind.startswith('progressive'):
            with torch.random.fork_rng(devices=[]):self.generator=ProgressiveGenerator(self.generator)
        if kind.endswith('_mlp'):
            with torch.random.fork_rng(devices=[]):self.head=TokenMLP(dropout)
            self.head_name='mlp'
    def forward_tokens(self,signed_timestamps,tam):
        if self.head_name=='mlp':return OriginalModel.forward_tokens(self,signed_timestamps,tam)
        return super().forward_tokens(signed_timestamps,tam)
class RFControl(nn.Module):
    def __init__(self,num_classes):super().__init__();self.rf=RFNative(num_classes)
    def forward(self,signed_timestamps,tam,return_features=False):
        z=self.rf(tam.float().log1p()[:,None]);return (z,z) if return_features else z

def MultiViewModel(kind,num_classes=102,dropout=.1):
    if kind not in KINDS:raise ValueError(kind)
    return RFControl(num_classes) if kind=='rf_matched' else Student(kind,num_classes,dropout)
