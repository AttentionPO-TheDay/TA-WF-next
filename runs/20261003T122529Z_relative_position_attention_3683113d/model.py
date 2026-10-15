import torch
from torch import nn
from framework_base import Student, LocalStage, ProgressiveGenerator
from candidate_base import span_mask

def learning_rate(step,total=12800):
    return .001 if step<=6400 else (.0003 if step<=9600 else .0001)
class NormStage(nn.Module):
    def __init__(self,previous):
        super().__init__(); self.conv1,self.conv2,self.skip,self.act=previous.conv1,previous.conv2,previous.skip,previous.act; c=self.conv2.out_channels
        self.norm1=nn.BatchNorm1d(c,eps=1e-5,momentum=.1,affine=True,track_running_stats=True); self.norm2=nn.BatchNorm1d(c,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
    def forward(self,x):
        residual=self.skip(x); x=self.act(self.norm1(self.conv1(x))); return self.act(self.norm2(self.conv2(x))+residual)
class WideGenerator(ProgressiveGenerator):
    def __init__(self,previous):
        nn.Module.__init__(self); self.packet_conv=previous.packet_conv; self.stages=nn.ModuleList([LocalStage(2,64),LocalStage(64,128),LocalStage(128,256)]); self.to_features=nn.Conv1d(256,80,1)
class MultiViewModel(Student):
    CONDITIONS=('ce_control','relative_position')
    def __init__(self,condition,num_classes=102,dropout=.1):
        if condition not in self.CONDITIONS: raise ValueError(condition)
        super().__init__('progressive_transformer',num_classes,dropout); self.condition=condition
        with torch.random.fork_rng(devices=[]):
            self.generator=WideGenerator(self.generator); self.generator.stages=nn.ModuleList([NormStage(s) for s in self.generator.stages])
        pos=torch.arange(220,dtype=torch.float32); d=(pos[:,None]-pos[None,:]).abs(); self.register_buffer('relative_bias',-.10*torch.log1p(d),persistent=True)
    def forward(self,signed_timestamps,tam):
        generated=self.generator(signed_timestamps,tam); packet,temporal=generated['packet'],generated['time']; p_valid,t_valid=generated['packet_mask'],generated['time_mask']
        p=self.packet_projection(packet)+self.packet_positions+self.view_types[0]; t=self.time_projection(temporal)+self.time_positions+self.view_types[1]
        valid=torch.cat([p_valid,t_valid],1); padding=~valid; z=torch.cat([p,t],1).masked_fill(padding[...,None],0)
        for layer in self.head:
            if self.condition=='relative_position': z=layer(z,src_mask=self.relative_bias,src_key_padding_mask=padding)
            else: z=layer(z,src_key_padding_mask=padding)
            z=z.masked_fill(padding[...,None],0)
        z=self.final_norm(z).masked_fill(padding[...,None],0); p,t=z[:,:100],z[:,100:]
        pm=(p*p_valid[...,None]).sum(1)/p_valid.sum(1).clamp_min(1)[...,None]; tm=(t*t_valid[...,None]).sum(1)/t_valid.sum(1).clamp_min(1)[...,None]
        return self.classifier(torch.cat([pm,tm],-1))
