"""Image hypotheses: soft EMA consistency, generator temporal topology, window feedback."""
import torch
from torch.nn import functional as F

@torch.no_grad()
def update_teacher(teacher,student,decay):
    for tp,sp in zip(teacher.parameters(),student.parameters()):tp.mul_(decay).add_(sp,alpha=1-decay)
    for tb,sb in zip(teacher.buffers(),student.buffers()):
        if tb.is_floating_point():tb.mul_(decay).add_(sb,alpha=1-decay)
        else:tb.copy_(sb)

class StageCapture:
    def __init__(self,model):
        self.values=[]
        self.handles=[layer.register_forward_hook(self.capture) for layer in model.generator.stages]
    def capture(self,module,args,out):self.values.append(out)
    def clear(self):self.values=[]
    def close(self):
        for h in self.handles:h.remove()

def topology_loss(stages,bins):
    assert len(stages)==3
    relations=[]
    for z in stages:
        # Each temporal bin is a channel-normalized local descriptor.
        # All three generator scales are divisible by the fixed 24-bin
        # topology grid; reshape+mean keeps the check deterministic on CUDA.
        z=z.reshape(z.shape[0], z.shape[1], bins, z.shape[2]//bins).mean(-1).transpose(1,2)
        z=F.normalize(z,dim=-1,eps=1e-8)
        relations.append(torch.bmm(z,z.transpose(1,2)))
    return (F.mse_loss(relations[0],relations[1])+F.mse_loss(relations[1],relations[2]))/2

class WindowController:
    def __init__(self,initial_entropy,length=16,stride=4,gain=4.):
        self.h0=max(float(initial_entropy),1e-8);self.length=length;self.stride=stride;self.gain=gain
        self.history=[];self.scale=1.;self.growth=0.
    def step(self,h):
        self.history.append(float(h));n=len(self.history)
        refreshed=n>=self.length and (n-self.length)%self.stride==0
        if refreshed:
            w=self.history[-self.length:];edge=self.stride
            self.growth=(sum(w[-edge:])/edge-sum(w[:edge])/edge)/self.h0
            # Entropy rises => conservative updates; never exceed baseline LR.
            self.scale=max(.25,min(1.,1/(1+self.gain*max(self.growth,0.))))
        return {'scale':self.scale,'window_growth':self.growth,'window_refreshed':refreshed}

def set_bn_momentum(model,momentum):
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.momentum=float(momentum)
