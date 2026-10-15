"""Progressive student: width/BN factorial plus isolated Mixup condition."""
import torch
from torch import nn
from torch.nn import functional as F
from framework_base import Student,LocalStage,ProgressiveGenerator,span_mask,learning_rate,make_optimizer
FACTORS={'baseline':(1,False,False),'wide':(2,False,False),'bn':(1,True,False),'wide_bn':(2,True,False),'mixup':(1,False,True)}
class NormStage(nn.Module):
    def __init__(self,previous):
        super().__init__();self.conv1=previous.conv1;self.conv2=previous.conv2;self.skip=previous.skip;self.act=previous.act
        cout=self.conv2.out_channels
        self.norm1=nn.BatchNorm1d(cout,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
        self.norm2=nn.BatchNorm1d(cout,eps=1e-5,momentum=.1,affine=True,track_running_stats=True)
    def forward(self,x):return self.act(self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))+self.skip(x))
class WideGenerator(ProgressiveGenerator):
    def __init__(self,previous):
        nn.Module.__init__(self);self.packet_conv=previous.packet_conv
        self.stages=nn.ModuleList([LocalStage(2,64),LocalStage(64,128),LocalStage(128,256)])
        self.to_features=nn.Conv1d(256,80,1)
class MultiViewModel(Student):
    def __init__(self,kind,num_classes=102,dropout=.1):
        if kind not in FACTORS:raise ValueError(kind)
        super().__init__('progressive_transformer',num_classes,dropout);self.condition=kind
        width,bn,mixup=FACTORS[kind]
        with torch.random.fork_rng(devices=[]):
            if width==2:self.generator=WideGenerator(self.generator)
            if bn:self.generator.stages=nn.ModuleList([NormStage(s) for s in self.generator.stages])

def mix_batch(tam,labels,rng,alpha=.2):
    weight=float(rng.beta(alpha,alpha));perm=torch.as_tensor(rng.permutation(len(tam)),device=tam.device)
    return weight*tam+(1-weight)*tam[perm],labels[perm],weight

def mixed_ce(logits,labels,other,weight):
    return weight*F.cross_entropy(logits,labels)+(1-weight)*F.cross_entropy(logits,other)
