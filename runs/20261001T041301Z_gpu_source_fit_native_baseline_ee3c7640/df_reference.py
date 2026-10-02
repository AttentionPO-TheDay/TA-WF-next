"""Explicit PyTorch port of upstream DF NoDef at commit 38df0c15.
Source snapshots and framework differences: PLAN.md and artifacts/upstream.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SamePool(nn.Module):
    def forward(self,x):
        pad=max(0,(math.ceil(x.shape[-1]/4)-1)*4+8-x.shape[-1])
        return F.max_pool1d(F.pad(x,(pad//2,pad-pad//2),value=-float('inf')),8,4)

class DFReference(nn.Module):
    def __init__(self,classes=102):
        super().__init__()
        layers=[]; incoming=1
        for i,out in enumerate([32,64,128,256]):
            for _ in range(2):
                layers.extend([nn.Conv1d(incoming,out,8,padding='same',bias=True),nn.BatchNorm1d(out,eps=.001,momentum=.01),nn.ELU() if i==0 else nn.ReLU()]);incoming=out
            layers.extend([SamePool(),nn.Dropout(.1)])
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Linear(20*256,512),nn.BatchNorm1d(512,eps=.001,momentum=.01),nn.ReLU(),nn.Dropout(.7),nn.Linear(512,512),nn.BatchNorm1d(512,eps=.001,momentum=.01),nn.ReLU(),nn.Dropout(.5),nn.Linear(512,classes))
        for m in self.modules():
            if isinstance(m,(nn.Linear,nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.features(x)
        # Match Keras channels-last Flatten order.
        return self.head(x.transpose(1,2).reshape(len(x),-1))
