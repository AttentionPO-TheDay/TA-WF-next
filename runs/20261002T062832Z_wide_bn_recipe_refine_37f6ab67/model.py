"""Fixed wide+BatchNorm progressive generator with optimizer recipe factors."""
import torch
from torch import optim
from structure_base import MultiViewModel as StructureModel
from structure_base import span_mask,mix_batch,mixed_ce
KINDS=['baseline','lr05','wd05','lr05_wd05']
class MultiViewModel(StructureModel):
    def __init__(self,kind,num_classes=102,dropout=.1):
        if kind not in KINDS:raise ValueError(kind)
        super().__init__('wide_bn',num_classes,dropout);self.condition=kind

def learning_rate(kind,step,total=12800):
    if kind in ('lr05','lr05_wd05'):return .0005 if step<=6400 else (.00015 if step<=9600 else .00005)
    return .001 if step<=6400 else (.0003 if step<=9600 else .0001)
def make_optimizer(model,kind):
    wd=.0005 if kind in ('wd05','lr05_wd05') else .0001
    return optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=wd,betas=(.9,.999),eps=1e-8)
