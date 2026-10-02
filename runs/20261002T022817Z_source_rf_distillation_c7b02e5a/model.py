"""Baseline student unchanged; source-only detached teacher KL."""
import torch
from torch.nn import functional as F
from generator_base import MultiViewModel as ParentModel,span_mask,learning_rate,make_optimizer
FACTORS={k:(False,False) for k in ['ce_only','kd_01','kd_05','shuffled_05']}
class MultiViewModel(ParentModel):
    def __init__(self,condition,num_classes=102,dropout=.1):
        if condition not in FACTORS:raise ValueError(condition)
        super().__init__('flat_none',num_classes,dropout)
        self.condition=condition

def distillation_loss(student,teacher,temperature):
    return F.kl_div(F.log_softmax(student/temperature,dim=1),F.softmax(teacher.detach()/temperature,dim=1),reduction='batchmean')*temperature**2
