"""Unchanged baseline student with source-only soft targets and TAM Mixup."""
import torch
from torch.nn import functional as F
from generator_base import MultiViewModel as ParentModel,span_mask,learning_rate,make_optimizer
FACTORS={k:(False,False) for k in ['ce_only','shuffled_05','uniform_05','marginal_05','mixup','mixup_uniform_05']}
class MultiViewModel(ParentModel):
    def __init__(self,condition,num_classes=102,dropout=.1):
        if condition not in FACTORS:raise ValueError(condition)
        super().__init__('flat_none',num_classes,dropout)
        self.condition=condition

def probability_kl(student,probability,temperature=2.):
    if probability.ndim==1:probability=probability.unsqueeze(0).expand(len(student),-1)
    return F.kl_div(F.log_softmax(student/temperature,dim=1),probability.detach(),reduction='batchmean')*temperature**2

def mix_batch(tam,labels,rng,alpha=.2):
    weight=float(rng.beta(alpha,alpha));permutation=torch.as_tensor(rng.permutation(len(tam)),device=tam.device)
    return weight*tam+(1-weight)*tam[permutation],labels[permutation],weight

def mixed_ce(logits,labels,other_labels,weight):
    return weight*F.cross_entropy(logits,labels)+(1-weight)*F.cross_entropy(logits,other_labels)
