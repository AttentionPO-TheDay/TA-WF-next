"""Frozen entropy feedback and BN interpolation; no target labels or pseudo labels."""
import math
import torch
from torch import nn

def control(mean_entropy, initial_entropy, base_lr):
    relative=float(mean_entropy)/max(float(initial_entropy),1e-8)
    scale=min(1.0,max(0.25,1.0/max(relative,1e-8)))
    return {'relative_entropy':relative,'scale':scale,'learning_rate':float(base_lr)*scale,
            'target_bn_momentum':0.01+0.09*scale}

def keep_mask(normalized_entropies,lower=.02,upper=.50):
    return (normalized_entropies>=lower)&(normalized_entropies<=upper)

def set_bn_momentum(model,value):
    count=0
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            module.momentum=float(value);count+=1
    assert count>0
    return count
