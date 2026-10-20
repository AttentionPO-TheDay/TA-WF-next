"""Coverage-preserving support perturbations; no query labels in routing."""
import numpy as np
from maintenance_base import unit, margin_gain


def bags(y, count=5):
    result=[]
    for b in range(count):
        keep=np.ones(len(y),dtype=bool)
        for cls in np.unique(y):
            ids=np.flatnonzero(y==cls)
            if len(ids)>1:keep[ids[(b+int(cls))%len(ids)]]=False
        result.append(np.flatnonzero(keep))
    return result


def local(x, train_x, train_y):
    sim=unit(x)@unit(train_x).T
    scores=np.full((len(x),102),-np.inf,dtype=np.float32)
    for cls in np.unique(train_y):scores[:,cls]=sim[:,train_y==cls].max(1)
    return scores


def gate(p, base, scores):
    i=np.arange(len(p))
    return np.where((p==scores.argmax(1))&(scores[i,p]>scores[i,base]),p,base)


def vote(base, proposals, gains, k):
    ps=np.stack(proposals);gs=np.stack(gains);i=np.arange(len(base))
    counts=np.zeros((len(base),102),dtype=np.int8)
    for p in ps:counts[i,p]+=1
    candidate=counts.argmax(1)
    accept=(candidate!=base)&(counts[i,candidate]>=k)
    support=ps==candidate[None,:]
    score=(gs*support).sum(0)/np.maximum(support.sum(0),1)
    return np.where(accept,candidate,base),np.where(accept,score,0).astype(np.float32)


def outputs(original, old_logits, old_local, bag_logits, bag_local, full_local):
    base=original.argmax(1);result={}
    for family,zs,ls,ks in [('old',old_logits,old_local,[2,3]),('bag',bag_logits,bag_local,[3,4,5])]:
        ps=[z.argmax(1) for z in zs]
        gs=[margin_gain(z,original,p,base) for z,p in zip(zs,ps)]
        routed=[gate(p,base,s) for p,s in zip(ps,ls)]
        for k in ks:
            result[f'{family}_local_vote{k}']=vote(base,routed,gs,k)
            if k<(3 if family=='old' else 5):
                p,g=vote(base,ps,gs,k);p=gate(p,base,full_local)
                result[f'{family}_heads_vote{k}_full_local']=(p,np.where(p!=base,g,0))
    return result
