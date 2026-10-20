"""Explicit inputs and label-free scoring/cluster fitting APIs."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from runtime import ROOT,C,entropy
from candidate import gpu_dict,banks
from routing import geometry,normalized_margin
from methods import standardize


def inputs(seed):
    art=torch.load(ROOT/C['previous_run']/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True);p=gpu_dict(art['fits']['1']);bb=banks(art,p)
    prior=torch.load(ROOT/C['baseline_run']/'artifacts'/f'source_calibration_{seed}.pt',map_location='cpu',weights_only=True)
    source=torch.load(ROOT/C['v4_run']/'artifacts'/f'source_{seed}.pt',map_location='cpu',weights_only=True)
    return p,bb,prior,source


def canonical(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}


def old_adapter(seed,name):return torch.load(ROOT/C['v6_run']/'checkpoints'/f'{seed}_{name}_sharp_target_only.pt',map_location='cpu',weights_only=True)['adapter']


def geo_margin(g,cp):
    d=g['cosine'];mine=d.gather(1,cp[:,None]).squeeze(1);other=d.clone().scatter_(1,cp[:,None],torch.inf).min(-1).values
    return (other-mine)/(other+1e-8)


@torch.no_grad()
def signals(base_z,z,base_e,e,p):
    cp=z.argmax(-1);bg=geometry(base_e,p);ag=geometry(e,p)
    return dict(margin_gain=(normalized_margin(z,cp)-normalized_margin(base_z,cp)).cpu().numpy(),nearest=ag['nearest'].cpu().numpy(),geometry_margin=geo_margin(ag,cp).cpu().numpy(),frozen_geometry_margin=geo_margin(bg,cp).cpu().numpy())


@torch.no_grad()
def kl_gate(z,e,p,valid_z):
    hcut=torch.quantile(entropy(valid_z),.75);g=geometry(e,p);v=(entropy(z)<=hcut)&(z.argmax(-1)==g['nearest'])
    return v.float().detach(),dict(entropy_cutoff=float(hcut),eligible=int(v.sum()),fraction=float(v.float().mean()))


def assign(x,centers):
    return np.concatenate([cdist(x[i:i+512],centers,metric='sqeuclidean').argmin(-1) for i in range(0,len(x),512)])


def cluster(ref_e,query_e,p):
    # No query statistics/labels in centroid fitting or standardization.
    r=standardize(ref_e,p).cpu().numpy().astype('float64');q=standardize(query_e,p).cpu().numpy().astype('float64')
    fit=KMeans(n_clusters=102,n_init=3,max_iter=100,random_state=20261019).fit(r)
    return fit.cluster_centers_,assign(q,fit.cluster_centers_)


def diagnostics(y,bp,cp,clusters,nearest,common):
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
    table=np.zeros((102,102),dtype=np.int64);np.add.at(table,(clusters,y),1)
    n=max(int(common.sum()),1)
    return dict(cluster_purity=float(table.max(1).sum()/len(y)),cluster_ari=float(adjusted_rand_score(y,clusters)),cluster_nmi=float(normalized_mutual_info_score(y,clusters)),source_prototype_accuracy=float((nearest==y).mean()),common_wrong_count=int(common.sum()),common_same_wrong=float(((cp==bp)&common).sum()/n),common_corrected=float(((cp==y)&common).sum()/n),common_other_wrong=float(((cp!=bp)&(cp!=y)&common).sum()/n))
