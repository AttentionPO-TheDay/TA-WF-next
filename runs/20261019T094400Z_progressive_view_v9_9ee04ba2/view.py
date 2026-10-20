"""Frozen source-only coarse-TAM PCA view; no primary-encoder representations."""
import math
import numpy as np
import torch
from runtime import RUN,ROOT,C,save,sha,metrics


def features(tam):
    x=tam.detach().cpu().double();assert x.ndim==3 and x.shape[1:]==(2,1800) and (x>=0).all()
    return torch.cat([x.reshape(len(x),2,k,1800//k).sum(-1).log1p().flatten(1) for k in [1,6,24,120]],1).numpy()


def project(tam,s):
    f=features(tam);x=(f-s['mean'].numpy())/s['sd'].numpy();return ((x-s['pca_mean'].numpy())@s['components'].numpy().T)/s['projection_sd'].numpy()


def probs(tam,s):
    v=project(tam,s);c=s['centers'].numpy();d=((v[:,None,:]-c[None])**2).mean(-1);z=torch.from_numpy(-d/s['temperature']);q=z.softmax(-1).float()
    return q


def fit():
    d=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True);f=features(d['source']['tam']);y=d['source']['labels'].numpy();mean=f.mean(0);sd=np.maximum(f.std(0),1e-3);x=(f-mean)/sd;pm=x.mean(0);_,_,v=np.linalg.svd(x-pm,full_matrices=False);comp=v[:32];raw=(x-pm)@comp.T;vs=np.maximum(raw.std(0),1e-6);a=raw/vs;centers=np.stack([a[y==c].mean(0) for c in range(102)])
    s={k:torch.from_numpy(val) for k,val in dict(mean=mean,sd=sd,pca_mean=pm,components=comp,projection_sd=vs,centers=centers).items()};trials=[];vy=d['valid']['labels'].numpy()
    for temp in [.03,.1,.3,1.,3.,10.,30.]:
        s['temperature']=temp;q=probs(d['valid']['tam'],s);nll=float(-q[torch.arange(len(q)),torch.from_numpy(vy)].clamp_min(1e-12).log().mean());trials.append(dict(temperature=temp,nll=nll))
    best=min(trials,key=lambda x:x['nll']);s['temperature']=best['temperature'];q=probs(d['valid']['tam'],s);pred=q.argmax(-1).numpy();met=metrics(vy,pred,pred);s['enabled']=bool(met['macro_f1']>=.1 and best['nll']<math.log(102));s['valid_q']=q;s['source_rows']=d['source']['rows'];s['valid_rows']=d['valid']['rows']
    assert len(y)==18553 and len(set(y))==102 and np.allclose(comp@comp.T,np.eye(32),atol=1e-10)
    torch.save(s,RUN/'artifacts/source_view.pt');save('source_view_audit.json',dict(passed=True,source_rows=18553,valid_rows=len(vy),pca_input_dimension=302,output_dimension=32,orthogonality_error=float(np.max(np.abs(comp@comp.T-np.eye(32)))),valid_macro_f1=met['macro_f1'],valid_accuracy=met['accuracy'],temperature=s['temperature'],nll=best['nll'],trials=trials,enabled=s['enabled'],artifact_sha256=sha(RUN/'artifacts/source_view.pt'),query_seen=False))

if __name__=='__main__':
    torch.set_num_threads(2);fit()
