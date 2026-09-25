from __future__ import annotations
import csv, hashlib, json, os, sys, zipfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
DATA = Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
SEED = 1729
L = 5000

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return z['X'], z['y']

def features(x):
    # Direction-only, deliberately matching the previous diagnostic's raw prefix.
    d = np.sign(x[:, :L]).astype(np.float32)
    out = {'packet': d}
    windows=[]
    for row in d:
        vals=[]
        for width in (50, 250):
            for start in range(0, L, width):
                part=row[start:start+width]
                n=int(np.count_nonzero(part))
                if n == 0:
                    vals.extend((0., 0.))
                else:
                    q=part[:n]
                    vals.extend((float((q>0).mean()), float((q[1:]!=q[:-1]).mean()) if n>1 else 0.))
        windows.append(vals)
    out['windows'] = np.asarray(windows, dtype=np.float32)
    return out

def select(x, y, cap):
    rng=np.random.default_rng(SEED)
    return np.concatenate([rng.permutation(np.flatnonzero(y==c))[:cap] for c in range(102)])

def metric(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102); tp=np.diag(cm); den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float((p==y).mean()),'macro_f1':float(np.divide(2*tp,den,out=np.zeros(102),where=den!=0).mean())}

def main():
    paths={'source':DATA/'NetworkDrift'/'train.npz','valid':DATA/'NetworkDrift'/'valid.npz',
           'jp':DATA/'NetworkDrift'/'JP.npz','subpage':DATA/'BehaviorDrift'/'subpage.npz'}
    data={}; manifest={}
    for role,path in paths.items():
        x,y=load(path); cap={'source':20,'valid':5,'jp':20,'subpage':20}[role]; idx=select(x,y,cap)
        if set(y[idx].tolist()) != set(range(102)): raise ValueError(role+' missing class')
        data[role]={'x':features(x[idx]),'y':y[idx].astype(int),'idx':idx.tolist()}
        manifest[role]={'path':str(path.relative_to(DATA)),'rows':idx.tolist(),'count':len(idx),'label_use':'source fit; other roles scoring only'}
        print('features',role,len(idx),flush=True)
    models={}; preds={}; dimensions={}
    for view in ('packet','windows'):
        scaler=StandardScaler(); train=scaler.fit_transform(data['source']['x'][view]);
        clf=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=50,tol=1e-3,random_state=SEED,shuffle=True)
        clf.fit(train,data['source']['y']); models[view]=(scaler,clf); dimensions[view]=int(train.shape[1])
        for role in ('valid','jp','subpage'):
            preds[role+'_'+view]=clf.predict(scaler.transform(data[role]['x'][view])).astype(np.int16)
    np.savez_compressed(RUN/'artifacts/predictions.npz',**preds)
    seal=hashlib.sha256((RUN/'artifacts/predictions.npz').read_bytes()).hexdigest()
    (RUN/'artifacts/manifest.json').write_text(json.dumps({'sampling':manifest,'dimensions':dimensions,'prediction_sha256':seal},indent=2))
    rows=[]
    for role in ('valid','jp','subpage'):
        for view in ('packet','windows'):
            rows.append({'role':role,'view':view,**metric(data[role]['y'],preds[role+'_'+view])})
    with (RUN/'artifacts/metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['role','view','accuracy','macro_f1']); w.writeheader(); w.writerows(rows)
    (RUN/'artifacts/summary.json').write_text(json.dumps({'metrics':rows,'dimensions':dimensions,'config':'fixed CPU SGD; no target tuning; predictions sealed before scoring'},indent=2))
    print('COMPLETE')

if __name__=='__main__': main()
