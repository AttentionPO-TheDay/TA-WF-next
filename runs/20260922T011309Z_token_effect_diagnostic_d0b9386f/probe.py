"""Frozen source-only centroid probe. No target-label fitting or tuning."""
import ast
import csv
import hashlib
import json
import struct
import zipfile
from pathlib import Path
import numpy as np
from ta_wf_next.traffic_views import generate_views

RUN=Path(__file__).resolve().parent
DATA=Path(json.loads((RUN.parents[1]/'configs/datasets.json').read_text())['data_root'])
NAMES=['packet','exact_run','coarse_run','windows','timing']
L=5000

def mmap_x(path):
    with zipfile.ZipFile(path) as z: info=z.getinfo('X.npy'); assert info.compress_type==0
    with path.open('rb') as f:
        f.seek(info.header_offset); h=struct.unpack('<IHHHHHIIIHH',f.read(30)); f.seek(h[-2]+h[-1],1)
        assert f.read(6)==b'\x93NUMPY'
        major,_=f.read(2); assert major in (1,2)
        length=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
        header=ast.literal_eval(f.read(length).decode('latin1').strip()); offset=f.tell()
    assert header['descr']=='<f8' and not header['fortran_order']
    return np.memmap(path,mode='r',dtype='<f8',shape=header['shape'],offset=offset)

def features(row):
    v=generate_views(row[:L],input_kind='signed_timestamp',budget=L,enable_timing=True)
    assert v.timing.invalid_same_direction_intervals==v.timing.invalid_run_spans==0
    packet=np.zeros(L,np.float32); packet[:len(v.packet_direction)]=v.packet_direction
    exact=np.zeros(L,np.float32); coarse=np.zeros(L,np.float32)
    for i,(r,c) in enumerate(zip(v.runs.runs,v.runs.coarse())):
        exact[i]=r.direction*np.log1p(r.count); coarse[i]=c.direction*(1+c.log2_count_bin)
    windows=[]
    for width,entries in v.direction_windows:
        a=np.zeros(((L+width-1)//width,2),np.float32)
        for i,w in enumerate(entries): a[i]=[w.positive_fraction,w.transition_fraction]
        windows.extend(a.ravel())
    # Distinct 2-channel slots: value and validity, not fabricated timing zero.
    timing=np.zeros((2,L,2),np.float32)
    for k,seq in enumerate([v.timing.same_direction_interval,v.timing.run_span]):
        for i,value in enumerate(seq):
            if value is not None: timing[k,i]=[np.log1p(value),1.]
    return dict(packet=packet,exact_run=exact,coarse_run=coarse,windows=np.array(windows,np.float32),timing=timing.ravel())

def normalize(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def fit(x,y):
    mean=x.mean(axis=0); std=x.std(axis=0); std=np.where(std<1e-6,1.,std)
    z=normalize((x-mean)/std)
    prototypes=normalize(np.stack([z[y==c].mean(axis=0) for c in range(102)]))
    return mean,std,prototypes

def predict_scores(x,model):
    mean,std,prototypes=model
    return normalize((x-mean)/std)@prototypes.T

def metric(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102)
    tp=np.diag(cm); denom=cm.sum(0)+cm.sum(1)
    return dict(accuracy=float((p==y).mean()),macro_f1=float(np.divide(2*tp,denom,out=np.zeros(102),where=denom!=0).mean()),
                per_class_accuracy=(tp/np.maximum(cm.sum(1),1)).tolist())

def main():
    assert not (RUN/'artifacts/predictions.npz').exists(), 'refuse result overwrite'
    role_paths=dict(source=('NetworkDrift/train.npz',20),valid=('NetworkDrift/valid.npz',5),
                    jp=('NetworkDrift/JP.npz',20),subpage=('BehaviorDrift/subpage.npz',20))
    xs={}; ys={}; manifest={}; hashes={}; direction_hashes={}
    for role,(relative,cap) in role_paths.items():
        path=DATA/relative; x=mmap_x(path)
        with zipfile.ZipFile(path) as z:
            with z.open('y.npy') as f: raw_y=np.load(f,allow_pickle=False)
        assert np.isfinite(raw_y).all() and np.equal(raw_y,raw_y.astype(int)).all()
        y=raw_y.astype(int); assert set(y)==set(range(102))
        rng=np.random.default_rng(1729)
        selected=np.concatenate([rng.permutation(np.flatnonzero(y==c))[:cap] for c in range(102)])
        manifest[role]=dict(path=relative,rows=selected.tolist(),count=len(selected),
                            label_use='stratified sampling and scoring only, except source supervised prototype')
        ys[role]=y[selected]; feats={k:[] for k in NAMES}; hashes[role]=set(); direction_hashes[role]=set()
        for i in selected:
            row=x[i]; hashes[role].add(hashlib.sha256(row.tobytes()).hexdigest())
            direction_hashes[role].add(hashlib.sha256(np.sign(row[:L]).astype(np.int8).tobytes()).hexdigest())
            for key,value in features(row).items(): feats[key].append(value)
        xs[role]={key:np.stack(value) for key,value in feats.items()}
        print('features',role,len(selected),flush=True)
    checks=[]
    for a in role_paths:
        for b in role_paths:
            if a>=b: continue
            entry=dict(a=a,b=b,full_hash_overlap=len(hashes[a]&hashes[b]),direction_hash_overlap=len(direction_hashes[a]&direction_hashes[b]))
            assert entry['full_hash_overlap']==entry['direction_hash_overlap']==0,entry
            checks.append(entry)
    (RUN/'artifacts/sampling_and_isolation.json').write_text(json.dumps(dict(manifest=manifest,checks=checks),indent=2))
    models={key:fit(xs['source'][key],ys['source']) for key in NAMES}
    preds={}; scores={}; model_arrays={}
    for key,model in models.items():
        for n,a in zip(['mean','std','prototype'],model): model_arrays[key+'_'+n]=a
    np.savez_compressed(RUN/'artifacts/source_fit.npz',**model_arrays)
    for role in ['valid','jp','subpage']:
        for key in NAMES:
            scores[role+'_'+key]=predict_scores(xs[role][key],models[key])
        scores[role+'_all_equal']=sum(scores[role+'_'+key] for key in NAMES)/len(NAMES)
        for key in NAMES+['all_equal']: preds[role+'_'+key]=scores[role+'_'+key].argmax(1)
    # All predictions sealed before any metric/error computation.
    np.savez_compressed(RUN/'artifacts/predictions.npz',**preds)
    np.savez_compressed(RUN/'artifacts/scores.npz',**scores)
    seals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'config.json',RUN/'PLAN.md',Path(__file__),RUN/'artifacts/predictions.npz',RUN/'artifacts/source_fit.npz']}
    (RUN/'artifacts/prediction_seal.json').write_text(json.dumps(seals,indent=2))
    metrics=[]; complement=[]
    for role in ['valid','jp','subpage']:
        base=preds[role+'_packet']==ys[role]
        for key in NAMES+['all_equal']:
            p=preds[role+'_'+key]; m=metric(ys[role],p)
            metrics.append(dict(role=role,view=key,n=len(p),**m))
            good=p==ys[role]
            complement.append(dict(role=role,view=key,repairs=int((good&~base).sum()),damages=int((~good&base).sum()),
                                   oracle_union_accuracy=float((good|base).mean()),net_repair=int(good.sum()-base.sum())))
    with (RUN/'artifacts/metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['role','view','n','accuracy','macro_f1']); w.writeheader()
        w.writerows({k:v for k,v in row.items() if k!='per_class_accuracy'} for row in metrics)
    summary=dict(metrics=metrics,complementarity=complement,dimensions={k:xs['source'][k].shape[1] for k in NAMES},
                 config='one fixed seed, source-only cosine prototypes; no target tuning',neural_training=0,
                 limitations=['small source support','no learned encoder','no multi-seed confirmation','oracle union uses truth for posthoc only, not deployable routing','JP/subpage are now development performance exposed'])
    (RUN/'artifacts/summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
