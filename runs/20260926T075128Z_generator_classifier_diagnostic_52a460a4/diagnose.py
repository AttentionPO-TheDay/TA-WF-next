"""Frozen checkpoint inference and prespecified source-only ridge diagnostics."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import torch
from ta_wf_next.learnable_generator import GeneratorClassifier
from ta_wf_next.transformer_proto import GeneratorTokenBatch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def config_read():
    c=json.loads((RUN/'config.json').read_text())
    if c['status']!='frozen' or c['device']!='cpu':raise RuntimeError('frozen CPU config required')
    freeze=json.loads((RUN/'artifacts/freeze.json').read_text());assert sha(RUN/'config.json')==freeze['config_sha256']
    for path,digest in freeze['code_sha256'].items():assert sha(ROOT/path)==digest,path+' changed'
    assert sha(ROOT/c['prepared_input'])==c['prepared_input_sha256']
    assert sha(ROOT/c['sampling_manifest'])==c['sampling_manifest_sha256']
    for path,digest in c['anchor_sha256'].items():assert sha(ROOT/path)==digest,path+' anchor changed'
    return c

def batch_slice(entry,start,end):
    old=entry['original']
    return GeneratorTokenBatch(*(old[n][start:end] for n in ('packet','packet_mask','runs','runs_mask','windows','windows_mask')),
                               {n:v[start:end] for n,v in old['spans'].items()})

def metrics(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102)
    tp=cm.diagonal().astype(float);den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*tp,den,out=np.zeros(102),where=den>0).mean())},cm

def lambda_key(value):return str(value).replace('.','p')

def ridge_fit(xs,xv,y,regularization):
    """Solve mean squared error + lambda L2, never use validation labels."""
    mu=xs.mean(0);sigma=xs.std(0);scale=np.where(sigma<1e-6,1.,sigma)
    z=(xs-mu)/scale;v=(xv-mu)/scale
    targets=np.eye(102,dtype=np.float64)[y];prior=targets.mean(0);centered=targets-prior
    gram=z@z.T/len(y)
    outputs=[]
    for lam in regularization:
        system=gram.copy();system.flat[::len(y)+1]+=lam
        dual=cho_solve(cho_factor(system,lower=True,check_finite=False,overwrite_a=True),centered,check_finite=False)
        weights=z.T@dual/len(y)
        outputs.append((lam,weights,prior,(z@weights+prior).argmax(1),(v@weights+prior).argmax(1)))
    return mu,scale,outputs


def worker(condition,seed):
    c=config_read();assert condition in c['conditions'] and seed in c['seeds']
    torch.set_num_threads(c['threads_per_worker']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.monotonic();key=f'{condition}_{seed}';out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
    prior=ROOT/c['prior_run'];anchor=prior/'checkpoints'/f'{key}_best.pt'
    assert sha(anchor)==c['anchor_sha256'][str(anchor.relative_to(ROOT))]
    data=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False);assert set(data)=={'source','valid'}
    state=torch.load(anchor,map_location='cpu',weights_only=True)
    model=GeneratorClassifier(pool='attention',generator_trainable=condition!='C_local_attention_frozen').eval()
    # Preserve original parameter flags for CPU parity; inference_mode disables gradients.
    model.load_state_dict(state['state_dict']);before={n:v.detach().clone() for n,v in model.state_dict().items()}
    features={};old_prediction={};true={};original_scores={};full_checks=0
    old_metrics=json.loads((prior/'artifacts'/f'metrics_{key}.json').read_text())
    assert old_metrics['best_epoch']==state['epoch']
    saved=np.load(prior/'artifacts'/f'predictions_{key}.npz',allow_pickle=False)
    for role in ('source','valid'):
        entry=data[role];pieces=[];predictions=[]
        with torch.inference_mode():
            for start in range(0,len(entry['directions']),c['batch_size']):
                end=min(start+c['batch_size'],len(entry['directions']));batch=batch_slice(entry,start,end)
                directions=entry['directions'][start:end];observed=directions!=0
                tokens=model.generator(directions,observed,batch.packet)
                logits=model.classifier(replace(batch,packet=tokens))
                pieces.append(tokens.numpy());predictions.append(logits.argmax(1).numpy())
        tokens=np.concatenate(pieces);mask=entry['original']['packet_mask'].numpy().astype(np.float32)
        packet=np.concatenate((tokens.reshape(len(tokens),-1),mask),axis=1)
        extra=np.concatenate([entry['original'][name].numpy().reshape(len(tokens),-1).astype(np.float32)
                              for name in ('runs','runs_mask','windows','windows_mask')],axis=1)
        assert packet.shape[1]==5300 and packet.shape[1]+extra.shape[1]==6540
        features[role]={'packet_tokens':packet,'all_views':np.concatenate((packet,extra),axis=1)}
        true[role]=entry['labels'].numpy();old_prediction[role]=np.concatenate(predictions)
        assert np.array_equal(old_prediction[role],saved[role]),'original prediction mismatch'
        scores,cm=metrics(true[role],old_prediction[role]);original_scores[role]=scores
        for metric,value in scores.items():assert abs(value-old_metrics[role][metric])<1e-12
        np.save(out/f'original_confusion_{role}.npy',cm);full_checks+=1
    saved.close()
    assert all(torch.equal(before[n],v) for n,v in model.state_dict().items()),'network changed during probe'
    np.savez_compressed(out/'features.npz',**{f'{role}_{kind}':features[role][kind] for role in features for kind in features[role]})
    np.savez_compressed(out/'original_predictions.npz',**old_prediction)
    rows=[]
    for kind in c['probe_kinds']:
        xs=features['source'][kind].astype(np.float64);xv=features['valid'][kind].astype(np.float64)
        mu,scale,fits=ridge_fit(xs,xv,true['source'],[c['primary_ridge_lambda']]+c['sensitivity_ridge_lambdas'])
        for lam,w,b,sp,vp in fits:
            tag=f'{kind}_lambda{lambda_key(lam)}'
            np.savez_compressed(out/f'probe_{tag}.npz',mean=mu,scale=scale,weights=w,intercept=b)
            np.savez_compressed(out/f'predictions_{tag}.npz',source=sp,valid=vp)
            row={'condition':condition,'seed':seed,'best_epoch':state['epoch'],'kind':kind,'lambda':lam}
            for role,p in [('source',sp),('valid',vp)]:
                row[role],cm=metrics(true[role],p);np.save(out/f'confusion_{tag}_{role}.npy',cm)
            rows.append(row);print(json.dumps(row),flush=True)
            if time.monotonic()-started>c['time_limit_seconds_per_job']:raise TimeoutError('diagnostic job budget')
    # Separately reload the saved feature/weight files and reproduce each fit.
    from verify_probe import verify_saved
    verification=verify_saved(out,true,rows,c)
    report={'condition':condition,'seed':seed,'best_epoch':state['epoch'],'original_scores':original_scores,
            'rows':rows,'verification':verification,'original_prediction_sets_checked':full_checks,
            'network_unchanged':True,'elapsed_seconds':time.monotonic()-started,'checkpoint_sha256':sha(anchor),
            'config_sha256':sha(RUN/'config.json'),'feature_sha256':sha(out/'features.npz')}
    write_json(out/'report.json',report)
    print('COMPLETE',key,report['elapsed_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();worker(args.condition,args.seed)
