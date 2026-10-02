"""CPU-only, bounded supervised comparison; deterministic resumable epochs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from ta_wf_next.hierarchical_transformer import HierarchicalViewTransformer
from ta_wf_next.packet_patches import packet_patch_inputs
from ta_wf_next.transformer_proto import GeneratorTokenBatch
from prepare import sha, unpack


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def atomic_torch(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def load_config():
    config=json.loads((RUN/'config.json').read_text())
    freeze=json.loads((RUN/'artifacts/freeze.json').read_text())
    if config['status']!='frozen' or config['device']!='cpu':
        raise RuntimeError('frozen CPU config required')
    assert sha(RUN/'config.json')==freeze['config_sha256']
    for path,digest in freeze['code_sha256'].items():
        assert sha(ROOT/path)==digest, 'code changed after freeze: '+path
    assert sha(RUN/'artifacts/prepared.pt')==config['prepared_sha256']
    return config,freeze['config_sha256']


def metric(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102)
    tp=cm.diagonal().astype(np.float64)
    den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*tp,den,out=np.zeros(102),where=den!=0).mean())}


def subset(batch,idx):
    return GeneratorTokenBatch(*(getattr(batch,n)[idx] for n in
        ('packet','packet_mask','runs','runs_mask','windows','windows_mask')),
        {n:v[idx] for n,v in batch.spans.items()})


def make_batch(entry,condition):
    return packet_patch_inputs(unpack(entry['original']),entry['directions'],condition=condition)


def make_model(config):
    return HierarchicalViewTransformer(**config['model'])


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def predict(model,batch,bs=64):
    model.eval();values=[]
    with torch.inference_mode():
        for start in range(0,len(batch.packet),bs):
            idx=torch.arange(start,min(start+bs,len(batch.packet)))
            values.append(model(subset(batch,idx)).argmax(1).numpy())
    return np.concatenate(values)


def train(condition,seed):
    config,config_sha=load_config()
    assert condition in config['conditions'] and seed in config['seeds']
    torch.set_num_threads(config['threads_per_worker'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
    source=make_batch(data['source'],condition);valid=make_batch(data['valid'],condition)
    y=data['source']['labels'];vy=data['valid']['labels'].numpy()
    key=f'{condition}_{seed}'
    latest_path=RUN/'checkpoints'/f'{key}_latest.pt'
    best_path=RUN/'checkpoints'/f'{key}_best.pt'
    report_path=RUN/'artifacts'/f'metrics_{key}.json'
    if report_path.exists():
        old=json.loads(report_path.read_text())
        if old.get('status')=='completed':
            assert old['config_sha256']==config_sha
            print('already completed',key,flush=True);return
        raise RuntimeError('prior stopped seed requires explicit audit; no automatic extra budget')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=make_model(config)
    initial_sha=state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    first_epoch=1;history=[];best_f1=-1.0;best_epoch=None;elapsed_base=0.0
    if latest_path.exists():
        previous=torch.load(latest_path,map_location='cpu',weights_only=False)
        assert previous['config_sha256']==config_sha and previous['initial_state_sha256']==initial_sha
        model.load_state_dict(previous['state_dict']);optimizer.load_state_dict(previous['optimizer'])
        torch.set_rng_state(previous['torch_rng']);np.random.set_state(previous['numpy_rng']);random.setstate(previous['python_rng'])
        history=previous['history'];first_epoch=previous['epoch']+1
        elapsed_base=previous['elapsed_seconds'];best_f1=previous['best_f1'];best_epoch=previous['best_epoch']
    started=time.monotonic()
    elapsed=lambda: elapsed_base+time.monotonic()-started
    parameters=sum(p.numel() for p in model.parameters())
    print(json.dumps({'key':key,'pid':os.getpid(),'device':'cpu','threads':torch.get_num_threads(),
                     'parameters':parameters,'initial_state_sha256':initial_sha,'start_epoch':first_epoch}),flush=True)
    status='completed'
    for epoch in range(first_epoch,config['epochs']+1):
        if elapsed() >= config['time_limit_seconds_per_seed']:
            status='stopped_budget';break
        epoch_started=time.monotonic();model.train();loss_total=0.0;correct=0
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+1000+epoch))
        for start in range(0,len(y),config['batch_size']):
            idx=order[start:start+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            logits=model(subset(source,idx));loss=F.cross_entropy(logits,y[idx])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('nonfinite gradient')
            optimizer.step()
            loss_total+=float(loss.detach())*len(idx);correct+=int((logits.argmax(1)==y[idx]).sum())
        record={'epoch':epoch,'train_loss':loss_total/len(y),'online_train_accuracy':correct/len(y)}
        if epoch%config['eval_every']==0:
            sp=predict(model,source,config['batch_size']);vp=predict(model,valid,config['batch_size'])
            scores=metric(vy,vp)
            record.update({'source':metric(y.numpy(),sp),'valid':scores,'valid_predicted_classes':int(len(np.unique(vp)))})
            if scores['macro_f1']>best_f1:
                best_f1=scores['macro_f1'];best_epoch=epoch
                atomic_torch(best_path,{'state_dict':model.state_dict(),'epoch':epoch,'condition':condition,'seed':seed,
                              'config_sha256':config_sha,'initial_state_sha256':initial_sha})
        record['epoch_seconds']=time.monotonic()-epoch_started;record['elapsed_seconds']=elapsed()
        history.append(record)
        atomic_torch(latest_path,{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                     'epoch':epoch,'history':history,'best_f1':best_f1,'best_epoch':best_epoch,
                     'elapsed_seconds':elapsed(),'initial_state_sha256':initial_sha,'config_sha256':config_sha})
        atomic_json(RUN/'artifacts'/f'history_{key}.json',{'history':history,'best_epoch':best_epoch,'config_sha256':config_sha})
        atomic_json(RUN/'artifacts'/f'progress_{key}.json',{'epoch':epoch,'elapsed_seconds':elapsed(),'best_epoch':best_epoch,
                    'best_valid_macro_f1':best_f1,'status':'running','pid':os.getpid()})
        if epoch%config['eval_every']==0:
            print(json.dumps({'key':key,**record}),flush=True)
    if not history or history[-1]['epoch']<config['epochs']:status='stopped_budget'
    report={'condition':condition,'seed':seed,'status':status,'epochs_completed':history[-1]['epoch'] if history else 0,
            'best_epoch':best_epoch,'parameters':parameters,'elapsed_seconds':elapsed(),
            'config_sha256':config_sha,'initial_state_sha256':initial_sha,'device':'cpu','threads':torch.get_num_threads()}
    if best_epoch is not None:
        best=torch.load(best_path,map_location='cpu',weights_only=True)
        model.load_state_dict(best['state_dict'])
        sp=predict(model,source,config['batch_size']);vp=predict(model,valid,config['batch_size'])
        report.update({'source':metric(y.numpy(),sp),'valid':metric(vy,vp),'valid_predicted_classes':int(len(np.unique(vp)))})
        np.savez_compressed(RUN/'artifacts'/f'predictions_{key}.npz',source=sp,valid=vp)
    atomic_json(report_path,report)
    atomic_json(RUN/'artifacts'/f'progress_{key}.json',report)
    print(json.dumps(report),flush=True)
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True);parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args();train(args.condition,args.seed)
