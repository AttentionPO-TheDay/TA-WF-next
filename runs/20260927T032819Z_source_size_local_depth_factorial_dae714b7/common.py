"""Run-local helpers, explicitly adapted from the preceding CPU representation run.

Only this project's src is imported. No old project checkpoints/training code.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from ta_wf_next.learnable_generator import GeneratorClassifier
from ta_wf_next.transformer_proto import GeneratorTokenBatch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def atomic_torch(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)

def config_read(frozen=True):
    c=json.loads((RUN/'config.json').read_text())
    assert c['device']=='cpu' and not c['gpu_access'] and not c['future_access']
    if frozen:
        assert c['status']=='frozen','draft config cannot train'
        f=json.loads((RUN/'artifacts/freeze.json').read_text())
        assert sha(RUN/'config.json')==f['config_sha256'] and sha(RUN/'PLAN.md')==f['plan_sha256']
        for p,h in f['code_sha256'].items():assert sha(ROOT/p)==h,p
        assert sha(ROOT/'configs/datasets.json')==f['datasets_sha256']
        assert sha(ROOT/c['prepared_input'])==f['prepared_sha256']
        assert sha(ROOT/c['sampling_manifest'])==f['manifest_sha256']
        for path,h in c['anchor_sha256'].items():assert sha(ROOT/path)==h,path
    return c

def load_data(config):
    data=torch.load(ROOT/config['prepared_input'],map_location='cpu',weights_only=False)
    assert set(data)=={'source20','source80','source150','valid'}
    return data

@dataclass
class Batch:
    original:GeneratorTokenBatch
    directions:torch.Tensor
    observed:torch.Tensor
    def __len__(self):return len(self.directions)
    def subset(self,idx):
        original=GeneratorTokenBatch(*(getattr(self.original,n)[idx] for n in
            ('packet','packet_mask','runs','runs_mask','windows','windows_mask')),
            {n:v[idx] for n,v in self.original.spans.items()})
        return Batch(original,self.directions[idx],self.observed[idx])
    def logits(self,model):return model(self.original,self.directions,self.observed)
    def tokens(self,model):return model.generator(self.directions,self.observed,self.original.packet)

def make_batch(entry):
    return Batch(GeneratorTokenBatch(**entry['original']),entry['directions'],entry['directions']!=0)

def condition_config(config,name):
    return next(row for row in config['conditions'] if row['id']==name)

def make_model(config,name):
    c=condition_config(config,name)
    model=GeneratorClassifier(pool='attention',generator_trainable=True,dropout=config['dropout'])
    if c['local_layers']==2:
        base_seed=torch.initial_seed()
        for i,encoder in enumerate(model.classifier.local_encoders.values()):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed+50000+i)
                layer=torch.nn.TransformerEncoderLayer(d_model=52,nhead=4,dim_feedforward=104,
                    dropout=config['dropout'],activation='gelu',batch_first=True,norm_first=True)
            encoder.layers.append(layer);encoder.num_layers=2
    else:assert c['local_layers']==1
    return model

def shared_state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if '.local_encoders.' in name and '.layers.1.' in name:continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def make_optimizer(model,config,name):
    return torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])

def index_stream(n,seed,c):
    total=c['optimizer_steps']*c['batch_size'];pieces=[];have=0;cycle=0
    while have<total:
        pieces.append(torch.randperm(n,generator=torch.Generator().manual_seed(seed+9000+cycle)))
        have+=n;cycle+=1
    return torch.cat(pieces)[:total].reshape(c['optimizer_steps'],c['batch_size'])

def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def metric(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102)
    tp=cm.diagonal().astype(float);den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*tp,den,out=np.zeros(102),where=den!=0).mean())}

def predict(model,batch,bs=64):
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(batch),bs):result.append(batch.subset(slice(start,start+bs)).logits(model).argmax(1).numpy())
    return np.concatenate(result)

def parameter_counts(model):
    return {'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'generator_total':sum(p.numel() for p in model.generator.parameters()),
            'generator_trainable':sum(p.numel() for p in model.generator.parameters() if p.requires_grad)}


def lr_for_step(c,step):
    for segment in c["lr_schedule"]:
        if segment["first_step"]<=step<=segment["last_step"]:return segment["lr"]
    raise ValueError("step outside frozen schedule")
