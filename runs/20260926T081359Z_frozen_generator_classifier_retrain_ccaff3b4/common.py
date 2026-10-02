"""Run-local helpers explicitly adapted from the preceding two project runs."""
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import replace
os.environ['CUDA_VISIBLE_DEVICES']=''
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from ta_wf_next.learnable_generator import GeneratorClassifier
from ta_wf_next.transformer_proto import GeneratorTokenBatch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_hash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()
def write_json(path,value):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(path)
def save(path,value):
 t=path.with_suffix(path.suffix+'.tmp');torch.save(value,t);t.replace(path)
def config_read():
 c=json.loads((RUN/'config.json').read_text())
 assert c['status']=='frozen' and c['device']=='cpu' and not c['gpu_access'] and not c['future_access']
 f=json.loads((RUN/'artifacts/freeze.json').read_text());assert sha(RUN/'config.json')==f['config_sha256']
 for p,h in f['code_sha256'].items():assert sha(ROOT/p)==h,p
 for p,h in c['anchor_sha256'].items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/c['prepared_input'])==c['prepared_input_sha256']
 assert sha(ROOT/c['sampling_manifest'])==c['sampling_manifest_sha256']
 return c
FIELDS=('packet','packet_mask','runs','runs_mask','windows','windows_mask')
def subset(batch,idx):
 return GeneratorTokenBatch(*(getattr(batch,k)[idx] for k in FIELDS),{k:v[idx] for k,v in batch.spans.items()})
def metric(y,p):
 cm=np.bincount(y*102+p,minlength=10404).reshape(102,102);tp=cm.diagonal().astype(float);d=cm.sum(0)+cm.sum(1)
 return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*tp,d,out=np.zeros(102),where=d>0).mean())}
def predict(model,batch):
 model.eval()
 with torch.inference_mode():return np.concatenate([model(subset(batch,slice(i,i+64))).argmax(1).numpy() for i in range(0,len(batch.packet),64)])
def fresh(seed):
 torch.manual_seed(seed)
 return GeneratorClassifier(pool='attention').classifier

def cache_load(path,data):
 packet=torch.load(path,map_location='cpu',weights_only=True)
 return {role:replace(GeneratorTokenBatch(**data[role]['original']),packet=packet[role]) for role in ('source','valid')}
