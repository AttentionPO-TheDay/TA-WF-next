from pathlib import Path
import json,hashlib
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)

def savet(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)

def config():
    c=json.loads((RUN/'config.json').read_text())
    assert c['status']=='frozen' and c['roles']==['source','valid'] and not c['future_access']
    assert not c['permissions']['valid_gradient'] and not c['permissions']['adaptation']
    return c

def verify_freeze():
    for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p

def metric(y,p):
    cm=np.bincount(y*102+p,minlength=10404).reshape(102,102);den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0).mean())}

def deterministic(device,threads):
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    if device=='cuda':
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
