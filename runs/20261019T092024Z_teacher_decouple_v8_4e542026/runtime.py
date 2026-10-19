"""Explicit run-local model, immutable inputs, atomic outputs and metric helpers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from model import MultiViewModel

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(name, value):
    p = RUN / 'artifacts' / name
    t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    t.replace(p)

def status(state, summary):
    subprocess.run([os.sys.executable, str(ROOT/'scripts/experiment.py'), 'status', '--id', RUN.name,
                    '--state', state, '--summary', summary], check=True)
    import fcntl
    with (ROOT/'.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p=ROOT/'STATUS.md'; lines=p.read_text().splitlines(); prefix=f'Teacher解耦 v8：`{RUN.name}`。'
        lines=[s for s in lines if not s.startswith(prefix)]
        lines.insert(1,prefix+summary);p.write_text('\n'.join(lines)+'\n')

def setup():
    assert C['status']=='frozen'
    torch.set_num_threads(C['budget']['threads'])
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    np.random.seed(20261019); torch.manual_seed(20261019)

def verify():
    for rel,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():
        assert sha(ROOT/rel)==h, rel

def load_model(seed):
    reports=json.loads((ROOT/C['source_run']/'artifacts/reports.json').read_text())
    r=next(x for x in reports if x['seed']==seed)
    p=ROOT/C['source_run']/'checkpoints'/f'pure_source_wide_bn_{seed}.pth'
    assert sha(p)==r['checkpoint_sha256']
    m=MultiViewModel('wide_bn',102).cuda().eval()
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    return m.requires_grad_(False)

def source():
    return torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True)['source']

def reference(seed):
    r=torch.load(ROOT/C['prototype_run']/'artifacts'/f'source_token_reference_{seed}.pt',map_location='cpu',weights_only=True)
    assert len(r['source_rows'])==18553 and r['class_prototypes'].shape==(102,110)
    return r

def load_slice(name):
    p=ROOT/C['diagnostic_run']/'artifacts'/f'data_{name}.pt'
    audit=json.loads((ROOT/C['diagnostic_run']/'artifacts/data_audit.json').read_text())
    assert audit['passed'] and sha(p)==audit['slices'][name]['cache_sha256']
    d=torch.load(p,map_location='cpu',weights_only=True)
    # Labels physically excluded from reference optimization API.
    return {k:v for k,v in d['reference'].items() if k in ('timestamps','tam','rows')},d['query']

def configure(m,condition):
    m.eval().requires_grad_(False)
    if condition=='tent_bn':
        for mod in m.modules():
            if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):
                mod.requires_grad_(True);mod.track_running_stats=False
                mod.running_mean=mod.running_var=mod.num_batches_tracked=None
    elif condition=='tent_entropy_gate':
        for n,mod in m.named_modules():
            if n.startswith('head.') and n.endswith(('norm1','norm2')) and isinstance(mod,torch.nn.LayerNorm):mod.requires_grad_(True)
    if condition=='tent_entropy_gate':assert sum(p.numel() for p in m.parameters() if p.requires_grad)==1024
    return [n for n,p in m.named_parameters() if p.requires_grad]

@torch.no_grad()
def predict(m,d):
    m.eval(); z=[]
    for i in range(0,len(d['rows']),128):
        z.append(m(d['timestamps'][i:i+128].cuda(),d['tam'][i:i+128].cuda()).cpu().numpy())
    z=np.concatenate(z);assert np.isfinite(z).all() and z.shape==(len(d['rows']),102)
    return z

def entropy(z):return -(z.softmax(-1)*z.log_softmax(-1)).sum(-1)

def metrics(y,p,base):
    from sklearn.metrics import f1_score
    correct=base==y; wrong=~correct
    return {'accuracy':float((y==p).mean()),'macro_f1':float(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)),
            'nar':float((p[correct]!=y[correct]).mean()) if correct.any() else None,
            'recovery':float((p[wrong]==y[wrong]).mean()) if wrong.any() else None}
