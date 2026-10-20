import hashlib,json,os,subprocess,fcntl
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(name,obj):
    p=RUN/'artifacts'/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def verify():
    assert C['status']=='frozen'
    for n,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/n)==h,n
def setup(seed):
    assert C['status']=='frozen'
    torch.set_num_threads(C['budget']['threads']);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
def status(state,summary):
    subprocess.run([os.sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    with (ROOT/'.status.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);p=ROOT/'STATUS.md';lines=p.read_text().splitlines();prefix=f'独立表征互补性 v10：`{RUN.name}`。';lines=[s for s in lines if not s.startswith(prefix)];lines.insert(1,prefix+summary)
        action='当前研究动作：v10'+summary+' 完全去KL及提高拒绝分位两分支仍终止，旧封存不变。'
        lines=[action if s.startswith('当前研究动作：') else s for s in lines];p.write_text('\n'.join(lines)+'\n')
def source():
    d=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True)
    assert len(d['source']['rows'])==18553 and len(d['valid']['rows'])==2160
    return d
def target(name):
    d=torch.load(ROOT/C['diagnostic_run']/'artifacts'/f'data_{name}.pt',map_location='cpu',weights_only=True)
    assert 'labels' not in d['reference'] and set(d['reference']['rows'].tolist()).isdisjoint(d['query']['rows'].tolist())
    return d
def metrics(y,p,b):
    ok=b==y;bad=~ok
    return dict(accuracy=float((p==y).mean()),macro_f1=float(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)),nar=float((p[ok]!=y[ok]).mean()) if ok.any() else 0.,recovery=float((p[bad]==y[bad]).mean()) if bad.any() else 0.)
