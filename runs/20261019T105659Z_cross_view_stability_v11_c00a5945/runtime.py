import hashlib,json,os,subprocess,fcntl
from pathlib import Path
import numpy as np
import torch
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
def setup(seed):
    assert C['status']=='frozen';torch.set_num_threads(C['budget']['threads']);torch.manual_seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
def verify():
    assert C['status']=='frozen'
    for n,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/n)==h,n
def status(state,summary):
    subprocess.run([os.sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    with (ROOT/'.status.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);p=ROOT/'STATUS.md';prefix=f'跨视图稳定性 v11：`{RUN.name}`。';ls=[s for s in p.read_text().splitlines() if not s.startswith(prefix)];ls.insert(1,prefix+summary);ls=['当前研究动作：v11 '+summary+' 去KL及更高分位拒绝分支仍终止，旧封存不变。' if s.startswith('当前研究动作：') else s for s in ls];p.write_text('\n'.join(ls)+'\n')
def read(p):
    with np.load(p) as a:return {k:a[k] for k in a.files}
def valid():
    d=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True);assert len(d['source']['rows'])==18553 and len(d['valid']['rows'])==2160
    return d['valid']
def target(name):
    d=torch.load(ROOT/C['diagnostic_run']/'artifacts'/f'data_{name}.pt',map_location='cpu',weights_only=True);assert 'labels' not in d['reference'] and set(d['reference']['rows'].tolist()).isdisjoint(d['query']['rows'].tolist());return d
def roles():
    yield 'valid',valid()
    for name in C['slices']:
        d=target(name)
        for role in ['reference','query']:yield f'{name}_{role}',d[role]
def old_logits(seed,arm,role):
    if role=='valid':return read(ROOT/C['v10_run']/'artifacts'/f'sourcevalid_{seed}_{arm}.npz')
    name,r=role.rsplit('_',1);return read(ROOT/C['v10_run']/'artifacts'/f'{r}_{seed}_{name}_{arm}.npz')
