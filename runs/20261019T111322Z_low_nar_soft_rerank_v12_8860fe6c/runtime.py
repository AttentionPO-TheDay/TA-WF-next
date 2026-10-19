import hashlib,json,os,subprocess,fcntl
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):
    with np.load(p) as a:return {k:a[k] for k in a.files}
def save(n,obj):
    p=RUN/'artifacts'/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def verify():
    assert C['status']=='frozen'
    for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
def status(state,summary):
    subprocess.run([os.sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    with (ROOT/'.status.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);p=ROOT/'STATUS.md';prefix=f'低NAR软重排序 v12：`{RUN.name}`。';ls=[s for s in p.read_text().splitlines() if not s.startswith(prefix)];ls.insert(1,prefix+summary);ls=['当前研究动作：v12 '+summary+' 仅v9固定候选缓存重排；旧封存不变，去KL/高分位拒绝分支仍终止。' if s.startswith('当前研究动作：') else s for s in ls];p.write_text('\n'.join(ls)+'\n')
def views(partition,name=None):
    zs=[];rows=None;labels=None
    for seed in C['seeds']:
        data=[]
        for view in C['views']:
            if view=='original':p=ROOT/C['v10_run']/'artifacts'/(f'sourcevalid_{seed}_direction_iat.npz' if partition=='valid' else f'query_{seed}_{name}_direction_iat.npz')
            else:p=ROOT/C['v11_run']/'artifacts'/f'logits_{seed}_direction_iat_{"valid" if partition=="valid" else name+"_query"}_{view}.npz'
            a=read(p)
            if rows is None:rows=a['rows'];labels=a['labels']
            assert np.array_equal(rows,a['rows']) and np.array_equal(labels,a['labels']);data.append(a['logits'].astype(np.float64))
        zs.append(np.stack(data))
    return np.stack(zs),rows,labels
def state(seed,name,partition):
    a=read(ROOT/C['v9_run']/'artifacts'/f'{partition}_{seed}_{name}_smooth_balance.npz');f=read(ROOT/C['v9_run']/'artifacts'/f'{partition}_{seed}_{name}_frozen.npz');old=read(ROOT/C['v11_run']/'artifacts'/f'scores_{partition}_{seed}_{name}_direction_iat_v9_smooth.npz')
    b=f['logits'].argmax(-1);c=a['logits'].argmax(-1)
    for v in [a,f,old]:assert np.array_equal(a['rows'],v['rows']) and np.array_equal(a['labels'],v['labels'])
    assert np.array_equal(old['base'],b) and np.array_equal(old['candidates'],c) and np.array_equal(old['baseline'],a['margin_gain'])
    return dict(rows=a['rows'],labels=a['labels'],base=b,candidate=c,margin=a['margin_gain'].astype(np.float64),old_min=old['joint_support'].astype(np.float64))
