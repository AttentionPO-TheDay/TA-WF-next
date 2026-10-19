import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import csv,fcntl,hashlib,json,subprocess,time
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
def save(n,o):
 p=RUN/'artifacts'/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def csvout(n,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (RUN/'artifacts'/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def verify():
 assert C['status']=='frozen'
 for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
def status(state,msg):
 subprocess.run([os.sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',msg],check=True)
 with (ROOT/'.status.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);p=ROOT/'STATUS.md';prefix=f'低风险目标/稳健/reference v15：`{RUN.name}`。'
  ls=[s for s in p.read_text().splitlines() if not s.startswith(prefix)];ls.insert(1,prefix+msg)
  ls=['当前研究动作：v15 '+msg+' WTT/AWF关闭，旧封存不变。' if s.startswith('当前研究动作：') else s for s in ls];p.write_text('\n'.join(ls)+'\n')
