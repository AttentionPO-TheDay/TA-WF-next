import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
rows=json.loads((R/'artifacts/metrics.json').read_text()); p=np.load(R/'artifacts/predictions.npz')
assert len(rows)==15 and len(p.files)==15 and len({r['key'] for r in rows})==15
assert all(p[r['key']].shape==(510,) for r in rows)
for r in rows:
 y=np.load(R/'artifacts/valid_inputs.npz')['labels']; q=p[r['key']]; f=[]
 for c in range(102):
  tp=int(((y==c)&(q==c)).sum()); den=int((y==c).sum()+(q==c).sum()); f.append(2*tp/den if den else 0)
 assert abs(float((q==y).mean())-r['accuracy'])<1e-12 and abs(sum(f)/102-r['macro_f1'])<1e-12
print('verified',len(rows),'rows',sum(len(p[k]) for k in p.files),'predictions')
