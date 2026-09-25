import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parents[1]
for f in ['pretraining_seal.json','output_seal.json']:
 for p,h in json.loads((R/'artifacts'/f).read_text()).items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
m=json.loads((R/'artifacts/manifest.json').read_text()); assert not set(m['direction_hashes']['source'])&set(m['direction_hashes']['valid'])
data=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
with np.load(data/m['sampling']['valid']['path'],allow_pickle=False) as z: y=z['y'][m['sampling']['valid']['rows']].astype(int)
pred=np.load(R/'artifacts/predictions.npz'); rows=json.loads((R/'artifacts/metrics.json').read_text()); history=json.loads((R/'artifacts/history.json').read_text())
assert len(rows)==len(pred.files)==6
for row in rows:
 k=row['view']; p=pred[k]; assert p.shape==y.shape and ((p>=0)&(p<102)).all()
 acc=sum(int(a==b) for a,b in zip(y,p))/len(y); f=[]
 for c in range(102):
  tp=sum(int(a==c and b==c) for a,b in zip(y,p)); den=sum(int(a==c)+int(b==c) for a,b in zip(y,p)); f.append(2*tp/den if den else 0.)
 assert abs(acc-row['accuracy'])<1e-12 and abs(sum(f)/102-row['macro_f1'])<1e-12
 assert len(history[k])==15 and max(history[k],key=lambda a:a['macro_f1'])['epoch']==row['best_epoch']
out={}
for view in ['exact','coarse']:
 values=np.array([[r['accuracy']*100,r['macro_f1']*100] for r in rows if r['view'].startswith(view)])
 out[view]=dict(values=values.tolist(),mean=values.mean(0).tolist(),sd=values.std(0,ddof=1).tolist())
(R/'artifacts/aggregate.json').write_text(json.dumps(out,indent=2))
(R/'artifacts/integrity.json').write_text(json.dumps(dict(errors=0,metric_rows=6,predictions=3060,hashes_match=True,selection_verified=True),indent=2))
print(json.dumps(out,indent=2))
