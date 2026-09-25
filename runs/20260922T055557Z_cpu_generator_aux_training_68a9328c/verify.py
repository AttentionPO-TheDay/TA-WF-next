import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parents[1]
for f in ['pretraining_seal.json','output_seal.json']:
 for p,h in json.loads((R/'artifacts'/f).read_text()).items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
m=json.loads((R/'artifacts/manifest.json').read_text())
assert not set(m['direction_hashes']['source'])&set(m['direction_hashes']['valid'])
data=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
with np.load(data/m['sampling']['valid']['path'],allow_pickle=False) as z: y=z['y'][m['sampling']['valid']['rows']].astype(int)
pred=np.load(R/'artifacts/predictions.npz'); rows=json.loads((R/'artifacts/metrics.json').read_text()); h=json.loads((R/'artifacts/history.json').read_text())
assert len(rows)==len(pred.files)==9
for row in rows:
 k=row['condition']; p=pred[k]; assert p.shape==y.shape and ((p>=0)&(p<102)).all()
 acc=sum(int(a==b) for a,b in zip(y,p))/len(y); f=[]
 for c in range(102):
  tp=sum(int(a==c and b==c) for a,b in zip(y,p)); den=sum(int(a==c)+int(b==c) for a,b in zip(y,p)); f.append(2*tp/den if den else 0)
 assert abs(acc-row['accuracy'])<1e-12 and abs(sum(f)/102-row['macro_f1'])<1e-12
 assert len(h[k])==15 and max(h[k],key=lambda a:a['macro_f1'])['epoch']==row['best_epoch']
 assert row['inference_parameters']==108326
for seed in [1729,3407,2026]:
 assert len({r['initial_hash'] for r in rows if r['condition'].endswith(str(seed))})==1
q={r['condition']:r for r in rows}; out={}
for c in ['baseline','window_aux','run_aux']:
 a=np.array([[q[f'{c}_{s}'][k]*100 for k in ['accuracy','macro_f1']] for s in [1729,3407,2026]])
 b=np.array([[q[f'baseline_{s}'][k]*100 for k in ['accuracy','macro_f1']] for s in [1729,3407,2026]])
 out[c]=dict(values=a.tolist(),mean=a.mean(0).tolist(),sd=a.std(0,ddof=1).tolist(),paired_delta=(a-b).tolist(),delta_mean=(a-b).mean(0).tolist(),candidate=bool(((a-b)[:,1]>0).all() and (a-b)[:,0].mean()>0))
(R/'artifacts/paired_summary.json').write_text(json.dumps(out,indent=2))
(R/'artifacts/integrity.json').write_text(json.dumps(dict(errors=0,metrics=9,predictions=4590,paired_initialization=True,inference_parameters=108326),indent=2))
print(json.dumps(out,indent=2))
