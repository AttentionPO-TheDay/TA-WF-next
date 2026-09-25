import csv,hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parents[1]; DATA=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
m=json.loads((R/'artifacts/manifest.json').read_text())
assert not any(m['direction_overlap_counts'].values())
assert hashlib.sha256((R/'artifacts/predictions.npz').read_bytes()).hexdigest()==m['prediction_sha256']
for p,h in json.loads((R/'artifacts/seal.json').read_text()).items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
pred=np.load(R/'artifacts/predictions.npz',allow_pickle=False); rows=list(csv.DictReader((R/'artifacts/metrics.csv').open()))
assert len(rows)==72 and len(pred.files)==72
ys={}
for role,item in m['sampling'].items():
 with np.load(DATA/item['path'],allow_pickle=False) as z: ys[role]=z['y'][item['rows']].astype(int)
for row in rows:
 y=ys[row['role']]; p=pred[row['role']+'_'+row['view']]; assert p.shape==y.shape and ((p>=0)&(p<102)).all()
 acc=sum(int(a==b) for a,b in zip(y,p))/len(y); f=[]
 for c in range(102):
  tp=sum(int(a==c and b==c) for a,b in zip(y,p)); den=sum(int(a==c)+int(b==c) for a,b in zip(y,p)); f.append(2*tp/den if den else 0.)
 assert abs(acc-float(row['accuracy']))<1e-12 and abs(sum(f)/102-float(row['macro_f1']))<1e-12
h=json.loads((R/'artifacts/history.json').read_text())
assert len(h)==12
for k,v in h.items(): assert len(v)==15 and max(v,key=lambda x:x['macro_f1'])['epoch']==m['best_epochs'][k]
out={'errors':0,'metric_rows':len(rows),'predictions':sum(pred[k].size for k in pred.files),'direction_overlaps':0,'seals_match':True,'selection_verified':True}
(R/'artifacts/integrity.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))

