"""Independent scalar metrics and artifact integrity; no fitting."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]
DATA=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
m=json.loads((RUN/'artifacts/manifest.json').read_text())
old=json.loads((ROOT/'runs/20260922T011309Z_token_effect_diagnostic_d0b9386f/artifacts/sampling_and_isolation.json').read_text())['manifest']
for role,item in m['sampling'].items():
    assert item['rows']==old[role]['rows'] and item['path']==old[role]['path']
assert hashlib.sha256((RUN/'artifacts/predictions.npz').read_bytes()).hexdigest()==m['prediction_sha256']
for path,digest in json.loads((RUN/'artifacts/seal.json').read_text()).items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
preds=np.load(RUN/'artifacts/predictions.npz',allow_pickle=False)
rows=list(csv.DictReader((RUN/'artifacts/metrics.csv').open()))
assert len(rows)==36 and len(preds.files)==36
labels={}
for role in ['valid','jp','subpage']:
    item=m['sampling'][role]
    with np.load(DATA/item['path'],allow_pickle=False) as z: labels[role]=z['y'][item['rows']].astype(int)
for row in rows:
    y=labels[row['role']]; p=preds[row['role']+'_'+row['view']]
    assert p.shape==y.shape and ((p>=0)&(p<102)).all()
    acc=sum(int(a==b) for a,b in zip(y,p))/len(y)
    f1=[]
    for c in range(102):
        tp=sum(int(a==c and b==c) for a,b in zip(y,p))
        denominator=sum(int(a==c)+int(b==c) for a,b in zip(y,p))
        f1.append(2*tp/denominator if denominator else 0.)
    assert abs(acc-float(row['accuracy']))<1e-12
    assert abs(sum(f1)/102-float(row['macro_f1']))<1e-12
history=json.loads((RUN/'artifacts/history.json').read_text())
for name,h in history.items():
    assert len(h)==15
    assert max(h,key=lambda e:e['macro_f1'])['epoch']==m['best_epochs'][name]
out={'errors':0,'metric_rows':len(rows),'predictions':sum(preds[k].size for k in preds.files),'sampling_matches':True,'seals_match':True,'selection_verified':True}
(RUN/'artifacts/integrity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))


