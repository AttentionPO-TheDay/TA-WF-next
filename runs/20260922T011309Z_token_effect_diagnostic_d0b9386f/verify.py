import csv
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

RUN=Path(__file__).resolve().parent; A=RUN/'artifacts'
DATA=Path(json.loads((RUN.parents[1]/'configs/datasets.json').read_text())['data_root'])
manifest=json.loads((A/'sampling_and_isolation.json').read_text())['manifest']
summary=json.loads((A/'summary.json').read_text())
sealed=json.loads((A/'prediction_seal.json').read_text())
for name,h in sealed.items():
    p=A/name if (A/name).exists() else RUN/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==h
preds=np.load(A/'predictions.npz'); scores=np.load(A/'scores.npz')
truth={}
for role in ['valid','jp','subpage']:
    with zipfile.ZipFile(DATA/manifest[role]['path']) as z:
        with z.open('y.npy') as f: y=np.load(f,allow_pickle=False)
    truth[role]=y[manifest[role]['rows']].astype(int)
for r in summary['metrics']:
    key=r['role']+'_'+r['view']; y=truth[r['role']]; p=preds[key]
    assert np.isfinite(scores[key]).all()
    assert np.array_equal(p,scores[key].argmax(1))
    f1=[]
    for c in range(102):
        tp=np.count_nonzero((y==c)&(p==c)); fp=np.count_nonzero((y!=c)&(p==c)); fn=np.count_nonzero((y==c)&(p!=c))
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    assert abs(np.mean(p==y)-r['accuracy'])<1e-12
    assert abs(np.mean(f1)-r['macro_f1'])<1e-12
for role in truth:
    expected=sum(scores[role+'_'+k] for k in ['packet','exact_run','coarse_run','windows','timing'])/5
    np.testing.assert_allclose(expected,scores[role+'_all_equal'])
result=dict(errors=0,metric_rows=len(summary['metrics']),predictions_checked=sum(len(preds[k]) for k in preds.files),
    seals_verified=len(sealed),scope='independent scalar metric calculation and prediction sealing; no feature refit')
(A/'integrity.json').write_text(json.dumps(result,indent=2)); print(result)
print('complementarity',json.dumps(summary['complementarity']))
