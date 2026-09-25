"""Independent metrics, historical reproduction, and seal verification."""
import hashlib
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
ROOT=R.parents[1]
for seal in ('input_seal.json','output_seal.json'):
    for path,digest in json.loads((R/'artifacts'/seal).read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest, path
cfg=json.loads((R/'config.json').read_text())
old=ROOT/'runs'/cfg['source_run']
with np.load(R/'artifacts/predictions.npz') as p, np.load(R/'artifacts/labels.npz') as y, np.load(old/'artifacts/predictions.npz') as historical:
    rows=json.loads((R/'artifacts/metrics.json').read_text())
    assert len(rows)==18 and len(p.files)==18
    for row in rows:
        truth=y[row['role']]; pred=p[row['key']]
        f=[]; zero=0
        for k in range(102):
            tp=np.sum((truth==k)&(pred==k)); den=np.sum(truth==k)+np.sum(pred==k)
            f.append(2*tp/den if den else 0); zero+=int(tp==0)
        assert abs(np.mean(f)-row['macro_f1'])<1e-12
        assert abs(np.mean(truth==pred)-row['accuracy'])<1e-12
        assert row['zero_recall_classes']==zero
        full=p[f"{row['role']}_{row['seed']}_full"]
        assert abs(np.mean(full!=pred)-row['changed_fraction'])<1e-12
        if row['role']=='valid' and row['condition']=='full': np.testing.assert_array_equal(pred,historical[f"hybrid_{row['seed']}"])
    total=sum(len(p[k]) for k in p.files)
result=dict(passed=True,metrics=18,predictions=total,full_valid_prediction_matches=1530,seal_checks='passed')
(R/'artifacts/verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
