import json, hashlib
from pathlib import Path
import numpy as np, torch
R=Path(__file__).resolve().parent
S=json.loads((R/'artifacts/scores.json').read_text())
assert len(S)==36 and {r['condition'] for r in S}=={'frozen','global','drift_selective','random_selective'}
for r in S:
    assert r['rows']>0 and np.isfinite([r[k] for k in ('accuracy','macro_f1','negative_adaptation_rate','recovery_rate')]).all()
    p=R/'artifacts'/f"pred_{r['seed']}_{r['role']}_{r['condition']}.npz"
    with np.load(p) as z:
        assert len(z['rows'])==r['rows'] and len(z['labels'])==r['rows'] and len(z['predictions'])==r['rows']
        assert np.array_equal(z['predictions'],z['predictions'].astype(np.int64))
for p in (R/'checkpoints').glob('adapter_*.pt'):
    d=torch.load(p,map_location='cpu',weights_only=True); assert set(d['state_dict'])=={'norm.weight','norm.bias','down.weight','down.bias','up.weight','up.bias'}
(R/'artifacts/audit.json').write_text(json.dumps({'passed':True,'rows':len(S),'prediction_files':len(list((R/'artifacts').glob('pred_*.npz'))),'adapter_files':len(list((R/'checkpoints').glob('adapter_*.pt')))},indent=2)+'\n')
