import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; cfg=json.loads((R/'config.json').read_text()); y=np.load(R/'artifacts/labels.npz')['valid'] if (R/'artifacts/labels.npz').exists() else None
rows=json.loads((R/'artifacts/metrics.json').read_text()); p=np.load(R/'artifacts/predictions.npz'); assert len(rows)==9 and len(p.files)==9
assert set(p.files)=={r['key'] for r in rows}
print('verified rows',len(rows),'predictions',sum(len(p[k]) for k in p.files))
