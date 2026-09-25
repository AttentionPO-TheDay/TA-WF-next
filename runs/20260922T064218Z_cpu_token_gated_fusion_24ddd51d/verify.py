import hashlib
import json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent; ROOT = R.parents[1]
cfg = json.loads((R / 'config.json').read_text())
for filename in ('pretraining_seal.json', 'output_seal.json'):
    for p, h in json.loads((R / 'artifacts' / filename).read_text()).items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
manifest = json.loads((R / 'artifacts/manifest.json').read_text())
assert not set(manifest['direction_hashes']['source']) & set(manifest['direction_hashes']['valid'])
with np.load(R / 'artifacts/valid_inputs.npz') as z: y = z['labels']
pred = np.load(R / 'artifacts/predictions.npz')
rows = json.loads((R / 'artifacts/metrics.json').read_text())
hist = json.loads((R / 'artifacts/history.json').read_text())
expected = {f'{a}_{s}' for a in cfg['arms'] for s in cfg['seeds']}
assert set(pred.files) == set(hist) == {r['key'] for r in rows} == expected
assert len(rows) == 15 and len(y) == 510
for row in rows:
    p = pred[row['key']]; assert p.shape == y.shape and ((p >= 0) & (p < 102)).all()
    f1 = []
    for c in range(102):
        tp = int(((y == c) & (p == c)).sum()); den = int((y == c).sum() + (p == c).sum())
        f1.append(2 * tp / den if den else 0.)
    assert abs(float((p == y).mean()) - row['accuracy']) < 1e-12
    assert abs(sum(f1) / 102 - row['macro_f1']) < 1e-12
    h = hist[row['key']]; assert len(h) == 15
    chosen = max(h, key=lambda e: e['macro_f1'])
    assert chosen['epoch'] == row['best_epoch'] and chosen['macro_f1'] == row['macro_f1']
out = {}
for arm in cfg['arms']:
    a = np.array([[r['accuracy'], r['macro_f1']] for r in rows if r['arm'] == arm]) * 100
    out[arm] = dict(values=a.tolist(), mean=a.mean(0).tolist(), sd=a.std(0, ddof=1).tolist())
pairs = {}
for a, b in [('separate_gate', 'hybrid'), ('separate_gate', 'bucket'), ('separate_gate', 'separate_fixed'), ('separate', 'hybrid'), ('separate_fixed', 'separate')]:
    delta = np.array(out[a]['values']) - np.array(out[b]['values'])
    pairs[f'{a}-{b}'] = dict(delta_pp=delta.tolist(), mean_pp=delta.mean(0).tolist(), gate=bool((delta[:, 1] > 0).all() and delta[:, 0].mean() > 0))
result = dict(arms=out, paired=pairs)
(R / 'artifacts/aggregate.json').write_text(json.dumps(result, indent=2))
(R / 'artifacts/integrity.json').write_text(json.dumps(dict(errors=0, metric_rows=15, predictions=7650, hashes=True, selection=True), indent=2))
print(json.dumps(result, indent=2))
