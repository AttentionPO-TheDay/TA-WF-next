import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    p = RUN / 'artifacts' / name
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def read(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def verify():
    assert C['status'] == 'frozen'
    for p, h in json.loads((RUN / 'artifacts/freeze.json').read_text()).items():
        assert sha(ROOT / p) == h, p


def setup(seed):
    import torch
    torch.set_num_threads(C['budget']['threads'])
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def status(state, msg):
    subprocess.run([os.sys.executable, str(ROOT / 'scripts/experiment.py'), 'status', '--id', RUN.name, '--state', state, '--summary', msg], check=True)
    with (ROOT / '.status.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        p = ROOT / 'STATUS.md'
        prefix = f'源域候选确认 v14：`{RUN.name}`。'
        lines = [l for l in p.read_text().splitlines() if not l.startswith(prefix)]
        lines.insert(1, prefix + msg)
        lines = ['当前研究动作：v14 ' + msg + ' 目标标签不拟合；旧封存不变，WTT/AWF关闭。' if l.startswith('当前研究动作：') else l for l in lines]
        p.write_text('\n'.join(lines) + '\n')


def csvout(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (RUN / 'artifacts' / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def selected(prepared, role):
    n = C['source_selection']['train_per_class' if role == 'train' else 'valid_per_class']
    d = prepared['source' if role == 'train' else 'valid']
    y, rows = d['labels'].numpy(), d['rows'].numpy()
    indices = []
    for label in range(102):
        ix = np.flatnonzero(y == label)
        ix = sorted(ix, key=lambda i: hashlib.sha256(f'v14|{role}|{int(rows[i])}'.encode()).digest())
        assert len(ix) >= n
        indices.extend(ix[:n])
    return np.array(indices, dtype=np.int64)
