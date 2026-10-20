"""Pure-source Temporal runtime; explicit run-local imports only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import numpy as np
from model import MultiViewModel

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    path = RUN / 'artifacts' / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def registry(state, summary):
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status', '--id', RUN.name,
                    '--state', state, '--summary', summary], check=True)
    import fcntl
    with (ROOT / '.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / 'STATUS.md'
        lines = path.read_text().splitlines()
        prefix = f'论文一纯净时序核心矩阵：`{RUN.name}`。'
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = prefix + summary
                break
        else:
            lines.insert(1, prefix + summary)
        path.write_text('\n'.join(lines) + '\n')


def setup():
    assert C['status'] == 'frozen' and C['id'] == RUN.name
    torch.set_num_threads(C['budget']['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def verify_freeze():
    for path, expected in json.loads((RUN / 'artifacts/freeze.json').read_text()).items():
        assert sha(ROOT / path) == expected, path


def load_model(seed):
    source = ROOT / C['source_run']
    reports = json.loads((source / 'artifacts/reports.json').read_text())
    report = next(r for r in reports if r['seed'] == seed)
    path = source / 'checkpoints' / f'pure_source_wide_bn_{seed}.pth'
    assert sha(path) == report['checkpoint_sha256']
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = MultiViewModel('wide_bn', 102).cuda()
    model.load_state_dict(ck['state_dict'], strict=True)
    return model.eval()


def load_source():
    source = torch.load(ROOT / C['source_run'] / 'artifacts/prepared.pt', map_location='cpu', weights_only=True)['source']
    assert len(source['labels']) == 18553 and len(source['labels'].unique()) == 102
    return source


def load_target(role):
    data = torch.load(ROOT / C['target_prepared'][role], map_location='cpu', weights_only=True)
    path = ROOT / C['split_source_run'] / 'artifacts' / f'split_{role}.npz'
    with np.load(path) as split:
        s, q = split['support_indices'], split['query_indices']
        assert np.array_equal(data['rows'][s].numpy(), split['support'])
        assert np.array_equal(data['rows'][q].numpy(), split['query'])
    # Only adaptation reference fields cross the optimization API boundary.
    support = {k: data[k][s] for k in ('timestamps', 'tam', 'rows')}
    query = {k: data[k][q] for k in ('timestamps', 'tam', 'rows', 'labels')}
    return support, query


def configure(model, condition):
    model.requires_grad_(False)
    if condition in ('proteus_default', 'no_pseudo_fixed'):
        model.requires_grad_(True)
        if condition == 'no_pseudo_fixed':
            model.classifier.requires_grad_(False)
    elif condition == 'tent_bn':
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.num_batches_tracked = None
    elif condition == 'tent_entropy_gate':
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm) and (name == 'final_norm' or
                    (name.startswith('head.') and name.endswith(('norm1', 'norm2')))):
                module.requires_grad_(True)
    else:
        raise ValueError(condition)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert names
    if condition == 'tent_entropy_gate':
        assert len(names) == 10 and sum(p.numel() for p in model.parameters() if p.requires_grad) == 1280
    return names


def adaptation_mode(model, condition):
    model.train()
    if condition == 'tent_entropy_gate':
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


@torch.no_grad()
def predict(model, data):
    model.eval()
    output = []
    for i in range(0, len(data['rows']), C['adaptation']['batch_size']):
        part = slice(i, i + C['adaptation']['batch_size'])
        output.append(model(data['timestamps'][part].cuda(), data['tam'][part].cuda()).cpu().numpy())
    logits = np.concatenate(output)
    assert logits.shape == (len(data['rows']), 102) and np.isfinite(logits).all()
    return logits


def metrics(y, p, base):
    assert len(y) and y.shape == p.shape == base.shape
    assert y.min() >= 0 and y.max() < 102 and p.min() >= 0 and p.max() < 102
    cm = np.bincount(y * 102 + p, minlength=102 ** 2).reshape(102, 102)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(102), where=den > 0).mean()
    correct = base == y
    return {'accuracy': float((y == p).mean()), 'macro_f1': float(f1),
            'negative_adaptation_rate': float((correct & (p != y)).sum() / correct.sum()) if correct.any() else None,
            'recovery_rate': float((~correct & (p == y)).sum() / (~correct).sum()) if (~correct).any() else None,
            'frozen_correct_count': int(correct.sum()), 'frozen_wrong_count': int((~correct).sum())}
