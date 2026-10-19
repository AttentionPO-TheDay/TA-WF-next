"""Bounded independent direction models; frozen external method-transfer batch."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import load_cached
from model import MultiViewModel
from token_alignment import AdaINToken, Moments, ResidualTokenProjector, aligned_generator, moment_alignment_loss

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())
START = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    path = RUN / 'artifacts' / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def registry(state, message):
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status', '--id', RUN.name,
                    '--state', state, '--summary', message], check=True)
    import fcntl
    with (ROOT / '.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / 'STATUS.md'
        lines = path.read_text().splitlines()
        prefix = f'论文二外部方法迁移：`{RUN.name}`。'
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = prefix + message
                break
        else:
            lines.insert(1, prefix + message)
        path.write_text('\n'.join(lines) + '\n')


def budget():
    if time.monotonic() - START > C['budget']['max_wall_seconds']:
        raise TimeoutError('external matrix wall budget exhausted')
    size = 0
    for path in RUN.rglob('*'):
        try:
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
        except FileNotFoundError:
            pass  # Other seed may have just atomically renamed its temporary.
    if size > C['budget']['max_output_gib'] * 1024 ** 3:
        raise TimeoutError('external matrix output budget exhausted')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(value, path):
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def inputs(x):
    direction = x.cuda().float()
    bins = (torch.arange(5000, device='cuda') * 1800 // 5000).expand(len(x), -1)
    counts = torch.zeros(len(x), 2, 1800, device='cuda')
    counts[:, 0].scatter_add_(1, bins, (direction > 0).float())
    counts[:, 1].scatter_add_(1, bins, (direction < 0).float())
    return direction, counts


def loader(split, shuffle=False, labels=False, seed=2026):
    # Adaptation loaders never contain reference labels.
    fields = [torch.from_numpy(split['x'])]
    if labels:
        fields.append(torch.from_numpy(split['y']))
    return DataLoader(TensorDataset(*fields), batch_size=C['batch_size'], shuffle=shuffle,
                      drop_last=False, num_workers=0, generator=torch.Generator().manual_seed(seed))


def train_source(name, source, classes, seed):
    path = RUN / 'checkpoints' / f'{name}_{seed}_source.pt'
    history_path = RUN / 'artifacts' / f'{name}_{seed}_source_history.json'
    seed_all(seed)
    model = MultiViewModel('wide_bn', classes).cuda()
    if path.exists():
        ck = torch.load(path, map_location='cpu', weights_only=True)
        assert ck['config_sha256'] == sha(RUN / 'config.json') and history_path.exists()
        model.load_state_dict(ck['state_dict'], strict=True)
        return model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=C['model']['source_learning_rate'], weight_decay=1e-4)
    history = []
    batches = loader(source, shuffle=True, labels=True, seed=seed)
    for epoch in range(C['model']['source_epochs']):
        budget()
        model.train()
        losses = []
        for x, y in batches:
            opt.zero_grad(set_to_none=True)
            z = model(*inputs(x))
            loss = nn.functional.cross_entropy(z, y.cuda())
            if not torch.isfinite(loss):
                raise ValueError('nonfinite source loss')
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses))})
        save(f'{name}_{seed}_source_history.json', history)
        print(json.dumps({'dataset': name, 'seed': seed, **history[-1]}), flush=True)
    model.eval()
    save_checkpoint({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'config_sha256': sha(RUN / 'config.json'), 'seed': seed, 'classes': classes,
                'source_only': True, 'terminal_epoch': C['model']['source_epochs']}, path)
    return model


def clone_source(name, classes, seed):
    model = MultiViewModel('wide_bn', classes).cuda()
    ck = torch.load(RUN / 'checkpoints' / f'{name}_{seed}_source.pt', map_location='cpu', weights_only=True)
    assert ck['config_sha256'] == sha(RUN / 'config.json')
    model.load_state_dict(ck['state_dict'], strict=True)
    return model.eval()


def moments(model, split):
    summary = Moments()
    model.eval()
    with torch.inference_mode():
        for (x,) in loader(split):
            summary.add(model.generator(*inputs(x))['time'])
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in summary.finish().items()}


def adapt(model, reference, condition, source_moments, seed):
    seed_all(seed)
    model.eval().requires_grad_(False)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if condition == 'token_adain':
        return AdaINToken(source_moments, moments(model, reference)).cuda(), {'executed_updates': 0}, []
    if condition == 'token_projector':
        aligner = ResidualTokenProjector(**C['token_alignment']['projector']).cuda()
        params = list(aligner.parameters())
    elif condition == 'tent_entropy_gate':
        aligner = nn.Identity()
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or (
                    isinstance(module, nn.LayerNorm) and name.startswith('head.') and name.endswith(('norm1', 'norm2'))):
                module.requires_grad_(True)
        # final_norm and classifier are locked; BN running buffers are fixed.
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        raise ValueError(condition)
    opt = torch.optim.Adam(params, lr=C['adaptation']['learning_rate'], betas=(.9, .999))
    batches = loader(reference, shuffle=True, seed=2026)
    iterator = iter(batches)
    trajectory = []
    begin = time.monotonic()
    for step in range(C['adaptation']['candidate_updates']):
        if step % 100 == 0:
            budget()
            if time.monotonic() - begin > C['budget']['max_task_seconds']:
                raise TimeoutError('adaptation task budget exhausted')
        try:
            (x,) = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            (x,) = next(iterator)
        opt.zero_grad(set_to_none=True)
        if condition == 'token_projector':
            with torch.no_grad():
                tokens = model.generator(*inputs(x))['time']
            aligned = aligner(tokens)
            loss = moment_alignment_loss(tokens, aligned, source_moments)
            retained = len(x)
        else:
            # Deterministic dropout, with all BN statistics held at source.
            model.train()
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            torch.manual_seed(seed + 100000 + step)
            z = model(*inputs(x))
            entropy = -(z.softmax(1) * z.log_softmax(1)).sum(1)
            keep = (entropy >= C['adaptation']['gate_low_nats']) & (entropy <= C['adaptation']['gate_high_nats'])
            retained = int(keep.sum())
            loss = entropy[keep].mean() if retained else entropy.sum() * 0
        if not torch.isfinite(loss):
            raise ValueError('nonfinite adaptation loss')
        if retained:
            loss.backward()
            opt.step()
        trajectory.append({'step': step + 1, 'retained': retained, 'loss': float(loss.detach()), 'executed': retained > 0})
    model.eval()
    unchanged_buffers = all(torch.equal(value.detach().cpu(), initial[key]) for key, value in model.named_buffers())
    assert unchanged_buffers
    if condition == 'token_projector':
        assert all(torch.equal(value.detach().cpu(), initial[key]) for key, value in model.state_dict().items())
    return aligner.eval(), {'executed_updates': sum(r['executed'] for r in trajectory),
             'retained_exposures': sum(r['retained'] for r in trajectory),
             'trainable_parameters': sum(p.numel() for p in params), 'source_bn_buffers_unchanged': True,
             'adaptation_seconds': time.monotonic() - begin}, trajectory


def predict(model, query, aligner=None):
    model.eval()
    outputs = []
    with aligned_generator(model, aligner or nn.Identity()), torch.inference_mode():
        for (x,) in loader(query):
            outputs.append(model(*inputs(x)).cpu().numpy())
    output = np.concatenate(outputs)
    assert np.isfinite(output).all()
    return output


def metrics(y, p, base, classes):
    assert y.shape == p.shape == base.shape and len(y)
    assert (y >= 0).all() and (y < classes).all() and (p >= 0).all() and (p < classes).all()
    cm = np.bincount(y * classes + p, minlength=classes ** 2).reshape(classes, classes)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(classes), where=denominator > 0)
    correct = base == y
    nar = float((correct & (p != y)).sum() / correct.sum()) if correct.any() else None
    recovery = float((~correct & (p == y)).sum() / (~correct).sum()) if (~correct).any() else None
    return {'accuracy': float((p == y).mean()), 'macro_f1': float(f1.mean()),
            'negative_adaptation_rate': nar, 'recovery_rate': recovery,
            'base_correct_count': int(correct.sum()), 'base_wrong_count': int((~correct).sum()),
            'query_class_count': int(len(set(y))), 'rows': len(y)}


def main():
    assert C['status'] == 'frozen' and C['id'] == RUN.name
    worker_seed = int(os.environ['P2_SEED'])
    assert worker_seed in C['seeds']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(C['budget']['seed_gpu_map'][str(worker_seed)])
    for relative, expected in json.loads((RUN / 'artifacts/freeze.json').read_text()).items():
        assert sha(ROOT / relative) == expected, relative
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    preflight = json.loads((RUN / 'artifacts/data_preflight.json').read_text())
    assert preflight['passed'] and preflight['config_sha256'] == sha(RUN / 'config.json')
    for path, expected in preflight['data_sha256'].items():
        assert sha(path) == expected, path
    registry('running', 'GPU0/1/2按seed并行的独立方向模型矩阵启动；四条件、有效输入隔离。')
    score_name = f'scores_seed_{worker_seed}.json'
    scores_path = RUN / 'artifacts' / score_name
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else []
    for name in ('wtt_time', 'awf'):
        budget()
        print(f'data_preflight_start {name}', flush=True)
        bundle = load_cached(name)
        manifest = {'mapping': bundle['mapping'], 'audit': bundle['audit'],
                    'source': {k: v for k, v in bundle['source'].items() if k != 'x'},
                    'targets': {date: {role: {k: v for k, v in value[role].items() if k != 'x'}
                                      for role in ('reference', 'query')} for date, value in bundle['targets'].items()}}
        # Arrays become lists only in the metadata; raw features stay in memory.
        manifest = json.loads(json.dumps(manifest, default=lambda x: x.tolist()))
        manifest_file = RUN / 'artifacts' / f'{name}_row_manifest.json'
        if manifest_file.exists():
            assert json.loads(manifest_file.read_text()) == manifest
        else:
            save(f'{name}_row_manifest.json', manifest)
        print(f'data_preflight_passed {name} classes={bundle["classes"]}', flush=True)
        for seed in [worker_seed]:
            model = train_source(name, bundle['source'], bundle['classes'], seed)
            source_stats = moments(model, bundle['source'])
            torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in source_stats.items()},
                       RUN / 'artifacts' / f'{name}_{seed}_source_moments.pt')
            del model
            for date, target in bundle['targets'].items():
                baseline_model = clone_source(name, bundle['classes'], seed)
                baseline = predict(baseline_model, target['query']).argmax(1)
                del baseline_model
                for condition in C['conditions']:
                    budget()
                    key = (name, seed, date, condition)
                    stem = f'{name}_{seed}_{date}_{condition}'
                    old = [s for s in scores if (s['dataset'], s['seed'], s['slice'], s['condition']) == key]
                    if old:
                        assert len(old) == 1 and sha(RUN / 'artifacts' / f'pred_{stem}.npz') == old[0]['prediction_sha256']
                        continue
                    model = clone_source(name, bundle['classes'], seed)
                    cost, trajectory, aligner = {'executed_updates': 0}, [], nn.Identity()
                    begin = time.monotonic()
                    if condition != 'frozen':
                        aligner, cost, trajectory = adapt(model, target['reference'], condition, source_stats, seed)
                    logits = predict(model, target['query'], aligner)
                    pred = logits.argmax(1)
                    prediction_file = RUN / 'artifacts' / f'pred_{stem}.npz'
                    np.savez_compressed(prediction_file, labels=target['query']['y'], predictions=pred,
                                        logits=logits, uids=target['query']['uids'])
                    if condition != 'frozen':
                        save_checkpoint({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                    'aligner': {k: v.detach().cpu() for k, v in aligner.state_dict().items()},
                                    'condition': condition, 'config_sha256': sha(RUN / 'config.json')},
                                   RUN / 'checkpoints' / f'{stem}.pt')
                    base_metrics = metrics(target['query']['y'], baseline, baseline, bundle['classes'])
                    adapted_metrics = metrics(target['query']['y'], pred, baseline, bundle['classes'])
                    if condition == 'frozen':
                        assert np.array_equal(pred, baseline)
                    row = {'dataset': name, 'seed': seed, 'slice': date, 'condition': condition,
                           'phase': target['phase'], **metrics(target['query']['y'], pred, baseline, bundle['classes']),
                           'base_macro_f1': base_metrics['macro_f1'],
                           'delta_f1_pp': 100 * (adapted_metrics['macro_f1'] - base_metrics['macro_f1']),
                           **cost, 'cell_seconds': time.monotonic() - begin, 'prediction_sha256': sha(prediction_file),
                           'peak_cuda_bytes': int(torch.cuda.max_memory_allocated())}
                    save(f'trajectory_{stem}.json', trajectory)
                    scores.append(row)
                    save(score_name, scores)
                    save(f'progress_seed_{worker_seed}.json', {'completed': len(scores), 'total': C['budget']['total_cells'] // len(C['seeds']),
                                          'last_cell': list(key), 'elapsed_seconds': time.monotonic() - START})
                    print(json.dumps({'cell_completed': row}), flush=True)
                    del model, aligner
                    torch.cuda.empty_cache()
        del bundle
    assert len(scores) == C['budget']['total_cells'] // len(C['seeds'])
    save(f'execution_complete_seed_{worker_seed}.json', {'cells': len(scores), 'requires_independent_postrun_audit': True})


if __name__ == '__main__':
    try:
        signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(InterruptedError('SIGTERM')))
        main()
    except BaseException as exc:
        save(f'failure_seed_{os.environ.get("P2_SEED", "unknown")}.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        registry('failed', f'外部迁移矩阵失败：{type(exc).__name__}: {exc}；产物保留。')
        raise
