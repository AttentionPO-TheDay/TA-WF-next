"""One bounded wide+BN student training task."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, json, time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from model import MultiViewModel, span_mask, learning_rate, make_optimizer
from common import RUN, ROOT, config, verify_freeze, deterministic, save, savet, sha, metric

SEEDS = [21729, 23407, 22026]


def predict(model, e, batch_size):
    model.eval()
    before = {k: v.detach().clone() for k, v in model.named_buffers()}
    out = []
    with torch.inference_mode():
        for i in range(0, len(e['labels']), batch_size):
            out.append(model(e['timestamps'][i:i + batch_size], e['tam'][i:i + batch_size]).cpu().numpy())
    after = dict(model.named_buffers())
    assert all(torch.equal(before[k], after[k]) for k in before), 'evaluation updated buffers'
    return np.concatenate(out)


def teacher_source(seed, kind):
    paths = [RUN / 'artifacts' / f'teacher_logits_{s}.npz' for s in SEEDS]
    arrays = []
    for path in paths:
        with np.load(path) as z:
            arrays.append(z['source'].astype(np.float32, copy=False))
    target = torch.from_numpy(np.mean(np.stack(arrays, axis=0), axis=0))
    assert target.shape == (15300, 102) and torch.isfinite(target).all()
    if kind == 'shuffled_kd':
        perm = torch.randperm(len(target), generator=torch.Generator().manual_seed(seed + 70000))
        target = target[perm]
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True, choices=['ce_only', 'ensemble_kd', 'shuffled_kd'])
    ap.add_argument('--seed', required=True, type=int, choices=SEEDS)
    ap.add_argument('--device', required=True, choices=['cpu', 'cuda'])
    a = ap.parse_args()
    c = config(); verify_freeze()
    if a.device == 'cuda':
        assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        torch.cuda.manual_seed_all(a.seed)
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    deterministic(a.device, 2 if a.device == 'cuda' else 4)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    raw = torch.load(RUN / 'artifacts/prepared.pt', weights_only=False, map_location='cpu')
    data = {role: {k: v.to(device) for k, v in entry.items() if k in ('timestamps', 'tam', 'labels')}
            for role, entry in raw.items()}
    labels = {role: e['labels'].cpu().numpy() for role, e in data.items()}
    del raw
    target = teacher_source(a.seed, a.kind).to(device)
    if a.kind == 'ce_only':
        target = None
    out = RUN / 'artifacts' / f'{a.kind}_{a.seed}'
    out.mkdir(exist_ok=False)
    model = MultiViewModel(a.kind).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    savet(out / 'initial_state.pt', initial)
    reference = torch.load(RUN / 'artifacts' / f'teacher_initial_{a.seed}.pt', weights_only=True, map_location='cpu')
    assert set(reference) == set(initial) and all(torch.equal(reference[k], initial[k]) for k in initial)
    opt = make_optimizer(model)
    n, bs, steps = len(data['source']['labels']), c['batch_size'], c['optimizer_steps']
    pieces, count, cycle = [], 0, 0
    while count < steps * bs:
        pieces.append(torch.randperm(n, generator=torch.Generator().manual_seed(a.seed + 9000 + cycle)))
        count += n; cycle += 1
    indices = torch.cat(pieces)[:steps * bs].reshape(steps, bs)
    savet(out / 'index_stream.pt', indices)
    aug_rng = torch.Generator().manual_seed(a.seed + 50000)
    T, W = c['distillation']['temperature'], c['distillation']['weight']
    budget = c['gpu_job_seconds'] if a.device == 'cuda' else c['cpu_job_seconds']
    start = time.monotonic(); history = []; best = -1.; best_step = 0
    sums = {'loss': 0., 'ce': 0., 'kd': 0.}; prev = 0
    print(json.dumps({'task': f'{a.kind}_{a.seed}', 'device': a.device, 'steps': steps, 'budget_seconds': budget}), flush=True)
    for step in range(1, steps + 1):
        if time.monotonic() - start > budget:
            raise TimeoutError(f'{a.kind}_{a.seed} budget exceeded')
        for group in opt.param_groups:
            group['lr'] = learning_rate(step, steps)
        model.train(); idx = indices[step - 1].to(device); e = data['source']
        tam = span_mask(e['tam'][idx], aug_rng); y = e['labels'][idx]
        opt.zero_grad(set_to_none=True)
        logits = model(e['timestamps'][idx], tam)
        ce = F.cross_entropy(logits, y)
        kd = ce.new_zeros(())
        if target is not None:
            kd = F.kl_div(F.log_softmax(logits / T, dim=1), F.softmax(target[idx] / T, dim=1), reduction='batchmean') * (T * T)
        loss = (1.0 - W) * ce + W * kd if target is not None else ce
        assert torch.isfinite(loss) and torch.isfinite(ce) and torch.isfinite(kd)
        loss.backward()
        if step == 1:
            grads = {k: (float(p.grad.norm()) if p.grad is not None else None) for k, p in model.named_parameters()}
            save(out / 'first_gradients.json', grads)
            active = [v for k, v in grads.items() if (k.startswith('generator.stages') or k == 'generator.to_features.weight') and k.endswith('weight')]
            assert active and all(v is not None and np.isfinite(v) and v > 0 for v in active)
        opt.step()
        sums['loss'] += float(loss.detach()); sums['ce'] += float(ce.detach()); sums['kd'] += float(kd.detach())
        if step % 100 == 0:
            save(out / 'progress.json', {'task': f'{a.kind}_{a.seed}', 'device': a.device, 'step': step, 'total_steps': steps,
                                         'best_step': best_step, 'elapsed_seconds': time.monotonic() - start})
        if step in c['eval_steps']:
            pred = {role: predict(model, e, c['eval_batch_size']) for role, e in data.items()}
            scores = {role: metric(labels[role], z.argmax(1)) for role, z in pred.items()}
            row = {'step': step, 'lr': learning_rate(step, steps), 'train_loss': sums['loss'] / (step - prev),
                   'train_ce': sums['ce'] / (step - prev), 'train_kd': sums['kd'] / (step - prev),
                   'elapsed_seconds': time.monotonic() - start, **scores}
            sums = {'loss': 0., 'ce': 0., 'kd': 0.}; prev = step; history.append(row)
            if scores['valid']['accuracy'] > best:
                best = scores['valid']['accuracy']; best_step = step
                savet(RUN / 'checkpoints' / f'{a.kind}_{a.seed}_best.pt', {'state_dict': model.state_dict(), 'step': step,
                        'config_sha256': sha(RUN / 'config.json')})
            save(out / 'history.json', history)
            print(json.dumps({'task': f'{a.kind}_{a.seed}', **row}), flush=True)
    last = {role: predict(model, e, c['eval_batch_size']) for role, e in data.items()}
    np.savez_compressed(out / 'predictions_last.npz', **{role: z.argmax(1) for role, z in last.items()})
    ck = torch.load(RUN / 'checkpoints' / f'{a.kind}_{a.seed}_best.pt', weights_only=True, map_location=device)
    model.load_state_dict(ck['state_dict'])
    pred = {role: predict(model, e, c['eval_batch_size']) for role, e in data.items()}
    np.savez_compressed(out / 'predictions_best.npz', **{role: z.argmax(1) for role, z in pred.items()})
    np.savez_compressed(out / 'logits_best.npz', **pred)
    reload = MultiViewModel(a.kind).to(device); reload.load_state_dict(ck['state_dict'])
    for role, e in data.items():
        fresh = predict(reload, e, c['eval_batch_size'])
        assert np.array_equal(fresh.argmax(1), pred[role].argmax(1))
        assert abs(metric(labels[role], fresh.argmax(1))['accuracy'] - accuracy_score(labels[role], fresh.argmax(1))) < 1e-12
        assert abs(metric(labels[role], fresh.argmax(1))['macro_f1'] - f1_score(labels[role], fresh.argmax(1), labels=np.arange(102), average='macro', zero_division=0)) < 1e-12
    report = {'task': f'{a.kind}_{a.seed}', 'kind': a.kind, 'seed': a.seed, 'device': a.device,
              'best_step': best_step, 'steps_completed': steps, 'validation_opportunities': len(history),
              'parameters_total': sum(p.numel() for p in reload.parameters()),
              'parameters_trainable': sum(p.numel() for p in reload.parameters() if p.requires_grad),
              'teacher_target': 'none' if target is None else ('shuffled_ensemble' if a.kind == 'shuffled_kd' else 'ensemble_mean'),
              'temperature': T, 'distillation_weight': 0.0 if target is None else W,
              'verified_roles': ['source', 'valid'], 'future_access': False,
              'bn_eval_buffers_unchanged': True,
              'best': {role: metric(labels[role], z.argmax(1)) for role, z in pred.items()},
              'last': {role: metric(labels[role], z.argmax(1)) for role, z in last.items()},
              'elapsed_seconds': time.monotonic() - start}
    save(out / 'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
