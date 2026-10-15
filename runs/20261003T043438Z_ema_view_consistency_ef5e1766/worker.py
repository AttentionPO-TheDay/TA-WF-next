"""GPU worker for online EMA consistency, without a fixed teacher."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, json, time
import numpy as np
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask, learning_rate, consistency_weight, make_optimizer
from common import RUN, config, verify_freeze, deterministic, save, savet, sha, metric

SEEDS = [21729, 23407, 22026]


def predict(model, e, bs):
    model.eval()
    before = {k: v.detach().clone() for k, v in model.named_buffers()}
    out = []
    with torch.inference_mode():
        for i in range(0, len(e['labels']), bs):
            out.append(model(e['timestamps'][i:i + bs], e['tam'][i:i + bs]).cpu().numpy())
    after = dict(model.named_buffers())
    assert all(torch.equal(before[k], after[k]) for k in before)
    return np.concatenate(out)


def ema_update(ema, model, decay):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.mul_(decay).add_(p, alpha=1.0 - decay)
        # Keep normalization statistics tied to the student's observed batch
        # stream; the EMA applies only to trainable parameters.
        for eb, b in zip(ema.buffers(), model.buffers()):
            eb.copy_(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True, choices=['ce_only', 'ema_same_view', 'ema_cross_view'])
    ap.add_argument('--seed', required=True, type=int, choices=SEEDS)
    ap.add_argument('--device', required=True, choices=['cuda'])
    a = ap.parse_args()
    c = config(); verify_freeze()
    assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    deterministic('cuda', 2)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cuda')
    raw = torch.load(RUN / 'artifacts/prepared.pt', weights_only=False, map_location='cpu')
    data = {role: {k: v.to(device) for k, v in e.items() if k in ('timestamps', 'tam', 'labels')}
            for role, e in raw.items()}
    labels = {r: e['labels'].cpu().numpy() for r, e in data.items()}
    out = RUN / 'artifacts' / f'{a.kind}_{a.seed}'; out.mkdir(exist_ok=False)
    model = MultiViewModel('ce_only').to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    savet(out / 'initial_state.pt', initial)
    opt = make_optimizer(model)
    ema = None
    if a.kind != 'ce_only':
        ema = MultiViewModel('ce_only').to(device)
        ema.load_state_dict(model.state_dict())
        ema.requires_grad_(False); ema.eval()
        assert all(not p.requires_grad for p in ema.parameters())
    n = len(data['source']['labels']); bs = c['batch_size']; steps = c['optimizer_steps']
    pieces = []; count = 0; cycle = 0
    while count < steps * bs:
        pieces.append(torch.randperm(n, generator=torch.Generator().manual_seed(a.seed + 9000 + cycle)))
        count += n; cycle += 1
    indices = torch.cat(pieces)[:steps * bs].reshape(steps, bs)
    savet(out / 'index_stream.pt', indices)
    aug_rng = torch.Generator().manual_seed(a.seed + 50000)
    T = c['ema_consistency']['temperature']; decay = c['ema_consistency']['decay']
    start = time.monotonic(); history = []; best = -1.; beststep = 0
    sums = {'loss': 0., 'ce': 0., 'consistency': 0., 'weight': 0.}; prev = 0
    print(json.dumps({'task': f'{a.kind}_{a.seed}', 'device': 'cuda', 'steps': steps}), flush=True)
    for step in range(1, steps + 1):
        if time.monotonic() - start > c['gpu_job_seconds']:
            raise TimeoutError(f'{a.kind}_{a.seed}')
        for g in opt.param_groups:
            g['lr'] = learning_rate(step, steps)
        model.train(); idx = indices[step - 1].cuda(); e = data['source']
        clean = e['tam'][idx]; tam = span_mask(clean, aug_rng); y = e['labels'][idx]
        opt.zero_grad(set_to_none=True)
        logits = model(e['timestamps'][idx], tam)
        ce = F.cross_entropy(logits, y); consistency = ce.new_zeros(()); weight = 0.
        if ema is not None:
            target_tam = tam if a.kind == 'ema_same_view' else clean
            with torch.no_grad():
                target = F.softmax(ema(e['timestamps'][idx], target_tam) / T, dim=1).detach()
            consistency = F.kl_div(F.log_softmax(logits / T, dim=1), target, reduction='batchmean') * (T * T)
            weight = consistency_weight(step, c['ema_consistency']['max_weight'], c['ema_consistency']['ramp_steps'])
        loss = ce + weight * consistency
        assert torch.isfinite(loss) and torch.isfinite(ce) and torch.isfinite(consistency)
        loss.backward()
        if step == 1:
            grads = {k: (float(p.grad.norm()) if p.grad is not None else None) for k, p in model.named_parameters()}
            save(out / 'first_gradients.json', grads)
            active = [v for k, v in grads.items() if (k.startswith('generator.stages') or k == 'generator.to_features.weight') and k.endswith('weight')]
            assert active and all(v is not None and np.isfinite(v) and v > 0 for v in active)
            if ema is not None:
                assert all(p.grad is None for p in ema.parameters())
        opt.step()
        if ema is not None:
            ema_update(ema, model, decay)
            assert all(p.grad is None for p in ema.parameters())
        sums['loss'] += float(loss.detach()); sums['ce'] += float(ce.detach()); sums['consistency'] += float(consistency.detach()); sums['weight'] += weight
        if step % 100 == 0:
            save(out / 'progress.json', {'task': f'{a.kind}_{a.seed}', 'step': step, 'total_steps': steps, 'best_step': beststep, 'elapsed_seconds': time.monotonic() - start})
        if step in c['eval_steps']:
            pred = {r: predict(model, e, c['eval_batch_size']) for r, e in data.items()}
            scores = {r: metric(labels[r], z.argmax(1)) for r, z in pred.items()}
            row = {'step': step, 'lr': learning_rate(step, steps), 'train_loss': sums['loss'] / (step - prev), 'train_ce': sums['ce'] / (step - prev), 'train_consistency': sums['consistency'] / (step - prev), 'mean_consistency_weight': sums['weight'] / (step - prev), 'elapsed_seconds': time.monotonic() - start, **scores}
            sums = {'loss': 0., 'ce': 0., 'consistency': 0., 'weight': 0.}; prev = step; history.append(row)
            if scores['valid']['accuracy'] > best:
                best = scores['valid']['accuracy']; beststep = step
                savet(RUN / 'checkpoints' / f'{a.kind}_{a.seed}_best.pt', {'state_dict': model.state_dict(), 'step': step, 'config_sha256': sha(RUN / 'config.json')})
            save(out / 'history.json', history); print(json.dumps({'task': f'{a.kind}_{a.seed}', **row}), flush=True)
    last = {r: predict(model, e, c['eval_batch_size']) for r, e in data.items()}
    np.savez_compressed(out / 'predictions_last.npz', **{r: z.argmax(1) for r, z in last.items()})
    ck = torch.load(RUN / 'checkpoints' / f'{a.kind}_{a.seed}_best.pt', weights_only=True, map_location=device)
    model.load_state_dict(ck['state_dict'])
    pred = {r: predict(model, e, c['eval_batch_size']) for r, e in data.items()}
    np.savez_compressed(out / 'predictions_best.npz', **{r: z.argmax(1) for r, z in pred.items()})
    np.savez_compressed(out / 'logits_best.npz', **pred)
    reload = MultiViewModel('ce_only').cuda(); reload.load_state_dict(ck['state_dict'])
    for r, e in data.items():
        assert np.array_equal(predict(reload, e, c['eval_batch_size']).argmax(1), pred[r].argmax(1))
    report = {'task': f'{a.kind}_{a.seed}', 'kind': a.kind, 'seed': a.seed, 'device': 'cuda', 'best_step': beststep, 'steps_completed': steps, 'validation_opportunities': len(history), 'parameters_total': sum(p.numel() for p in reload.parameters()), 'parameters_trainable': sum(p.numel() for p in reload.parameters() if p.requires_grad), 'ema_used': ema is not None, 'ema_decay': decay if ema is not None else None, 'temperature': T if ema is not None else None, 'max_consistency_weight': c['ema_consistency']['max_weight'] if ema is not None else 0., 'ramp_steps': c['ema_consistency']['ramp_steps'] if ema is not None else 0, 'verified_roles': ['source', 'valid'], 'future_access': False, 'best': {r: metric(labels[r], z.argmax(1)) for r, z in pred.items()}, 'last': {r: metric(labels[r], z.argmax(1)) for r, z in last.items()}, 'elapsed_seconds': time.monotonic() - start}
    save(out / 'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
