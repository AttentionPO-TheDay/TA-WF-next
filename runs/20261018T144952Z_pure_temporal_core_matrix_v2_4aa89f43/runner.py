"""One seed per GPU, terminal 800 updates, support-only dynamics logging."""
import hashlib
import json
import os
import random
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from methods import entropy, forward_features, gmm_pseudo, mmd
from runtime import (RUN, ROOT, C, save, sha, setup, verify_freeze, load_model,
                     load_source, load_target, configure, adaptation_mode, predict, metrics)


def indices(n, seed):
    return DataLoader(torch.arange(n), batch_size=C['adaptation']['batch_size'], shuffle=True,
                      drop_last=True, generator=torch.Generator().manual_seed(seed), num_workers=0)


def take(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def probe(model, support, step):
    # Eval-mode support probe changes neither RNG nor BN buffers. It never
    # reads query input/labels and never controls selection or early stopping.
    before = {k: v.detach().clone() for k, v in model.named_buffers()}
    logits = predict(model, support)
    assert all(torch.equal(value, before[key]) for key, value in model.named_buffers())
    z = torch.from_numpy(logits)
    h = entropy(z)
    norms = z.norm(dim=1)
    return {'step': step, 'logit_l2_mean': float(norms.mean()), 'logit_l2_p95': float(norms.quantile(.95)),
            'entropy_mean': float(h.mean()), 'fraction_entropy_lt_0_2': float((h < .2).float().mean()),
            'classifier_weight_l2': float(model.classifier.weight.detach().norm()),
            'final_norm_weight_l2': float(model.final_norm.weight.detach().norm()),
            'scope': 'fixed_1024_unlabeled_reference_eval_mode'}


def adapt(model, source, support, condition, seed, role, steps=None):
    assert set(support) == {'timestamps', 'tam', 'rows'}
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    trainable = configure(model, condition)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    source = {k: v.cuda() if k in ('timestamps', 'tam', 'labels') else v for k, v in source.items()}
    support = {k: v.cuda() if k in ('timestamps', 'tam') else v for k, v in support.items()}
    target_loader = indices(len(support['rows']), 2026)
    source_loader = indices(len(source['rows']), 2025)
    ti, si = iter(target_loader), iter(source_loader)
    lr = C['adaptation']['proteus_lr'] if condition == 'proteus_default' else C['adaptation']['tent_lr']
    if condition == 'no_pseudo_fixed':
        lr = C['adaptation']['no_pseudo_lr']
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(.9, .999), weight_decay=0)
    history, dynamics, gmm_history = [], [probe(model, support, 0)], []
    stream = {name: hashlib.sha256() for name in ('source', 'target', 'pseudo')}
    p_loader = p_iterator = None
    started = time.monotonic()
    for step in range(C['candidate_updates'] if steps is None else steps):
        if time.monotonic() - started > C['budget']['max_seconds_per_cell']:
            raise TimeoutError('adaptation cell budget exceeded')
        if condition == 'proteus_default' and step % 40 == 0:
            report = gmm_pseudo(model, support, 'cuda:0', 128, .6)
            selected = np.flatnonzero(report['selected'])
            p_loader = DataLoader(TensorDataset(torch.from_numpy(selected), torch.from_numpy(report['labels'][selected])),
                                  batch_size=128, shuffle=True, drop_last=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(2030 + step // 40))
            p_iterator = iter(p_loader)
            gmm_history.append({'step': step, 'coverage': report['coverage'], 'means': report['means']})
        target_idx, ti = take(ti, target_loader)
        stream['target'].update(support['rows'][target_idx].numpy().astype('<i8').tobytes())
        tx, tt = support['timestamps'][target_idx.cuda()], support['tam'][target_idx.cuda()]
        torch.manual_seed(C['adaptation']['dropout_seed_base'] + step)
        adaptation_mode(model, condition)
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        if condition in ('proteus_default', 'no_pseudo_fixed'):
            source_idx, si = take(si, source_loader)
            stream['source'].update(source['rows'][source_idx].numpy().astype('<i8').tobytes())
            sx, st, sy = [source[k][source_idx.cuda()] for k in ('timestamps', 'tam', 'labels')]
            so, sf = forward_features(model, sx, st)
            to, tf = forward_features(model, tx, tt)
            source_ce = nn.functional.cross_entropy(so, sy)
            marginal = to.softmax(1).mean(0)
            ent_div = entropy(to).mean() + (marginal * (marginal + 1e-5).log()).sum()
            discrepancy = mmd(sf, tf)
            pseudo_ce = to.new_zeros(())
            if condition == 'proteus_default':
                batch, p_iterator = take(p_iterator, p_loader)
                pi, py = batch
                stream['pseudo'].update(support['rows'][pi].numpy().astype('<i8').tobytes())
                pz, _ = forward_features(model, support['timestamps'][pi.cuda()], support['tam'][pi.cuda()])
                pseudo_ce = nn.functional.cross_entropy(pz, py.cuda())
            loss = source_ce + ent_div + discrepancy + pseudo_ce
            retained = len(target_idx)
            losses = {'source_ce': float(source_ce.detach()), 'entropy_diversity': float(ent_div.detach()),
                      'mmd': float(discrepancy.detach()), 'pseudo_ce': float(pseudo_ce.detach())}
            z = to
        else:
            z = model(tx, tt)
            h = entropy(z)
            keep = (h >= .2) & (h <= .4) if condition == 'tent_entropy_gate' else torch.ones_like(h, dtype=torch.bool)
            retained = int(keep.sum())
            loss = h[keep].mean() if retained else h.sum() * 0
        if not torch.isfinite(z).all() or not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss/logits: {seed}/{role}/{condition}/step{step}')
        if retained:
            loss.backward()
            optimizer.step()
        history.append({'step': step + 1, 'loss': float(loss.detach()), 'retained': retained,
                        'executed': retained > 0, 'batch_entropy_mean': float(entropy(z).detach().mean()),
                        'batch_logit_l2_mean': float(z.detach().norm(dim=1).mean()), **losses})
        if (step + 1) % C['diagnostics']['probe_interval'] == 0:
            dynamics.append(probe(model, support, step + 1))
            save(f'heartbeat_{seed}.json', {'seed': seed, 'role': role, 'condition': condition,
                                          'step': step + 1, 'cell_seconds': time.monotonic() - started})
    state = model.state_dict()
    for key, value in state.items():
        assert torch.isfinite(value).all(), key
        if condition == 'tent_entropy_gate' and key not in trainable:
            assert torch.equal(value.cpu(), initial[key]), key
        if condition == 'no_pseudo_fixed' and key.startswith('classifier.'):
            assert torch.equal(value.cpu(), initial[key]), key
        if condition == 'tent_bn' and key in dict(model.named_parameters()) and key not in trainable:
            assert torch.equal(value.cpu(), initial[key]), key
    return {'history': history, 'dynamics': dynamics, 'gmm_history': gmm_history,
            'cost': {'candidate_updates': len(history), 'executed_updates': sum(r['executed'] for r in history),
                     'retained_exposures': sum(r['retained'] for r in history),
                     'adaptation_seconds': time.monotonic() - started, 'learning_rate': lr,
                     'trainable_names': trainable, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'streams': {k: v.hexdigest() for k, v in stream.items()}}}


def main():
    setup()
    verify_freeze()
    seed = int(os.environ['TEMPORAL_SEED'])
    assert seed in C['seeds'] and os.environ['CUDA_VISIBLE_DEVICES'] == str(C['budget']['physical_gpu'])
    preflight = json.loads((RUN / 'artifacts/preflight.json').read_text())
    assert preflight['passed'] and preflight['config_sha256'] == sha(RUN / 'config.json')
    source = load_source()
    score_name = f'scores_seed_{seed}.json'
    path = RUN / 'artifacts' / score_name
    scores = json.loads(path.read_text()) if path.exists() else []
    start = time.monotonic()
    for role in C['roles']:
        support, query = load_target(role)
        baseline_path = ROOT / 'runs' / C['frozen_baseline_run'] / 'artifacts' / f'pred_{seed}_{role}_frozen.npz'
        with np.load(baseline_path) as baseline:
            assert np.array_equal(baseline['rows'], query['rows'].numpy())
            assert np.array_equal(baseline['labels'], query['labels'].numpy())
            frozen = baseline['predictions'].copy()
        for condition in C['adaptation_conditions']:
            if time.monotonic() - start > C['budget']['max_wall_seconds']:
                raise TimeoutError('worker wall budget exceeded')
            key = f'{seed}_{role}_{condition}'
            matched = [s for s in scores if s['role'] == role and s['condition'] == condition]
            if matched:
                assert len(matched) == 1
                assert sha(RUN / 'artifacts' / f'pred_{key}.npz') == matched[0]['prediction_sha256']
                assert sha(RUN / 'checkpoints' / f'{key}.pt') == matched[0]['checkpoint_sha256']
                continue
            torch.cuda.reset_peak_memory_stats()
            model = load_model(seed)
            report = adapt(model, source, support, condition, seed, role)
            logits = predict(model, query)
            pred = logits.argmax(1)
            base_scores = metrics(query['labels'].numpy(), frozen, frozen)
            row = {'seed': seed, 'role': role, 'condition': condition,
                   **metrics(query['labels'].numpy(), pred, frozen), **report['cost'],
                   'rows': len(pred), 'delta_f1_pp': 0.0, 'origin': 'new_pure_source_adaptation',
                   'query_batch_statistics': condition == 'tent_bn',
                   'query_entropy_mean': float(entropy(torch.from_numpy(logits)).mean()),
                   'query_logit_l2_mean': float(np.linalg.norm(logits, axis=1).mean()),
                   'peak_cuda_bytes': int(torch.cuda.max_memory_allocated())}
            row['delta_f1_pp'] = 100 * (row['macro_f1'] - base_scores['macro_f1'])
            prediction_path = RUN / 'artifacts' / f'pred_{key}.npz'
            with prediction_path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream, logits=logits, predictions=pred, labels=query['labels'].numpy(), rows=query['rows'].numpy())
            prediction_path.with_suffix('.npz.tmp').replace(prediction_path)
            checkpoint_path = RUN / 'checkpoints' / f'{key}.pt'
            temp = checkpoint_path.with_suffix('.partial')
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'seed': seed, 'role': role, 'condition': condition, 'config_sha256': sha(RUN / 'config.json')}, temp)
            temp.replace(checkpoint_path)
            row['prediction_sha256'] = sha(prediction_path)
            row['checkpoint_sha256'] = sha(checkpoint_path)
            save(f'updates_{key}.json', report['history'])
            save(f'dynamics_{key}.json', report['dynamics'])
            save(f'gmm_{key}.json', report['gmm_history'])
            scores.append(row)
            save(score_name, scores)
            save(f'progress_seed_{seed}.json', {'completed_new_cells': len(scores), 'total_new_cells': 12,
                                              'last_cell': key, 'elapsed_seconds': time.monotonic() - start})
            print(json.dumps({'completed': row}), flush=True)
            del model
            torch.cuda.empty_cache()
    assert len(scores) == 12


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(f'failure_seed_{os.environ.get("TEMPORAL_SEED", "unknown")}.json',
             {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
