"""Explicit copy of core-matrix adaptation, with only LR/step overrides."""
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


def adapt(model, source, support, condition, seed, role, steps=None, lr_override=None, gate_override=None):
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
    if lr_override is not None:
        assert condition in ('no_pseudo_fixed', 'tent_entropy_gate') and lr_override in (0.000125, 0.00025, 0.0005)
        lr = lr_override
    if gate_override is not None:
        assert condition == 'tent_entropy_gate' and tuple(gate_override) in ((.2,.4),(.25,.35),(.1,.5))
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
            gate = gate_override if gate_override is not None else (.2, .4)
            keep = (h >= gate[0]) & (h <= gate[1]) if condition == 'tent_entropy_gate' else torch.ones_like(h, dtype=torch.bool)
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


