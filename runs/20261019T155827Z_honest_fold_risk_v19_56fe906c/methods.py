"""Support-only supervised protection and finite cross-validation routes."""
import numpy as np
import torch
from maintenance_base import (unit, predict, metrics, margin_gain, frontier,
                              best_under_damage, least_damage_for_recovery, fit_head)


def proto(x, y, source, pseudocount=3.):
    count = np.bincount(y, minlength=102)
    sums = np.zeros_like(source, dtype=np.float64)
    np.add.at(sums, y, unit(x))
    # Empty target classes retain their source center, including OOF 1-shot.
    centers = (sums + pseudocount * source) / (count[:, None] + pseudocount)
    return {'kind': np.array('proto'), 'centers': unit(centers).astype(np.float32)}


def fit_linear(x, y, w0, b0, name, c):
    cfg = {'steps': c['steps'], 'lr': c['lr'], 'lambda': c['lambda_head8'] if name == 'head8' else c['lambda_head1']}
    if name != 'protect':
        return fit_head(x, y, w0, b0, cfg)
    tx = torch.from_numpy(x.astype(np.float32)); ty = torch.from_numpy(y.astype(np.int64))
    iw, ib = torch.from_numpy(w0.copy()), torch.from_numpy(b0.copy())
    w, b = torch.nn.Parameter(iw.clone()), torch.nn.Parameter(ib.clone())
    original = tx @ iw.T + ib
    correct = original.argmax(1) == ty
    mask = torch.nn.functional.one_hot(ty, 102).bool()
    oldmargin = original.gather(1, ty[:, None]).squeeze(1) - original.masked_fill(mask, -torch.inf).max(1).values
    scale = original.norm(dim=1).clamp_min(1e-6)
    optimizer = torch.optim.Adam([w, b], lr=c['lr'])
    history = []
    for step in range(c['steps']):
        optimizer.zero_grad(set_to_none=True)
        z = tx @ w.T + b
        ce = torch.nn.functional.cross_entropy(z, ty)
        anchor = c['protect_lambda'] / 204 * ((w-iw).square().sum() + (b-ib).square().sum())
        margin = z.gather(1, ty[:, None]).squeeze(1) - z.masked_fill(mask, -torch.inf).max(1).values
        protection = (torch.relu((c['protect_retained_margin_fraction'] * oldmargin - margin) / scale)[correct].square().mean()
                      if correct.any() else z.sum() * 0)
        loss = ce + anchor + c['protect_beta'] * protection
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
        if step in (0, 99, c['steps']-1):
            history.append({'step': step+1, 'ce': float(ce.detach()), 'anchor': float(anchor.detach()),
                            'protection': float(protection.detach()), 'protected_rows': int(correct.sum())})
    return {'kind': np.array('linear'), 'weight': w.detach().numpy(), 'bias': b.detach().numpy()}, history


def softmax(z):
    shifted = z-z.max(1, keepdims=True)
    ex = np.exp(shifted)
    return ex/ex.sum(1, keepdims=True)


def local_scores(x, train_x, train_y):
    similarity = unit(x) @ unit(train_x).T
    out = np.full((len(x), 102), -np.inf, dtype=np.float32)
    for cls in np.unique(train_y):
        out[:, cls] = similarity[:, train_y == cls].max(1)
    return out


def policies(original, logits, local, descriptions, prototype_scale=20.):
    base = original.argmax(1)
    p0 = softmax(original)
    probs = {name: softmax(z*prototype_scale if name=='shrink' else z) for name, z in logits.items()}
    top = {name: z.argmax(1) for name, z in logits.items()}
    nearest = local.argmax(1)
    for d in descriptions:
        name, kind, model = d['name'], d['kind'], d['model']
        if kind == 'prob':
            z = np.log(np.maximum((1-d['alpha'])*p0+d['alpha']*probs[model], 1e-12))
        elif kind == 'logit':
            z = (1-d['alpha'])*original+d['alpha']*logits[model]
        else:
            z = logits[model]
        p = z.argmax(1)
        if kind == 'agree':
            p = np.where(p == top[d['other']], p, base)
        elif kind == 'local':
            i = np.arange(len(base))
            keep = (p == nearest) & (local[i, p] > local[i, base])
            p = np.where(keep, p, base)
        gain = margin_gain(z, original, p, base)
        yield name, p.astype(np.int64), gain


def curve_with_thresholds(y, p, base, gain):
    idx = np.flatnonzero(p != base)
    idx = idx[np.argsort(-gain[idx], kind='stable')]
    curve = frontier(y, p, base, gain)
    if not len(idx):
        return curve, np.array([np.inf])
    end = np.r_[np.flatnonzero(np.diff(gain[idx]) != 0), len(idx)-1]
    return curve, np.r_[np.inf, gain[idx][end]]


def select_policy(y, original, outputs, config):
    base = original.argmax(1)
    correct = int((base == y).sum())
    result = {}
    for selector, rule in config.items():
        best = None
        tested_thresholds = 0
        for rank, (name, p, gain) in enumerate(outputs):
            curve, thresholds = curve_with_thresholds(y, p, base, gain)
            tested_thresholds += len(thresholds)
            eligible = np.flatnonzero((curve['D'] <= correct*rule['max_empirical_oof_NAR']) & (curve['R'] >= rule['minimum_R']))
            for j in eligible:
                r,d,w = (int(curve[k][j]) for k in ['R','D','W'])
                key = (r-4*d-.1*w, -d, -w, -rank, -(r+d+w))
                if best is None or key > best[0]:
                    best = (key, {'policy': name, 'threshold': float(thresholds[j]), 'OOF_R': r, 'OOF_D': d, 'OOF_W': w})
        chosen = best[1] if best is not None else {'policy': 'frozen', 'threshold': None, 'OOF_R': 0, 'OOF_D': 0, 'OOF_W': 0}
        chosen.update({'OOF_original_correct': correct, 'OOF_rows': len(y), 'OOF_empirical_NAR': chosen['OOF_D']/correct,
                       'tested_thresholds': tested_thresholds, 'risk_guarantee': False})
        result[selector] = chosen
    return result


def apply_choice(choice, outputs, base):
    if choice['policy'] == 'frozen':
        return base.copy(), np.zeros(len(base), dtype=np.float32)
    _, p, gain = next(o for o in outputs if o[0] == choice['policy'])
    chosen = np.where((p != base) & (gain >= choice['threshold']), p, base)
    return chosen, gain
