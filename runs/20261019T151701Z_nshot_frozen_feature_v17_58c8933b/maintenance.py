"""Supervised support-only maintenance; no query arguments in fitting APIs."""
import numpy as np
import torch


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def class_means(x, y, classes=102):
    counts = np.bincount(y, minlength=classes)
    assert counts.min() > 0
    sums = np.zeros((classes, x.shape[1]), dtype=np.float64)
    np.add.at(sums, y, x)
    return (sums / counts[:, None]).astype(np.float32), counts


def fit_proto(support_x, support_y, source_means, pseudocount):
    means, counts = class_means(unit(support_x), support_y)
    centers = (counts[:, None] * means + pseudocount * source_means) / (counts[:, None] + pseudocount)
    return {'kind': np.array('proto'), 'centers': unit(centers).astype(np.float32)}


def fit_head(support_x, support_y, w0, b0, config):
    x = torch.from_numpy(support_x.astype(np.float32))
    y = torch.from_numpy(support_y.astype(np.int64))
    initial_w, initial_b = torch.from_numpy(w0.copy()), torch.from_numpy(b0.copy())
    w = torch.nn.Parameter(initial_w.clone())
    b = torch.nn.Parameter(initial_b.clone())
    optimizer = torch.optim.Adam([w, b], lr=config['lr'])
    history = []
    for step in range(config['steps']):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ w.T + b
        ce = torch.nn.functional.cross_entropy(logits, y)
        penalty = config['lambda'] / (2 * len(w)) * ((w - initial_w).square().sum() + (b - initial_b).square().sum())
        loss = ce + penalty
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if step in (0, 49, 99, 149, config['steps'] - 1):
            history.append({'step': step + 1, 'ce_before_step': float(ce.detach()), 'penalty_before_step': float(penalty.detach())})
    assert all(torch.isfinite(v).all() for v in (w, b))
    model = {'kind': np.array('linear'), 'weight': w.detach().numpy(), 'bias': b.detach().numpy()}
    return model, history


def predict(model, x):
    if str(model['kind']) == 'proto':
        return unit(x) @ model['centers'].T
    assert str(model['kind']) == 'linear'
    return x @ model['weight'].T + model['bias']


def metrics(y, p, base):
    cm = np.bincount(y * 102 + p, minlength=102 ** 2).reshape(102, 102)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(102), where=den > 0).mean()
    correct = base == y
    r = int((~correct & (p == y)).sum())
    d = int((correct & (p != y)).sum())
    w = int((~correct & (p != y) & (p != base)).sum())
    return {'accuracy': float((p == y).mean()), 'macro_f1': float(f1), 'R': r, 'D': d, 'W': w,
            'NAR': d / int(correct.sum()), 'Recovery': r / int((~correct).sum())}


def margin_gain(logits, original, prediction, base):
    i = np.arange(len(base))
    current_margin = (logits[i, prediction] - logits[i, base]) / np.maximum(np.linalg.norm(logits, axis=1), 1e-12)
    old_margin = (original[i, prediction] - original[i, base]) / np.maximum(np.linalg.norm(original, axis=1), 1e-12)
    return (current_margin - old_margin).astype(np.float32)


def frontier(y, prediction, base, score):
    # Label-chosen operating points are diagnostic, never used by predict().
    idx = np.flatnonzero(prediction != base)
    idx = idx[np.argsort(-score[idx], kind='stable')]
    if not len(idx):
        return {k: np.array([0], dtype=np.int64) for k in ('R', 'D', 'W')}
    ends = np.r_[np.flatnonzero(np.diff(score[idx]) != 0), len(idx) - 1]
    correct = base[idx] == y[idx]
    recovered = ~correct & (prediction[idx] == y[idx])
    wasted = ~correct & (prediction[idx] != y[idx])
    return {k: np.r_[0, np.cumsum(v, dtype=np.int64)[ends]] for k, v in [('R', recovered), ('D', correct), ('W', wasted)]}


def best_under_damage(curve, budget):
    feasible = np.flatnonzero(curve['D'] <= budget)
    j = max(feasible, key=lambda i: (int(curve['R'][i]), -int(curve['D'][i]), -int(curve['W'][i])))
    return {k: int(curve[k][j]) for k in ('R', 'D', 'W')}


def least_damage_for_recovery(curve, corrections):
    feasible = np.flatnonzero(curve['R'] >= corrections)
    if not len(feasible):
        return None
    j = min(feasible, key=lambda i: (int(curve['D'][i]), int(curve['W'][i]), -int(curve['R'][i])))
    return {k: int(curve[k][j]) for k in ('R', 'D', 'W')}
