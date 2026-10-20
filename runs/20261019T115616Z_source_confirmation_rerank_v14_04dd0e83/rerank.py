import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import json
import warnings
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from runtime import ROOT, RUN, C, save, sha, read, verify, csvout
from metrics import curve, audit_direct, preflight


class SourceInsufficient(Exception):
    pass


def kind(y, b, c):
    t = np.zeros(len(y), dtype=np.int8)
    t[(b != c) & (b != y) & (c == y)] = 1
    t[(b != c) & (b == y)] = 2
    t[(b != c) & (b != y) & (c != y)] = 3
    assert np.array_equal(t > 0, b != c)
    return t


def evidence(z, b, c, cal):
    temp, weight = np.array(cal['temperatures']), np.array(cal['weights'])
    q = z.astype(np.float64)/temp[:, :, None, None]
    n = len(b)
    ix = np.arange(n)
    qc, qb = q[:, :, ix, c].copy(), q[:, :, ix, b].copy()
    odds = qc-qb
    mean = np.sum(weight[:, :, None]*odds, axis=(0, 1))
    sd = np.sqrt(np.maximum(np.sum(weight[:, :, None]*(odds-mean)**2, axis=(0, 1)), 0))
    logpc = qc-logsumexp(q, axis=-1)
    logpc = np.sum(weight[:, :, None]*logpc, axis=(0, 1))
    q[:, :, ix, c] = -np.inf
    q[:, :, ix, b] = -np.inf
    third = np.sum(weight[:, :, None]*(qc-q.max(-1)), axis=(0, 1))
    return logpc, third, mean-.25*sd


def source_features():
    cal = json.loads((ROOT/C['v12']/'artifacts/source_calibration.json').read_text())
    result = []
    for role in ['train', 'valid']:
        for seed in C['seeds']:
            for sl in C['slices']:
                parts = []
                scales = cal['states'][f'{seed}_{sl}']['scales']
                for ci, case in enumerate(C['perturbations']):
                    ds = [read(RUN/'artifacts'/f'synthetic_{s}_{role}_{case["name"]}.npz') for s in C['seeds']]
                    own = ds[C['seeds'].index(seed)]
                    for dd in ds:
                        assert np.array_equal(own['rows'], dd['rows']) and np.array_equal(own['labels'], dd['labels'])
                    b, c = own['frozen_logits'].argmax(-1), own[sl+'_logits'].argmax(-1)
                    lp, third, soft = evidence(np.stack([dd['aux_logits'] for dd in ds]), b, c, cal)
                    m = own[sl+'_margin'].astype(np.float64)
                    prior = (m-scales['margin']['mean'])/scales['margin']['sd']
                    baseline = prior+.25*(soft-scales['weighted_soft']['mean'])/scales['weighted_soft']['sd']
                    t = kind(own['labels'], b, c)
                    parts.append(dict(X=np.stack([m, lp, third], -1), rows=own['rows'], labels=own['labels'], base=b, candidates=c, types=t, case_id=np.full(len(b), ci, dtype=np.int8), v9_margin=m, v12_primary=baseline))
                allrows = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
                np.savez_compressed(RUN/'artifacts'/f'source_features_{role}_{seed}_{sl}.npz', **allrows)
                result.append(dict(role=role, seed=seed, slice=sl, total_rows=len(allrows['rows']), changed=int((allrows['types'] > 0).sum()), **{k: int((allrows['types'] == v).sum()) for k, v in [('R', 1), ('D', 2), ('W', 3)]}))
    save('source_type_counts.json', result)
    csvout('source_type_counts.csv', result)
    checks = [r for r in result if r['role'] == 'train']
    passed = all(min(r[k] for k in ['R', 'D', 'W']) >= C['fit']['min_each_type'] for r in checks)
    save('source_feasibility.json', dict(passed=passed, checks=checks, query_opened=False))
    if not passed:
        raise SourceInsufficient('fixed source changed examples insufficient; no added perturbations or target scoring')


def projection(X, model):
    xx = X[:, model['columns']]
    xx = (xx-np.array(model['mean']))/np.array(model['sd'])
    g = xx@np.array(model['coef']).T+np.array(model['intercept'])
    return g[:, 0]-logsumexp(g[:, 1:], axis=1)


def scores(d, models):
    out = {k: d[k] for k in ['v9_margin', 'v12_primary']}
    for arm, model in models.items():
        u = projection(d['X'], model)
        out[arm] = d['v12_primary'] + C['fit']['blend_weight']*(u-model['u_mean'])/model['u_sd']
    assert list(out) == C['scores'] and all(np.isfinite(x).all() for x in out.values())
    return out


def prefix(t, s):
    ix = np.flatnonzero(t > 0)
    if not len(ix):
        return np.array([[np.nextafter(float(np.max(s)), np.inf), 0., 0., 0.]])
    ix = ix[np.argsort(-s[ix], kind='stable')]
    ends = np.flatnonzero(np.r_[s[ix][1:] != s[ix][:-1], True])
    thresholds = np.r_[np.nextafter(s[ix].max(), np.inf), s[ix][ends]]
    return np.stack([thresholds]+[np.r_[0, np.cumsum(t[ix] == k)[ends]] for k in [1, 2, 3]], axis=-1)


def calibrate(d, values):
    t, b, y = d['types'], d['base'], d['labels']
    cr = prefix(t, values)
    nc = int((b == y).sum())
    utility = cr[:, 1]-cr[:, 2]-C['calibration']['utility_W_cost']*cr[:, 3]
    allowed = np.flatnonzero((cr[:, 2]/nc <= .01) & (cr[:, 1] >= C['calibration']['min_R']) & (utility > 0))
    if not len(allowed):
        return dict(threshold=None, R=0, D=0, W=0, utility=0., reason='no_feasible_positive_utility_source_rule')
    j = max(allowed, key=lambda i: (utility[i], -cr[i, 2], cr[i, 0]))
    return dict(threshold=float(cr[j, 0]), R=int(cr[j, 1]), D=int(cr[j, 2]), W=int(cr[j, 3]), utility=float(utility[j]), source_NAR=float(cr[j, 2]/nc), reason='fixed_source_utility_rule')


def fit():
    verify()
    save('metric_preflight.json', preflight())
    source_features()
    fitted, calibration = {}, {}
    fit_rows = []
    for seed in C['seeds']:
        for sl in C['slices']:
            train = read(RUN/'artifacts'/f'source_features_train_{seed}_{sl}.npz')
            valid = read(RUN/'artifacts'/f'source_features_valid_{seed}_{sl}.npz')
            mask = train['types'] > 0
            models = {}
            for arm, cols in [('margin_control', [0]), ('confirmation_primary', [0, 1, 2])]:
                x = train['X'][mask][:, cols]
                mean, sd = x.mean(0), np.maximum(x.std(0), C['fit']['sd_floor'])
                xx = (x-mean)/sd
                m = LogisticRegression(C=C['fit']['C'], class_weight='balanced', solver='lbfgs', max_iter=C['fit']['max_iter'], tol=C['fit']['tol'])
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    m.fit(xx, train['types'][mask])
                assert np.array_equal(m.classes_, [1, 2, 3])
                model = dict(columns=cols, mean=mean.tolist(), sd=sd.tolist(), coef=m.coef_.tolist(), intercept=m.intercept_.tolist(), classes=m.classes_.tolist(), n_iter=m.n_iter_.tolist())
                g = m.decision_function(xx)
                u = g[:, 0]-logsumexp(g[:, 1:], axis=1)
                assert np.allclose(projection(train['X'][mask], model), u, atol=1e-12)
                model.update(u_mean=float(u.mean()), u_sd=max(float(u.std()), C['fit']['sd_floor']))
                models[arm] = model
                fit_rows.append(dict(seed=seed, slice=sl, arm=arm, train_rows=int(mask.sum()), parameter_count=int(m.coef_.size+m.intercept_.size), n_iter=int(m.n_iter_[0]), query_seen=False))
            key = f'{seed}_{sl}'
            fitted[key] = models
            ss = scores(valid, models)
            calibration[key] = {k: calibrate(valid, v) for k, v in ss.items()}
            for role, d in [('train', train), ('valid', valid)]:
                ss = scores(d, models)
                np.savez_compressed(RUN/'artifacts'/f'source_scores_{role}_{seed}_{sl}.npz', **ss)
    (RUN/'checkpoints/rerank_models.json').write_text(json.dumps(fitted, indent=2, allow_nan=False)+'\n')
    save('source_calibrated_thresholds.json', calibration)
    save('training_summary.json', fit_rows)
    # Freeze every fitted/calibrated object before opening query arrays.
    paths = [RUN/'checkpoints/rerank_models.json', RUN/'artifacts/source_calibrated_thresholds.json'] + list((RUN/'artifacts').glob('source_features_*.npz')) + list((RUN/'artifacts').glob('source_scores_*.npz'))
    save('source_fit_freeze.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})
    save('source_fit_complete.json', dict(passed=True, linear_models=len(fit_rows), query_opened=False, model_sha256=sha(RUN/'checkpoints/rerank_models.json')))


def fitverify():
    assert json.loads((RUN/'artifacts/source_fit_complete.json').read_text())['passed']
    for p, h in json.loads((RUN/'artifacts/source_fit_freeze.json').read_text()).items():
        assert sha(ROOT/p) == h, p


def policy_metrics(y, b, c, score, threshold):
    mask = (b != c) & (score >= threshold) if threshold is not None else np.zeros(len(y), bool)
    p = np.where(mask, c, b)
    nc, nw = int((b == y).sum()), int((b != y).sum())
    r = int((mask & (b != y) & (c == y)).sum())
    d = int((mask & (b == y)).sum())
    w = int((mask & (b != y) & (c != y)).sum())
    mat = np.zeros((102, 102), dtype=np.int64)
    np.add.at(mat, (y, p), 1)
    den = mat.sum(0)+mat.sum(1)
    f1 = np.divide(2*mat.diagonal(), den, out=np.zeros(102), where=den > 0).mean()
    return dict(R=r, D=d, W=w, NAR=d/nc, Recovery=r/nw, accuracy=float((p == y).mean()), macro_f1=float(f1), accepted=int(mask.sum()), threshold=threshold)


def evaluate():
    verify()
    fitverify()
    models = json.loads((RUN/'checkpoints/rerank_models.json').read_text())
    thresholds = json.loads((RUN/'artifacts/source_calibrated_thresholds.json').read_text())
    oldcells = json.loads((ROOT/C['v12']/'artifacts/cells.json').read_text())
    cells, matched, policies, curves = [], [], [], {}
    for seed in C['seeds']:
        for sl in C['slices']:
            old = read(ROOT/C['v13']/'artifacts'/f'rows_query_{seed}_{sl}.npz')
            d = dict(X=np.stack([old['v9_margin'], old['candidate_logprob'], old['third_margin']], -1), v9_margin=old['v9_margin'], v12_primary=old['margin_plus_soft'], labels=old['labels'], base=old['base'], candidates=old['candidates'], rows=old['rows'])
            t = kind(d['labels'], d['base'], d['candidates'])
            ss = scores(d, models[f'{seed}_{sl}'])
            local = {}
            for score, v in ss.items():
                met, _, _ = curve(d['labels'], d['base'], d['candidates'], v)
                audit_direct(d['labels'], d['base'], d['candidates'], v, met)
                if score in ['v9_margin', 'v12_primary']:
                    expected = next(r for r in oldcells if r['partition'] == 'query' and r['seed'] == seed and r['slice'] == sl and r['score'] == ('v9_margin' if score == 'v9_margin' else 'margin_plus_soft'))
                    for k, value in met.items():
                        assert expected[k] is None if value is None else abs(expected[k]-value) < 1e-12, (k, score)
                row = dict(seed=seed, slice=sl, score=score, **met)
                cells.append(row)
                local[score] = row
                curves[f'{seed}_{sl}_{score}'] = prefix(t, v)
                threshold = thresholds[f'{seed}_{sl}'][score]['threshold']
                policies.append(dict(seed=seed, slice=sl, score=score, **policy_metrics(d['labels'], d['base'], d['candidates'], v, threshold)))
            b = local['v12_primary']
            for score in C['scores']:
                cr = curves[f'{seed}_{sl}_{score}']
                for mode in ['v12_actual_damage', 'v12_corrections']:
                    if mode == 'v12_actual_damage':
                        allowed = np.flatnonzero(cr[:, 2] <= b['damaged'])
                        j = int(allowed[np.argmax(cr[allowed, 1])])
                    else:
                        allowed = np.flatnonzero((cr[:, 1] >= b['corrected']) & (cr[:, 2]/b['frozen_correct'] <= .01))
                        j = int(allowed[0]) if len(allowed) else None
                    r = dict(seed=seed, slice=sl, score=score, mode=mode, feasible=j is not None, baseline_R=b['corrected'], baseline_D=b['damaged'], baseline_W=b['other_wrong_accepted'])
                    if j is not None:
                        R, D, W = [int(x) for x in cr[j, 1:]]
                        r.update(R=R, D=D, W=W, R_overshoot=R-b['corrected'], Recovery=R/b['frozen_wrong'], NAR=D/b['frozen_correct'])
                    matched.append(r)
            np.savez_compressed(RUN/'artifacts'/f'query_scores_{seed}_{sl}.npz', **ss, labels=d['labels'], base=d['base'], candidates=d['candidates'], rows=d['rows'])
    save('cells.json', cells)
    save('matched_frontiers.json', matched)
    save('source_policy_target.json', policies)
    for name, rows in [('cells', cells), ('matched_frontiers', matched), ('source_policy_target', policies)]:
        csvout(name+'.csv', rows)
    np.savez_compressed(RUN/'artifacts/frontiers.npz', **curves)
    fitverify()


def audit():
    verify()
    fitverify()
    models = json.loads((RUN/'checkpoints/rerank_models.json').read_text())
    for seed in C['seeds']:
        for sl in C['slices']:
            for role in ['train', 'valid']:
                d = read(RUN/'artifacts'/f'source_features_{role}_{seed}_{sl}.npz')
                ss = scores(d, models[f'{seed}_{sl}'])
                saved = read(RUN/'artifacts'/f'source_scores_{role}_{seed}_{sl}.npz')
                assert all(np.array_equal(saved[k], v) for k, v in ss.items())
    cells = json.loads((RUN/'artifacts/cells.json').read_text())
    checked = 0
    for seed in C['seeds']:
        for sl in C['slices']:
            old = read(ROOT/C['v13']/'artifacts'/f'rows_query_{seed}_{sl}.npz')
            d = dict(X=np.stack([old['v9_margin'], old['candidate_logprob'], old['third_margin']], -1), v9_margin=old['v9_margin'], v12_primary=old['margin_plus_soft'])
            saved = read(RUN/'artifacts'/f'query_scores_{seed}_{sl}.npz')
            assert np.array_equal(saved['candidates'], old['candidates']) and np.array_equal(saved['rows'], old['rows'])
            for score, values in scores(d, models[f'{seed}_{sl}']).items():
                assert np.array_equal(values, saved[score])
                met, _, pred = curve(saved['labels'], saved['base'], saved['candidates'], values)
                record = next(r for r in cells if r['seed'] == seed and r['slice'] == sl and r['score'] == score)
                for k, v in met.items():
                    assert record[k] is None if v is None else abs(record[k]-v) < 1e-12, k
                direct = policy_metrics(saved['labels'], saved['base'], saved['candidates'], values, met['threshold'])
                assert direct['R'] == met['corrected'] and direct['D'] == met['damaged'] and direct['W'] == met['other_wrong_accepted']
                assert abs(direct['macro_f1']-met['diag_f1']) < 1e-12
                checked += 1
    assert checked == 24
    save('rerank_reload_audit.json', dict(passed=True, loaded_linear_models=12, source_states=12, query_metric_states=24, baseline_states_exact=12, source_fit_hashes_unchanged=True, candidate_rows_unchanged=True))


if __name__ == '__main__':
    import sys
    {'fit': fit, 'evaluate': evaluate, 'audit': audit}[sys.argv[1]]()
