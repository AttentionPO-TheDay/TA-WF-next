"""Frozen cached diagnostics; no model imports, fitting, or new policy outputs."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())
A = RUN / 'artifacts'
NAMES = {1: 'R', 2: 'D', 3: 'W'}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    (A / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvout(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (A / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cachepath(part, sl, seed, view):
    if view == 'original':
        stem = f'sourcevalid_{seed}_direction_iat' if part == 'valid' else f'query_{seed}_{sl}_direction_iat'
        return ROOT / C['v10'] / 'artifacts' / f'{stem}.npz'
    stem = 'valid' if part == 'valid' else sl + '_query'
    return ROOT / C['v11'] / 'artifacts' / f'logits_{seed}_direction_iat_{stem}_{view}.npz'


def verify():
    for p, h in json.loads((A / 'freeze.json').read_text()).items():
        assert sha(ROOT / p) == h, p


def freeze():
    assert C['status'] == 'frozen'
    parent = ROOT / C['v12']
    # Validate historical records before using them as evidence.
    delivery = json.loads((parent / 'artifacts/delivery_audit.json').read_text())
    assert delivery['passed']
    for p, h in delivery['outputs'].items():
        assert sha(parent / p) == h, p
    old = json.loads((parent / 'artifacts/freeze.json').read_text())
    for p, h in old.items():
        assert sha(ROOT / p) == h, p
    inputs = [RUN / x for x in ['PLAN.md', 'config.json', 'SOURCES.md', 'audit.py']]
    inputs += [ROOT / x for x in ['PROTOCOL.md', 'HANDOFF.md', 'configs/datasets.json']]
    inputs += [parent / 'artifacts' / x for x in ['freeze.json', 'delivery_audit.json', 'source_calibration.json', 'cells.json']]
    for part in C['partitions']:
        for sl in C['slices']:
            for seed in C['seeds']:
                inputs.append(parent / 'artifacts' / f'scores_{part}_{seed}_{sl}.npz')
                for ss in C['seeds']:
                    inputs += [cachepath(part, sl, ss, v) for v in C['views']]
    inputs = sorted(set(inputs))
    save('freeze.json', {str(p.relative_to(ROOT)): sha(p) for p in inputs})
    save('reuse_audit.json', dict(passed=True, frozen_inputs=len(inputs), parent_outputs_verified=len(delivery['outputs']), parent_inputs_verified=len(old), old_run_unchanged=True))
    verify()


def loadstate(part, sl, seed):
    with np.load(ROOT / C['v12'] / 'artifacts' / f'scores_{part}_{seed}_{sl}.npz') as z:
        return {k: z[k] for k in ['rows', 'labels', 'base', 'candidates'] + C['scores']}


def features(part, sl, s, cal):
    b, c, y = s['base'], s['candidates'], s['labels']
    n = len(b)
    ix = np.arange(n)
    sums = {k: np.zeros(n) for k in C['features']}
    mean_prob = np.zeros((n, 102))
    odds2 = np.zeros(n)
    abs_old_prob = np.zeros(n)
    weights = np.array(cal['weights'])
    temps = np.array(cal['temperatures'])
    for si, seed in enumerate(C['seeds']):
        for vi, view in enumerate(C['views']):
            with np.load(cachepath(part, sl, seed, view)) as z:
                assert np.array_equal(z['rows'], s['rows']) and np.array_equal(z['labels'], y)
                q = z['logits'].astype(np.float64) / temps[si, vi]
            w = weights[si, vi]
            lse = logsumexp(q, axis=-1)
            lp_c, lp_b = q[ix, c] - lse, q[ix, b] - lse
            odds = q[ix, c] - q[ix, b]
            sums['candidate_logprob'] += w * lp_c
            sums['candidate_probability'] += w * np.exp(lp_c)
            sums['old_rejection'] -= w * lp_b
            sums['mean_logodds'] += w * odds
            sums['candidate_vote'] += w * (q.argmax(-1) == c)
            mean_prob += w * np.exp(q - lse[:, None])
            abs_old_prob += w * np.exp(lp_b)
            qc, qb = q[ix, c].copy(), q[ix, b].copy()
            q[ix, c] = -np.inf
            allmax = q.max(-1)
            q[ix, b] = -np.inf
            thirdmax = q.max(-1)
            sums['all_competitor_margin'] += w * (qc - allmax)
            sums['third_margin'] += w * (qc - thirdmax)
            odds2 += w * odds**2
    sums['odds_stability'] = -np.sqrt(np.maximum(odds2 - sums['mean_logodds']**2, 0))
    # Two independent ways to obtain the same probability ratio.
    assert np.allclose(sums['mean_logodds'], sums['candidate_logprob'] + sums['old_rejection'], atol=1e-12)
    sums['aux_winner'] = mean_prob.argmax(-1)
    sums['old_probability'] = abs_old_prob
    assert all(np.isfinite(v).all() for v in sums.values())
    return sums


def taxonomy(y, b, c):
    change = b != c
    t = np.zeros(len(y), dtype=np.int8)
    t[change & (b != y) & (c == y)] = 1
    t[change & (b == y)] = 2
    t[change & (b != y) & (c != y)] = 3
    assert np.array_equal(t > 0, change)
    return t


def counts(mask, t):
    r = {v: int((mask & (t == k)).sum()) for k, v in NAMES.items()}
    r['total'] = int(mask.sum())
    assert r['total'] == r['R'] + r['D'] + r['W']
    r['W_per_R'] = r['W'] / r['R'] if r['R'] else None
    r['R_precision'] = r['R'] / r['total'] if r['total'] else None
    return r


def frontier(s, t):
    ix = np.flatnonzero(t > 0)
    ix = ix[np.argsort(-s[ix], kind='stable')]
    ends = np.flatnonzero(np.r_[s[ix][1:] != s[ix][:-1], True])
    thresholds = np.r_[np.nextafter(s[ix].max(), np.inf), s[ix][ends]]
    cr = [np.r_[0, np.cumsum(t[ix] == k)[ends]] for k in [1, 2, 3]]
    return np.stack([thresholds] + cr, axis=-1)


def maskfrom(s, t, threshold):
    return (t > 0) & (s >= threshold) if threshold is not None else np.zeros(len(t), bool)


def boundary(s, t, accepted):
    ix = np.flatnonzero(t > 0)
    ix = ix[np.argsort(-s[ix], kind='stable')]
    k = int(accepted.sum())
    radius = max(C['boundary_min_radius'], math.ceil(C['boundary_fraction'] * len(ix)))
    lo, hi = max(0, k - radius), min(len(ix), k + radius)
    return (t > 0) & (s >= s[ix[hi - 1]]) & (s <= s[ix[lo]])


def auc_rank(pos, x):
    a, b = int(pos.sum()), int((~pos).sum())
    if not a or not b:
        return None, 0
    ranks = rankdata(x, method='average')
    u = float(ranks[pos].sum() - a * (a + 1) / 2)
    return u / (a * b), a * b


def rawpartial(pos, x):
    fpr, tpr, _ = roc_curve(pos, x, drop_intermediate=False)
    alpha = C['alpha']
    j = np.searchsorted(fpr, alpha, side='right')
    xx, yy = fpr[:j], tpr[:j]
    if xx[-1] < alpha:
        xx, yy = np.r_[xx, alpha], np.r_[yy, np.interp(alpha, fpr, tpr)]
    return float(np.sum(np.diff(xx) * (yy[:-1] + yy[1:]) / 2) / alpha)


def metric(pos, x, bins=None):
    npos, nneg = int(pos.sum()), int((~pos).sum())
    a, pairs = auc_rank(pos, x)
    result = dict(n_R=npos, n_negative=nneg, limited=min(npos, nneg) < C['min_group_n'])
    if bins is None:
        if a is not None:
            assert abs(a - roc_auc_score(pos, x)) < 1e-12
        result.update(auc=a, comparable_pairs=pairs, pair_coverage=1. if pairs else 0., ap=float(average_precision_score(pos, x)) if a is not None else None, prevalence=float(pos.mean()) if len(pos) else None, pauc_norm=rawpartial(pos, x) if a is not None else None)
        return result
    num, den, groups = 0., 0, 0
    for k in np.unique(bins):
        m = bins == k
        aa, pp = auc_rank(pos[m], x[m])
        if pp:
            num += aa * pp
            den += pp
            groups += 1
    result.update(auc=num / den if den else None, comparable_pairs=den, pair_coverage=den / pairs if pairs else 0., comparable_bins=groups, ap=None, prevalence=float(pos.mean()) if len(pos) else None, pauc_norm=None)
    return result


def groupdetails(meta, s, f, t, mask, group, rows, classes, evidence):
    row = dict(meta, group=group, **counts(mask, t))
    rows.append(row)
    for kind in ['R', 'D', 'W']:
        k = next(k for k, v in NAMES.items() if v == kind)
        m = mask & (t == k)
        n = int(m.sum())
        if not n:
            continue
        er = dict(meta, group=group, kind=kind, n=n, third_margin_negative=float((f['third_margin'][m] < 0).mean()), auxiliary_third_winner=float(((f['aux_winner'][m] != s['base'][m]) & (f['aux_winner'][m] != s['candidates'][m])).mean()), auxiliary_candidate_winner=float((f['aux_winner'][m] == s['candidates'][m]).mean()))
        for feature in C['features'] + ['old_probability']:
            er[feature + '_mean'] = float(f[feature][m].mean())
            er[feature + '_median'] = float(np.median(f[feature][m]))
        evidence.append(er)
        for axis in ['labels', 'base', 'candidates']:
            cs = np.bincount(s[axis][m], minlength=102)
            top = np.argsort(-cs, kind='stable')[:5]
            for label in np.flatnonzero(cs):
                classes.append(dict(meta, group=group, kind=kind, axis=axis, label=int(label), count=int(cs[label]), group_n=n, top5_share=float(cs[top].sum() / n)))


def run():
    start = time.monotonic()
    verify()
    cal = json.loads((ROOT / C['v12'] / 'artifacts/source_calibration.json').read_text())
    oldcells = json.loads((ROOT / C['v12'] / 'artifacts/cells.json').read_text())
    groups, classes, evidence, metrics, matches, operations, correlations = [], [], [], [], [], [], []
    curves = {}
    check_states = 0
    for part in C['partitions']:
        for sl in C['slices']:
            for seed in C['seeds']:
                assert time.monotonic() - start < C['budget']['wall_seconds']
                s = loadstate(part, sl, seed)
                f = features(part, sl, s, cal)
                with np.load(ROOT / C['v12'] / 'artifacts' / f'scores_{part}_{seed}_{sl}.npz') as previous:
                    assert np.allclose(f['mean_logodds'], previous['weighted_logodds'], atol=1e-12)
                    assert np.allclose(f['mean_logodds'] + .25 * f['odds_stability'], previous['weighted_soft'], atol=1e-10)
                t = taxonomy(s['labels'], s['base'], s['candidates'])
                meta = dict(partition=part, slice=sl, seed=seed)
                masks = {}
                base_correct = int((s['base'] == s['labels']).sum())
                base_wrong = len(t) - base_correct
                for score in C['scores']:
                    old = next(r for r in oldcells if all(r[k] == v for k, v in meta.items()) and r['score'] == score)
                    m = maskfrom(s[score], t, old['threshold'])
                    cnt = counts(m, t)
                    assert cnt['R'] == old['corrected'] and cnt['D'] == old['damaged'] and cnt['W'] == old['other_wrong_accepted']
                    assert cnt['D'] / base_correct == old['attained_nar1']
                    masks[score] = m
                    cr = frontier(s[score], t)
                    # Independently reconstruct the original maximum feasible R, minimal-prefix tie break.
                    feasible = np.flatnonzero(cr[:, 2] / base_correct <= C['alpha'])
                    best = feasible[np.argmax(cr[feasible, 1])]
                    assert np.array_equal(cr[best, 1:], [cnt['R'], cnt['D'], cnt['W']])
                    curves[f'{part}_{seed}_{sl}_{score}'] = cr
                    operations.append(dict(meta, score=score, **cnt, NAR=cnt['D'] / base_correct, Recovery=cnt['R'] / base_wrong, frozen_correct=base_correct, frozen_wrong=base_wrong))
                    check_states += 1
                b, p = masks['v9_margin'], masks['margin_plus_soft']
                for group, mask in [('common', b & p), ('added', p & ~b), ('removed', b & ~p), ('rejected', (t > 0) & ~b & ~p), ('v9_accepted', b), ('v12_accepted', p)]:
                    groupdetails(meta, s, f, t, mask, group, groups, classes, evidence)
                bcnt = counts(b, t)
                for score in C['scores']:
                    cr = curves[f'{part}_{seed}_{sl}_{score}']
                    for mode in ['baseline_actual_NAR', 'baseline_corrections']:
                        if mode == 'baseline_actual_NAR':
                            feasible = np.flatnonzero(cr[:, 2] <= bcnt['D'])
                            j = feasible[np.argmax(cr[feasible, 1])]
                        else:
                            feasible = np.flatnonzero((cr[:, 1] >= bcnt['R']) & (cr[:, 2] / base_correct <= C['alpha']))
                            j = int(feasible[0]) if len(feasible) else None
                        r = dict(meta, score=score, match=mode, feasible=j is not None, baseline_R=bcnt['R'], baseline_D=bcnt['D'], baseline_W=bcnt['W'])
                        if j is not None:
                            rr, dd, ww = [int(v) for v in cr[j, 1:]]
                            r.update(R=rr, D=dd, W=ww, R_overshoot=rr - bcnt['R'], NAR=dd / base_correct, Recovery=rr / base_wrong, W_per_R=ww / rr if rr else None)
                        matches.append(r)
                bd = boundary(s['v9_margin'], t, b) | boundary(s['margin_plus_soft'], t, p)
                cohorts = dict(changed=t > 0, accepted_union=b | p, added=p & ~b, boundary=bd)
                for cohort, mask in cohorts.items():
                    for comp, neg in [('R_vs_W', 3), ('R_vs_D', 2)]:
                        m = mask & ((t == 1) | (t == neg))
                        pos = t[m] == 1
                        # Bin boundaries depend only on the full unlabeled cohort, not on outcome selection.
                        for conditioner in C['conditioners']:
                            bins = None
                            if conditioner != 'none' and mask.any():
                                edges = np.quantile(s[conditioner][mask], np.linspace(0, 1, C['bins'] + 1)[1:-1])
                                bins = np.searchsorted(edges, s[conditioner][m], side='right')
                            for feature in C['features']:
                                metrics.append(dict(meta, cohort=cohort, comparison=comp, conditioner=conditioner, feature=feature, **metric(pos, f[feature][m], bins)))
                for feature in C['features']:
                    for score in C['scores']:
                        if bd.sum() > 1 and np.std(f[feature][bd]) > 0 and np.std(s[score][bd]) > 0:
                            r = float(spearmanr(f[feature][bd], s[score][bd]).statistic)
                        else:
                            r = None
                        correlations.append(dict(meta, feature=feature, conditioner=score, boundary_spearman=r))
                np.savez_compressed(A / f'rows_{part}_{seed}_{sl}.npz', **s, **f, type=t, v9_accepted=b, v12_accepted=p, boundary=bd)
                print(json.dumps(dict(meta, changed=int((t > 0).sum()), boundary=int(bd.sum()), added=counts(p & ~b, t))), flush=True)
    assert check_states == 24
    for name, rows in [('groups', groups), ('class_counts', classes), ('evidence', evidence), ('signals', metrics), ('matched_frontiers', matches), ('operations', operations), ('correlations', correlations)]:
        save(name + '.json', rows)
        csvout(name + '.csv', rows)
    np.savez_compressed(A / 'frontiers.npz', **curves)
    verify()
    save('analysis_audit.json', dict(passed=True, original_operations_exact=check_states, states=12, candidate_pool_unchanged=True, label_fitted_parameters=0, new_threshold_or_score_selected=False, seconds=time.monotonic() - start))


def report():
    data = {k: json.loads((A / (k + '.json')).read_text()) for k in ['groups', 'signals', 'matched_frontiers', 'operations', 'evidence']}
    summaries = []
    for sl in C['slices']:
        for cohort in C['cohorts']:
            for comp in C['comparisons']:
                for cond in C['conditioners']:
                    for feature in C['features']:
                        rr = [r for r in data['signals'] if r['partition'] == 'query' and r['slice'] == sl and r['cohort'] == cohort and r['comparison'] == comp and r['conditioner'] == cond and r['feature'] == feature]
                        assert len(rr) == 3
                        vv = [r['auc'] for r in rr if r['auc'] is not None]
                        summaries.append(dict(slice=sl, cohort=cohort, comparison=comp, conditioner=cond, feature=feature, auc_mean=float(np.mean(vv)) if vv else None, auc_sd=float(np.std(vv, ddof=1)) if len(vv) > 1 else None, defined_seeds=len(vv), above_half_seeds=sum(v > .5 for v in vv), nonlimited_seeds=sum(not r['limited'] for r in rr), min_R=min(r['n_R'] for r in rr), min_negative=min(r['n_negative'] for r in rr), comparable_pairs=sum(r['comparable_pairs'] for r in rr)))
    save('signal_summary.json', summaries)
    csvout('signal_summary.csv', summaries)
    lines = ['# 候选清洗审查 v13', '', '已完成固定 v9/v12 缓存审查；无训练、无新前向、无新评分或部署阈值。以下是已观察两切片的事后诊断，不是独立确认。', '', '## 接纳集合变化', '', 'R=原错→正确，D=原对→错误，W=原错→其他错。下表为三个源 seed 的均值；逐 seed 明细全部保留。', '', '|切片|集合|R|D|W|', '|---|---|---:|---:|---:|']
    for sl in C['slices']:
        for group in ['v9_accepted', 'v12_accepted', 'common', 'added', 'removed']:
            rr = [r for r in data['groups'] if r['partition'] == 'query' and r['slice'] == sl and r['group'] == group]
            vals = [float(np.mean([r[k] for r in rr])) for k in ['R', 'D', 'W']]
            lines.append('|' + '|'.join([sl, group] + [f'{x:.3f}' for x in vals]) + '|')
    lines += ['', '## 匹配原基线纠错数与实际 NAR', '', '同纠错数采用达到原 v9 纠错数的最小完整同分前缀，且 NAR≤1%；若同分导致超出，R_overshoot 在机器表中明确记录。同 NAR 使用原 v9 实际 D 计数为预算，不假定恰好1%。', '', '|切片|匹配目标|评分|可行seed|R|D|W|', '|---|---|---|---:|---:|---:|---:|']
    for sl in C['slices']:
        for mode in ['baseline_corrections', 'baseline_actual_NAR']:
            for score in C['scores']:
                rr = [r for r in data['matched_frontiers'] if r['partition'] == 'query' and r['slice'] == sl and r['match'] == mode and r['score'] == score and r['feasible']]
                vals = [float(np.mean([r[k] for r in rr])) for k in ['R', 'D', 'W']] if rr else [None] * 3
                lines.append('|' + '|'.join([sl, mode, score, str(len(rr))] + [f'{x:.3f}' if x is not None else 'NA' for x in vals]) + '|')
    lines += ['', '## 边界内可分性', '', '正例R、负例W；固定方向越大越支持接纳。分层AUROC只比较同一裕度/主评分十分位中的正负样本对，不能解释成新评分器的部署效果。主表为seed均值；样本量和每seed结果见signals.csv。pAUC、AP及R vs D对照全部保留在机器表中。', '', '|信号|Version原始AUC|Version按v9分层|Version按v12分层|USA原始AUC|USA按v9分层|USA按v12分层|', '|---|---:|---:|---:|---:|---:|---:|']
    for feature in C['features']:
        vals = []
        for sl in C['slices']:
            for cond in C['conditioners']:
                r = next(r for r in summaries if r['slice'] == sl and r['feature'] == feature and r['cohort'] == 'boundary' and r['comparison'] == 'R_vs_W' and r['conditioner'] == cond)
                vals.append(r['auc_mean'])
        lines.append('|' + '|'.join([feature] + [f'{v:.4f}' if v is not None else 'NA' for v in vals]) + '|')
    lines += ['', '## 权限与解释边界', '', '边界集合依赖 v12 已有 query 标签选择的操作点，因此以上是条件化诊断，不是无监督部署协议。源valid在两目标状态中重复，只作控制，不能与query或彼此合并为独立重复。', '', '没有用标签训练评分器、翻转特征方向、选择新阈值或给出误改概率。分层AUROC的条件化不构成因果证明；小样本分层、网站间依赖和仅3个seed限制结论，不作显著性声明。', '', '全部类别分布（真实/旧预测/新候选）、新增/移除接纳、辅助第三类获胜情况、相关性、两种错误对照及sourcevalid结果均已保存，不能只摘取有利特征。原评分和候选类别保持不变。']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sl in zip(axes, C['slices']):
        mat = []
        for feat in C['features']:
            rr = [next(r for r in summaries if r['slice'] == sl and r['cohort'] == 'boundary' and r['comparison'] == 'R_vs_W' and r['conditioner'] == cond and r['feature'] == feat)['auc_mean'] for cond in C['conditioners']]
            mat.append([np.nan if x is None else x for x in rr])
        im = ax.imshow(mat, vmin=.3, vmax=.8, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(3), ['Unconditioned', 'Within v9 bins', 'Within v12 bins'], rotation=15)
        ax.set_yticks(range(len(C['features'])), C['features'], fontsize=8)
        for i, row in enumerate(mat):
            for j, v in enumerate(row):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8)
        ax.set_title(sl + ': boundary R vs W')
    fig.suptitle('Descriptive AUROC; cached diagnostic only')
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(RUN / ('figure_boundary.' + ext), dpi=160)
    plt.close(fig)


def check():
    verify()
    oldcells = json.loads((ROOT / C['v12'] / 'artifacts/cells.json').read_text())
    groups = json.loads((A / 'groups.json').read_text())
    signals = json.loads((A / 'signals.json').read_text())
    checked = 0
    for p in sorted(A.glob('rows_*.npz')):
        with np.load(p) as z:
            part, seed, sl = p.stem.split('_', 3)[1:]
            seed = int(seed)
            y, b, c = z['labels'], z['base'], z['candidates']
            t = z['type']
            assert np.array_equal(t, taxonomy(y, b, c))
            for k in z.files:
                if np.issubdtype(z[k].dtype, np.floating):
                    assert np.isfinite(z[k]).all(), (p, k)
            bc, pc = z['v9_accepted'], z['v12_accepted']
            for mask, name in [(bc & pc, 'common'), (pc & ~bc, 'added'), (bc & ~pc, 'removed'), ((t > 0) & ~bc & ~pc, 'rejected'), (bc, 'v9_accepted'), (pc, 'v12_accepted')]:
                rr = next(r for r in groups if r['partition'] == part and r['seed'] == seed and r['slice'] == sl and r['group'] == name)
                assert counts(mask, t) == {k: rr[k] for k in counts(mask, t)}
            for score, mask in [('v9_margin', bc), ('margin_plus_soft', pc)]:
                rr = next(r for r in oldcells if r['partition'] == part and r['seed'] == seed and r['slice'] == sl and r['score'] == score)
                assert np.array_equal(mask, maskfrom(z[score], t, rr['threshold']))
            assert np.allclose(z['mean_logodds'], z['candidate_logprob'] + z['old_rejection'], atol=1e-12)
            # Independent sklearn check of all conditional pair sums from persisted rows.
            for feature in C['features']:
                mask = z['boundary']
                for cond in C['scores']:
                    edges = np.quantile(z[cond][mask], np.linspace(0, 1, C['bins'] + 1)[1:-1])
                    bins = np.searchsorted(edges, z[cond], side='right')
                    for neg, comp in [(3, 'R_vs_W'), (2, 'R_vs_D')]:
                        num, den = 0., 0
                        for k in range(C['bins']):
                            m = mask & (bins == k) & ((t == 1) | (t == neg))
                            yy = t[m] == 1
                            pairs = int(yy.sum()) * int((~yy).sum())
                            if pairs:
                                num += pairs * roc_auc_score(yy, z[feature][m])
                                den += pairs
                        rr = next(r for r in signals if r['partition'] == part and r['slice'] == sl and r['seed'] == seed and r['feature'] == feature and r['conditioner'] == cond and r['cohort'] == 'boundary' and r['comparison'] == comp)
                        assert den == rr['comparable_pairs']
                        assert (rr['auc'] is None) if not den else abs(rr['auc'] - num / den) < 1e-12
            checked += 1
    assert checked == 12
    save('postrun_audit.json', dict(passed=True, independently_reloaded_states=checked, boundary_conditional_auc_independent_sklearn=True, original_thresholds_and_error_counts_exact=True, no_fitted_query_parameters=True, old_inputs_unchanged=True))


if __name__ == '__main__':
    import sys
    {'freeze': freeze, 'run': run, 'report': report, 'check': check}[sys.argv[1]]()
