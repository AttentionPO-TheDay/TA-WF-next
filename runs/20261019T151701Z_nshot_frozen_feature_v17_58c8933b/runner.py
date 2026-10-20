"""Fixed stages: structure -> frozen features -> all fits -> terminal scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from model import MultiViewModel
from maintenance import (unit, class_means, fit_proto, fit_head, predict, metrics,
                         margin_gain, frontier, best_under_damage, least_damage_for_recovery)

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())
ART = RUN / 'artifacts'
CK = RUN / 'checkpoints'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    path = ART / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def npz(path, **values):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **values)
    temp.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def registry(state, summary):
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status', '--id', RUN.name,
                    '--state', state, '--summary', summary], check=True)
    import fcntl
    with (ROOT / '.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / 'STATUS.md'
        lines = path.read_text().splitlines()
        prefix = f'n-shot冻结表征 v17：`{RUN.name}`。'
        lines = [line for line in lines if not line.startswith(prefix)]
        lines.insert(1, prefix + summary)
        for i, line in enumerate(lines):
            if line.startswith('当前研究动作：'):
                lines[i] = '当前研究动作：' + prefix + summary + ' WTT/AWF关闭，旧实验封存不变。'
        path.write_text('\n'.join(lines) + '\n')


def prepared(path):
    return torch.load(ROOT / path, map_location='cpu', weights_only=True, mmap=True)


def split(role):
    return read_npz(ROOT / C['split_run'] / 'artifacts' / f'split_{role}.npz')


def fingerprints(values):
    return [hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest() for row in values]


def verify():
    assert C['status'] == 'frozen' and C['id'] == RUN.name
    frozen = json.loads((ART / 'freeze.json').read_text())
    for path, expected in frozen.items():
        assert sha(ROOT / path) == expected, path


def freeze():
    assert not (ART / 'freeze.json').exists()
    paths = list(RUN.glob('*.py')) + [RUN / n for n in ['PLAN.md', 'SOURCES.md', 'config.json']]
    paths += [ROOT / C['dataset_config'], ROOT / C['source_run'] / 'artifacts/prepared.pt',
              ROOT / C['source_run'] / 'artifacts/reports.json', ROOT / C['source_run'] / 'artifacts/weight_audit.json']
    reports = json.loads((ROOT / C['source_run'] / 'artifacts/reports.json').read_text())
    assert json.loads((ROOT / C['source_run'] / 'artifacts/weight_audit.json').read_text())['passed']
    for seed in C['model_seeds']:
        path = ROOT / C['source_run'] / 'checkpoints' / f'pure_source_wide_bn_{seed}.pth'
        assert sha(path) == next(r['checkpoint_sha256'] for r in reports if r['seed'] == seed)
        paths.append(path)
        for role in C['targets']:
            paths.append(ROOT / C['baseline_run'] / 'artifacts' / f'pred_{seed}_{role}_frozen.npz')
    for role, path in C['targets'].items():
        paths += [ROOT / path, ROOT / C['split_run'] / 'artifacts' / f'split_{role}.npz']
    save('freeze.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})
    source = prepared(C['source_run'] + '/artifacts/prepared.pt')
    assert len(source['source']['rows']) == 18553
    source_fp = set(fingerprints(source['source']['timestamps'].numpy()))
    source_fp.update(fingerprints(source['valid']['timestamps'].numpy()))
    source_y = source['source']['labels'].numpy()
    source_rows = source['source']['rows'].numpy()
    source_idx = []
    for cls in range(102):
        pool = np.flatnonzero(source_y == cls)
        source_idx.extend(sorted(pool, key=lambda i: hashlib.sha256(f'v17-source|{int(source_rows[i])}'.encode()).hexdigest())[:C['source_features_per_class']])
    npz(ART / 'source_selection.npz', indices=np.array(source_idx), rows=source_rows[source_idx], labels=source_y[source_idx])
    audit = {}
    for role, path in C['targets'].items():
        data = prepared(path)
        s = split(role)
        si, qi = s['support_indices'], s['query_indices']
        assert len(si) == 1024 and not set(si) & set(qi)
        assert np.array_equal(data['rows'][si].numpy(), s['support'])
        assert np.array_equal(data['rows'][qi].numpy(), s['query'])
        sf = fingerprints(data['timestamps'][si].numpy())
        qf = fingerprints(data['timestamps'][qi].numpy())
        assert len(set(sf)) == len(sf) and len(set(qf)) == len(qf)
        assert source_fp.isdisjoint(set(sf) | set(qf)) and set(sf).isdisjoint(qf)
        y = data['labels'][si].numpy()  # Reference class index only, never query labels.
        counts = np.bincount(y, minlength=102)
        assert counts.min() >= 1
        budgets = {}
        for draw in C['support_seeds']:
            ordered = {cls: sorted(np.flatnonzero(y == cls), key=lambda i: hashlib.sha256(f'v17|{draw}|{role}|{sf[i]}'.encode()).hexdigest()) for cls in range(102)}
            prior = set()
            for n in C['nominal_shots']:
                idx = np.array([i for cls in range(102) for i in ordered[cls][:n]], dtype=np.int64)
                assert prior <= set(idx)
                prior = set(idx)
                used = np.bincount(y[idx], minlength=102)
                assert np.array_equal(used, np.minimum(counts, n))
                name = f'support_{role}_{draw}_cap{n}.npz'
                npz(ART / name, reference_positions=idx, prepared_indices=si[idx], rows=s['support'][idx], labels=y[idx], class_counts=used)
                budgets[str(n)] = {'labels': len(idx), 'strict_nshot': bool((used == n).all()), 'min_per_class': int(used.min()), 'deficit_vs_full': int(n * 102 - len(idx))}
        audit[role] = {'reference': len(si), 'query': len(qi), 'counts': counts.tolist(), 'budgets': budgets,
                       'reference_query_overlap': 0, 'source_valid_target_overlap': 0}
    save('preflight.json', {'passed': True, 'roles': audit, 'query_labels_accessed': False})
    registry('running', '训练前配置/代码/清单已冻结；严格1shot及cap3/5标签支持清单已固定，query未用于拟合。')


def load_model(seed):
    model = MultiViewModel('wide_bn', 102).cuda().eval().requires_grad_(False)
    ck = torch.load(ROOT / C['source_run'] / 'checkpoints' / f'pure_source_wide_bn_{seed}.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(ck['state_dict'], strict=True)
    return model


@torch.inference_mode()
def extract(model, data, indices, label):
    got = []
    logits = []
    captured = []
    def hook(module, args):
        captured.append(args[0].detach().cpu().numpy().copy())
    handle = model.classifier.register_forward_pre_hook(hook)
    began = time.monotonic()
    try:
        for start in range(0, len(indices), C['budget']['batch_size']):
            part = indices[start:start + C['budget']['batch_size']]
            captured.clear()
            z = model(data['timestamps'][part].cuda(), data['tam'][part].cuda()).cpu().numpy()
            assert len(captured) == 1
            got.append(captured[0]); logits.append(z)
            if start % 2048 == 0:
                progress = {'phase': 'frozen_features', 'cell': label, 'done': min(start + len(part), len(indices)), 'total': len(indices), 'seconds': time.monotonic() - began}
                save('progress.json', progress)
                print(json.dumps(progress), flush=True)
    finally:
        handle.remove()
    x, z = np.concatenate(got), np.concatenate(logits)
    assert np.isfinite(x).all() and np.isfinite(z).all()
    np.testing.assert_allclose(x @ model.classifier.weight.cpu().numpy().T + model.classifier.bias.cpu().numpy(), z, atol=.001, rtol=1e-5)
    return x, z


def features():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(C['budget']['feature_gpu'])
    torch.cuda.set_per_process_memory_fraction(C['budget']['cuda_memory_fraction'])
    selection = read_npz(ART / 'source_selection.npz')
    source = prepared(C['source_run'] + '/artifacts/prepared.pt')['source']
    costs = []
    for seed in C['model_seeds']:
        model = load_model(seed)
        before = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        began = time.monotonic()
        sx, sz = extract(model, source, selection['indices'], f'source_{seed}')
        mu = sx.mean(0); sd = np.maximum(sx.std(0), 1e-3)
        means, _ = class_means(unit((sx - mu) / sd), selection['labels'])
        w = model.classifier.weight.cpu().numpy().copy()
        b = model.classifier.bias.cpu().numpy().copy()
        npz(ART / f'source_features_{seed}.npz', features=sx, logits=sz, mu=mu, sd=sd, means=means,
            weight=w, bias=b, rows=selection['rows'], labels=selection['labels'])
        for role, path in C['targets'].items():
            data = prepared(path); s = split(role)
            for kind, idx in [('reference', s['support_indices']), ('query', s['query_indices'])]:
                x, z = extract(model, data, idx, f'{seed}_{role}_{kind}')
                if kind == 'query':
                    with np.load(ROOT / C['baseline_run'] / 'artifacts' / f'pred_{seed}_{role}_frozen.npz', allow_pickle=False) as old:
                        baseline = {k: old[k].copy() for k in ('logits', 'predictions', 'rows')}
                    # Historical labels deliberately not read for fitting.
                    np.testing.assert_allclose(z, baseline['logits'], atol=C['audit']['baseline_logit_atol'], rtol=C['audit']['baseline_logit_rtol'])
                    assert np.array_equal(z.argmax(1), baseline['predictions'])
                    assert np.array_equal(data['rows'][idx].numpy(), baseline['rows'])
                npz(ART / f'features_{seed}_{role}_{kind}.npz', features=x, logits=z, rows=data['rows'][idx].numpy())
            del data
        assert all(torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items())
        costs.append({'seed': seed, 'seconds': time.monotonic() - began, 'backbone_unchanged': True,
                      'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'checkpoint_sha256': sha(ROOT / C['source_run'] / 'checkpoints' / f'pure_source_wide_bn_{seed}.pth')})
        save('feature_costs.json', costs)
        del model
        torch.cuda.empty_cache()
    save('features_complete.json', {'passed': True, 'query_used_only_for_individual_frozen_forward': True})


def train():
    assert json.loads((ART / 'features_complete.json').read_text())['passed']
    CK.mkdir(exist_ok=True)
    fits = []
    for seed in C['model_seeds']:
        source = read_npz(ART / f'source_features_{seed}.npz')
        mu, sd = source['mu'], source['sd']
        w0, b0 = source['weight'] * sd[None, :], source['bias'] + source['weight'] @ mu
        sx = (source['features'] - mu) / sd
        np.testing.assert_allclose(sx @ w0.T + b0, source['logits'], atol=.001, rtol=1e-5)
        npz(CK / f'{seed}_source_proto.npz', kind=np.array('proto'), centers=unit(source['means']).astype(np.float32))
        for role in C['targets']:
            reference = read_npz(ART / f'features_{seed}_{role}_reference.npz')
            for draw in C['support_seeds']:
                for n in C['nominal_shots']:
                    support = read_npz(ART / f'support_{role}_{draw}_cap{n}.npz')
                    idx = support['reference_positions']
                    assert np.array_equal(reference['rows'][idx], support['rows'])
                    x, y = (reference['features'][idx] - mu) / sd, support['labels']
                    for arm in C['arms']:
                        began = time.monotonic()
                        key = f'{seed}_{role}_{draw}_cap{n}_{arm}'
                        if arm == 'head_l2':
                            model, history = fit_head(x, y, w0, b0, C['linear'])
                        else:
                            model = fit_proto(x, y, source['means'], C['prototype']['shrink_source_pseudocount'] if arm == 'shrink_proto' else 0)
                            history = []
                        path = CK / f'{key}.npz'
                        npz(path, **model)
                        fits.append({'key': key, 'seed': seed, 'role': role, 'draw': draw, 'cap': n, 'arm': arm,
                                     'labels': len(y), 'strict_nshot': bool((support['class_counts'] == n).all()),
                                     'checkpoint_sha256': sha(path), 'support_sha256': sha(ART / f'support_{role}_{draw}_cap{n}.npz'),
                                     'seconds': time.monotonic() - began, 'history': history})
                        save('fits.json', fits)
                        save('progress.json', {'phase': 'support_only_training', 'completed': len(fits), 'total': C['budget']['max_fit_cells'], 'last': key})
                        if len(fits) % 9 == 0:
                            print(json.dumps({'phase': 'support_only_training', 'completed': len(fits), 'last': key}), flush=True)
    assert len(fits) == C['budget']['max_fit_cells']
    save('fits_complete.json', {'passed': True, 'count': len(fits), 'query_labels_used': False, 'fit_manifest_sha256': sha(ART / 'fits.json')})
    registry('running', '162个support-only拟合状态完成（54个监督线性头）；进入固定query终点评价，不返调模型。')


def score():
    complete = json.loads((ART / 'fits_complete.json').read_text())
    assert complete['passed'] and complete['fit_manifest_sha256'] == sha(ART / 'fits.json')
    fits = json.loads((ART / 'fits.json').read_text())
    scores, matched = [], []
    for seed in C['model_seeds']:
        source = read_npz(ART / f'source_features_{seed}.npz')
        for role, path in C['targets'].items():
            raw = prepared(path); s = split(role)
            y = raw['labels'][s['query_indices']].numpy().copy()  # First new terminal query label access.
            query = read_npz(ART / f'features_{seed}_{role}_query.npz')
            assert np.array_equal(raw['rows'][s['query_indices']].numpy(), query['rows'])
            x = (query['features'] - source['mu']) / source['sd']
            original = query['logits']; base = original.argmax(1)
            base_metrics = metrics(y, base, base)
            wrong = int((base != y).sum()); correct = len(y) - wrong
            controls = []
            for arm in C['zero_label_controls']:
                z = original if arm == 'frozen' else predict(read_npz(CK / f'{seed}_source_proto.npz'), x)
                p = z.argmax(1)
                key = f'{seed}_{role}_{arm}'
                gain = margin_gain(z, original, p, base)
                curve = frontier(y, p, base, gain)
                at1 = best_under_damage(curve, int(np.floor(correct * .01)))
                row = {'key': key, 'seed': seed, 'role': role, 'draw': None, 'cap': 0, 'arm': arm, 'labels': 0,
                       'strict_nshot': False, **metrics(y, p, base), 'delta_f1_pp': 100 * (metrics(y, p, base)['macro_f1'] - base_metrics['macro_f1']),
                       'diagnostic_Recovery_at_NAR1': at1['R'] / wrong, 'at1': at1}
                controls.append(row)
                npz(ART / f'pred_{key}.npz', predictions=p, gain=gain)
            scores.extend(controls)
            for draw in C['support_seeds']:
                for n in C['nominal_shots']:
                    group = [r for r in fits if r['seed'] == seed and r['role'] == role and r['draw'] == draw and r['cap'] == n]
                    curves, rows = {}, []
                    for fit in group:
                        key, arm = fit['key'], fit['arm']
                        assert sha(CK / f'{key}.npz') == fit['checkpoint_sha256']
                        z = predict(read_npz(CK / f'{key}.npz'), x)
                        p = z.argmax(1); gain = margin_gain(z, original, p, base)
                        curve = frontier(y, p, base, gain); curves[arm] = curve
                        at1 = best_under_damage(curve, int(np.floor(correct * .01)))
                        row = {k: fit[k] for k in ['key', 'seed', 'role', 'draw', 'cap', 'arm', 'labels', 'strict_nshot']}
                        row.update(metrics(y, p, base))
                        row.update({'delta_f1_pp': 100 * (row['macro_f1'] - base_metrics['macro_f1']), 'diagnostic_Recovery_at_NAR1': at1['R'] / wrong, 'at1': at1})
                        rows.append(row)
                        npz(ART / f'pred_{key}.npz', predictions=p, gain=gain)
                    anchor = next(r for r in rows if r['arm'] == 'target_proto')['at1']
                    for row in rows:
                        curve = curves[row['arm']]
                        same_d = best_under_damage(curve, anchor['D'])
                        same_r = least_damage_for_recovery(curve, anchor['R'])
                        matched.append({k: row[k] for k in ['key', 'seed', 'role', 'draw', 'cap', 'arm']})
                        matched[-1].update({'anchor': anchor, 'same_damage': same_d,
                                           'same_damage_recovery_gain_pp': 100 * (same_d['R'] - anchor['R']) / wrong,
                                           'same_recovery': same_r, 'same_recovery_feasible_at_NAR1': bool(same_r is not None and same_r['D'] <= correct * .01)})
                    scores.extend(rows)
            npz(ART / f'evaluation_labels_{role}.npz', rows=query['rows'], labels=y)
            save('scores.json', scores); save('matched.json', matched)
            print(json.dumps({'phase': 'terminal_scoring', 'seed': seed, 'role': role, 'states': len(scores)}), flush=True)
    assert len(scores) == C['budget']['max_fit_cells'] + 12
    save('scores_complete.json', {'passed': True, 'states': len(scores), 'query_thresholds_diagnostic_only': True})


def summarize():
    scores = json.loads((ART / 'scores.json').read_text()); matched = json.loads((ART / 'matched.json').read_text())
    lines = ['# n-shot 冻结表征维护 v17', '', '全部预定拟合和终点评价完成。仅TemporalDrift已观察开发证据；主判决见下文，不能把query诊断阈值部署。', '',
             '严格1shot为每类1条、102标签。cap3/5取各类min(n,available)，实际预算如下，不是完整3/5shot。reference全池类别索引用于平衡基准采样，不声称同等主动采集成本。', '']
    preflight = json.loads((ART / 'preflight.json').read_text())
    for role, a in preflight['roles'].items():
        lines.append(f"- {role}: query {a['query']}，预算 " + ', '.join(f"cap{k}={v['labels']}标签（缺{v['deficit_vs_full']}）" for k, v in a['budgets'].items()))
    lines += ['', '表中为模型seed×support抽样均值；F1的SD是先按抽样平均后的三个modelseed样本SD，不是独立数据集置信区间。NAR与Recovery均为直接输出的实际比例；末列来自query标签选择的诊断前沿。', '',
              '|日期|预算|方法|F1均值±模型SD %|ΔF1 pp|Accuracy %|NAR %|Recovery %|诊断Rec@NAR1 %|',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for role in C['targets']:
        for n in [0] + C['nominal_shots']:
            for arm in (C['zero_label_controls'] if n == 0 else C['arms']):
                rows = [r for r in scores if r['role'] == role and r['cap'] == n and r['arm'] == arm]
                means = {k: float(np.mean([r[k] for r in rows])) for k in ['macro_f1', 'accuracy', 'delta_f1_pp', 'NAR', 'Recovery', 'diagnostic_Recovery_at_NAR1']}
                seedmeans = [np.mean([r['macro_f1'] for r in rows if r['seed'] == seed]) for seed in C['model_seeds']]
                label = 'zero' if n == 0 else ('strict1' if n == 1 else f'cap{n}')
                lines.append(f"|{role}|{label}|{arm}|{100*means['macro_f1']:.3f} ± {100*np.std(seedmeans,ddof=1):.3f}|{means['delta_f1_pp']:+.3f}|{100*means['accuracy']:.3f}|{100*means['NAR']:.3f}|{100*means['Recovery']:.3f}|{100*means['diagnostic_Recovery_at_NAR1']:.3f}|")
    gates = []
    for role in C['targets']:
        main = [r for r in scores if r['role'] == role and r['cap'] == 1 and r['arm'] == C['primary_arm']]
        control = [r for r in scores if r['role'] == role and r['cap'] == 1 and r['arm'] == 'target_proto']
        f1gain = 100 * (np.mean([r['macro_f1'] for r in main]) - np.mean([r['macro_f1'] for r in control]))
        same = [r for r in matched if r['role'] == role and r['cap'] == 1 and r['arm'] == C['primary_arm']]
        recgain = np.mean([r['same_damage_recovery_gain_pp'] for r in same])
        nonnegative = sum(np.mean([r['same_damage_recovery_gain_pp'] for r in same if r['seed'] == seed]) >= 0 for seed in C['model_seeds'])
        gates.append({'role': role, 'F1_gain_vs_target_proto_pp': float(f1gain), 'same_damage_recovery_gain_pp': float(recgain),
                      'nonnegative_model_seeds': int(nonnegative), 'passed': bool(f1gain >= .5 and recgain >= .5 and nonnegative >= 2)})
    decision = {'primary': C['primary_arm'], 'budget': 'strict_1shot', 'passed': all(g['passed'] for g in gates), 'roles': gates,
                'integrated': False, 'external_evaluation': False}
    save('decision.json', decision)
    lines += ['', '## 主比较：严格1shot收缩原型 vs 目标原型', '', '```json', json.dumps(decision,ensure_ascii=False,indent=2), '```', '',
              '所有174个状态、162个拟合、54个监督线性头及匹配前沿见artifacts。完整3/5-shot由于固定reference覆盖不足不可行；没有抽取query补齐。没有根据query选择超参数、checkpoint或部署阈值。仅优于frozen不证明收缩机制有效；次级方法较好也不替换主臂。', '',
              '有限样本与固定reference的类别索引是限制；support重复抽样和模型seed共享同一query，不能按174次独立重复解释。源标准化/原型仅2448个源样本，三方法享有相同源信息。主干仍是18,553纯源训练权重。', '',
              '审计通过后才登记completed。未启动监督LoRA、主动采样、其他切片或WTT/AWF。', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze','features','train','score','summarize']); args = parser.parse_args()
    torch.set_num_threads(C['budget']['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage != 'freeze':
        verify()
    globals()[args.stage]()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        registry('failed', f'阶段失败：{type(exc).__name__}；部分产物保留，未追加调参。')
        raise
