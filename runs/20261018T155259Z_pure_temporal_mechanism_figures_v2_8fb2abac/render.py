"""Read-only rendering of the audited pure-source five-condition matrix."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())
CORE = ROOT / C['core_run']
NAMES = {'frozen': 'Frozen', 'tent_bn': 'TENT BN*', 'tent_entropy_gate': 'LN entropy gate',
         'proteus_default': 'Proteus port', 'no_pseudo_fixed': 'No-pseudo + lock'}
COLORS = {'frozen': '#777777', 'tent_bn': '#8762a8', 'tent_entropy_gate': '#2277aa',
          'proteus_default': '#bf5b43', 'no_pseudo_fixed': '#278451'}
READ_HASHES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path, expected=None):
    digest = sha(path)
    if expected is not None:
        assert digest == expected, str(path)
    READ_HASHES[str(path.relative_to(ROOT))] = digest
    return path


def registry(state, summary):
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status',
                    '--id', RUN.name, '--state', state, '--summary', summary], check=True)


def export(fig, stem):
    for suffix in C['formats']:
        fig.savefig(RUN / 'artifacts' / f'{stem}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def stats(rows, key, scale=100):
    values = np.array([r[key] for r in rows], dtype=np.float64) * scale
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def main():
    started = time.monotonic()
    assert C['status'] == 'frozen' and C['id'] == RUN.name
    source_freeze = json.loads((RUN / 'artifacts/source_freeze.json').read_text())
    for name, digest in source_freeze.items():
        assert sha(RUN / name) == digest, name
    freeze = json.loads(read(CORE / 'artifacts/freeze.json').read_text())
    for relative, digest in freeze.items():
        assert sha(ROOT / relative) == digest, relative
    audit = json.loads(read(CORE / 'artifacts/postrun_audit.json').read_text())
    assert audit['passed'] and audit['total_metrics_recomputed'] == 45 and audit['adapted_checkpoints_reloaded'] == 36
    core_config = json.loads(read(CORE / 'config.json').read_text())
    assert all(C[k] == core_config[k] for k in ('seeds', 'roles', 'conditions'))
    rows = json.loads(read(CORE / 'artifacts/scores.json').read_text())
    expected = {(s, d, c) for s in C['seeds'] for d in C['roles'] for c in C['conditions']}
    assert len(rows) == len(expected) == 45
    assert {(r['seed'], r['role'], r['condition']) for r in rows} == expected
    lookup = {(r['seed'], r['role'], r['condition']): r for r in rows}
    registry('running', '读取已审计的纯净五条件矩阵，生成主表及三张实测机理图。')

    # Paired three-seed day rows, followed by nine-cell equal-weight means.
    table = []
    for role in C['roles'] + ['all_days_equal_cells']:
        for condition in C['conditions']:
            selected = [r for r in rows if r['condition'] == condition and
                        (role == 'all_days_equal_cells' or r['role'] == role)]
            assert len(selected) == (9 if role == 'all_days_equal_cells' else 3)
            item = {'role': role, 'condition': condition, 'cells': len(selected)}
            for key, scale in [('macro_f1', 100), ('accuracy', 100), ('delta_f1_pp', 1),
                               ('negative_adaptation_rate', 100), ('recovery_rate', 100)]:
                item[key + '_mean'], item[key + '_sd'] = stats(selected, key, scale)
            table.append(item)
    with (RUN / 'artifacts/benchmark_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ['# 纯净源域五条件时序矩阵主表', '',
             '三 seed × 三日期，45 个单元；frozen 的 9 个单元引用已审计历史基线，36 个适应单元来自本次矩阵。数值为等权均值±样本 SD；ΔF1 单位百分点，其余为%。', '',
             '| 日期 | 条件 | Macro-F1 | Accuracy | ΔF1 | NAR | Recovery |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        vals = [f"{r[k+'_mean']:.3f}±{r[k+'_sd']:.3f}" for k in
                ('macro_f1', 'accuracy', 'delta_f1_pp', 'negative_adaptation_rate', 'recovery_rate')]
        lines.append(f"| {r['role']} | {r['condition']} | " + ' | '.join(vals) + ' |')
    lines += ['', 'TENT BN 的 query 前向使用批统计量，属于单独声明的 transductive-forward 对照。其余方法在独立 reference 上适应后固定模型评分。所有目标日期属于已观察的 TemporalDrift 开发数据；no_pseudo_fixed 是去伪标签、低学习率和分类头锁定的联合配方，主表不能单独归因某个组件。']
    (RUN / 'artifacts/benchmark_table.md').write_text('\n'.join(lines) + '\n')

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
    norms = {}
    for axis, role in zip(axes, C['roles']):
        norms[role] = {}
        for condition in C['conditions'][1:]:
            histories = []
            for seed in C['seeds']:
                path = CORE / 'artifacts' / f'dynamics_{seed}_{role}_{condition}.json'
                d = json.loads(read(path).read_text())
                assert [x['step'] for x in d] == list(range(0, 801, 10))
                assert all(x['scope'] == 'fixed_1024_unlabeled_reference_eval_mode' for x in d)
                histories.append([x['logit_l2_mean'] for x in d])
            values = np.array(histories)
            assert np.isfinite(values).all()
            mean, sd = values.mean(0), values.std(0, ddof=1)
            steps = np.arange(0, 801, 10)
            axis.plot(steps, mean, color=COLORS[condition], label=NAMES[condition], linewidth=1.5)
            axis.fill_between(steps, mean - sd, mean + sd, color=COLORS[condition], alpha=.12)
            norms[role][condition] = {'step0_mean': float(mean[0]), 'step800_mean': float(mean[-1]),
                                      'step800_sd': float(sd[-1]), 'max_mean': float(mean.max()),
                                      'per_seed_step800_to_step0_ratio': (values[:, -1] / values[:, 0]).tolist()}
        frozen0 = np.mean([json.loads(read(CORE / 'artifacts' /
            f'dynamics_{seed}_{role}_tent_bn.json').read_text())[0]['logit_l2_mean'] for seed in C['seeds']])
        axis.axhline(frozen0, color=COLORS['frozen'], linestyle='--', linewidth=1, label='Frozen initial')
        axis.set(title=role.replace('day', 'Day '), xlabel='Adaptation step', ylabel='Mean logit L2 on reference', xlim=(0,800))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7, loc='best')
    fig.tight_layout()
    export(fig, C['figures'][0])

    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(0, np.log(102), 51)
    entropy_report = {}
    for condition in C['conditions']:
        histograms, per_seed = [], []
        for seed in C['seeds']:
            row = lookup[(seed, 'day270', condition)]
            path = ROOT / row['prediction_file'] if condition == 'frozen' else CORE / 'artifacts' / f'pred_{seed}_day270_{condition}.npz'
            with np.load(read(path, row['prediction_sha256'])) as z:
                logits = z['logits'].astype(np.float64)
            assert logits.ndim == 2 and logits.shape[1] == 102 and len(logits) == row['rows']
            logits -= logits.max(1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
            h = -(np.exp(logp) * logp).sum(1)
            assert np.isfinite(h).all() and (h >= 0).all()
            histograms.append(np.histogram(h, bins=bins)[0] / len(h))
            per_seed.append({'seed': seed, 'rows': len(h), 'mean_entropy_nats': float(h.mean()),
                             'fraction_below_0_2': float((h < .2).mean()),
                             'fraction_in_0_2_to_0_4': float(((h >= .2) & (h <= .4)).mean())})
        axis.stairs(np.mean(histograms, axis=0), bins, label=NAMES[condition], color=COLORS[condition], linewidth=1.6)
        entropy_report[condition] = per_seed
    axis.axvspan(.2, .4, color='#71b579', alpha=.15, label='Predefined gate [0.2, 0.4]')
    axis.set(xlabel='Day 270 terminal query entropy (nats)', ylabel='Fraction per bin (equal seed weight)',
             title='Empirical entropy distribution')
    axis.legend(fontsize=8)
    fig.tight_layout()
    export(fig, C['figures'][1])

    fig, axis = plt.subplots(figsize=(7, 5))
    tradeoff = {}
    for condition in C['conditions']:
        selected = [lookup[(seed, role, condition)] for role in C['roles'] for seed in C['seeds']]
        x = np.array([r['negative_adaptation_rate'] for r in selected]) * 100
        y = np.array([r['recovery_rate'] for r in selected]) * 100
        axis.scatter(x, y, color=COLORS[condition], alpha=.25, s=22)
        axis.scatter([x.mean()], [y.mean()], color=COLORS[condition], edgecolors='black',
                     linewidths=.5, s=85, label=NAMES[condition])
        tradeoff[condition] = {'nar_mean_pct': float(x.mean()), 'recovery_mean_pct': float(y.mean()),
                               'cells': 9, 'aggregation': 'equal_seed_date_cells'}
    axis.set(xlabel='Negative adaptation rate (%)', ylabel='Recovery rate (%)',
             title='Safety–recovery tradeoff (faint points: individual cells)', xlim=(0,None), ylim=(0,None))
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    export(fig, C['figures'][2])

    (RUN / 'artifacts/figure_data.json').write_text(json.dumps({'norms': norms,
        'day270_entropy': entropy_report, 'nar_recovery': tradeoff,
        'entropy_gate_nats': [0.2, 0.4]}, indent=2, allow_nan=False) + '\n')
    outputs = [p for p in (RUN / 'artifacts').iterdir() if p.is_file() and p.name != 'render_audit.json']
    assert len([p for p in outputs if p.suffix in ('.pdf','.svg','.png')]) == 9
    size = sum(p.stat().st_size for p in outputs)
    assert size <= C['budget']['max_output_gib'] * 1024**3
    elapsed = time.monotonic() - started
    assert elapsed <= C['budget']['max_seconds']
    (RUN / 'artifacts/render_audit.json').write_text(json.dumps({'passed': True,
        'source_freeze': source_freeze,
        'core_audit_sha256': READ_HASHES[str((CORE / 'artifacts/postrun_audit.json').relative_to(ROOT))],
        'core_scores_sha256': READ_HASHES[str((CORE / 'artifacts/scores.json').relative_to(ROOT))],
        'input_sha256': READ_HASHES, 'output_sha256': {p.name: sha(p) for p in outputs},
        'figure_count': 3, 'formats_per_figure': 3, 'table_rows': len(table),
        'elapsed_seconds': elapsed, 'output_bytes': size}, indent=2) + '\n')
    result = ['# 五条件机理图结果', '', '基于已审计的45单元纯净时序矩阵；三图各导出 PDF/SVG/PNG，五条件主表在 `artifacts/benchmark_table.md` 和 CSV。输出及输入哈希均通过 render_audit。', '',
              '图1使用固定无标签 reference 的 0–800 步 probe，阴影为三 seed 样本标准差；图2使用 Day270 终端独立 query logits，三个 seed 等权；图3使用全部九个 seed×日期单元的 NAR 与 Recovery。', '']
    for condition in C['conditions']:
        rate = np.mean([x['fraction_below_0_2'] for x in entropy_report[condition]]) * 100
        result.append(f'- {condition}：Day270 终端 query 熵 <0.2 nat 比例 {rate:.3f}%。')
    result += ['', '这些图只说明已观察 TemporalDrift 开发数据上的关联与诊断，不证明物理因果。TENT BN 的 query 前向使用批统计量，应单独解释；旧含 tent_coadapt 的图表属于另一套矩阵。']
    (RUN / 'RESULTS.md').write_text('\n'.join(result) + '\n')
    registry('completed', '五条件纯净时序矩阵主表与三张实测图完成；输入/输出散列及45单元来源核验通过。')


if __name__ == '__main__':
    main()
