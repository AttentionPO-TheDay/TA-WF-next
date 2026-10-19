"""Three empirical mechanism figures and the five-condition paper table."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())
CORE = ROOT / C['core_run']
LABELS = {'frozen': 'Frozen', 'tent_bn': 'TENT BN*', 'tent_entropy_gate': 'LN entropy gate',
          'proteus_default': 'Proteus port (1.0x)', 'no_pseudo_fixed': 'No-pseudo + lock (0.25x)', 'tent_coadapt': 'Full entropy coadapt'}


def registry(state, summary):
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status', '--id', RUN.name,
                    '--state', state, '--summary', summary], check=True)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export(fig, stem):
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(RUN / 'artifacts' / f'{stem}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    assert C['status'] == 'frozen'
    for relative, expected in json.loads((CORE / 'artifacts/freeze.json').read_text()).items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    audit = json.loads((CORE / 'artifacts/postrun_audit.json').read_text())
    assert audit['passed']
    registry('running', '核心矩阵审计通过，生成三张实测机理图及正文主表。')
    rows = json.loads((CORE / 'artifacts/scores.json').read_text())
    assert len(rows) == 54
    core_config = json.loads((CORE / 'config.json').read_text())
    seeds, roles, conditions = core_config['seeds'], core_config['roles'], core_config['conditions']
    table = []
    for role in roles:
        for condition in conditions:
            selected = [r for r in rows if r['role'] == role and r['condition'] == condition]
            assert len(selected) == 3
            row = {'role': role, 'condition': condition}
            for key, scale in [('macro_f1', 100), ('accuracy', 100), ('delta_f1_pp', 1),
                               ('negative_adaptation_rate', 100), ('recovery_rate', 100)]:
                values = np.array([r[key] for r in selected]) * scale
                row[key + '_mean'] = float(values.mean())
                row[key + '_sd'] = float(values.std(ddof=1))
            table.append(row)
    table_path = CORE / 'artifacts/official_benchmark_table.csv'
    with table_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ['# 论文一纯净源域时序核心矩阵', '',
             '三seed×三日期×五条件，共45个主表单元。9个frozen复用已审计的纯净基线，36个适应单元新执行；另9个tent_coadapt仅作机理对照，不混入主表。以下均为三seed均值±样本标准差，ΔF1单位pp，其余为%。', '',
             '| Day | Condition | Macro-F1 | Accuracy | ΔF1 | NAR | Recovery |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        values = [f"{r[key+'_mean']:.3f}±{r[key+'_sd']:.3f}" for key in ['macro_f1','accuracy','delta_f1_pp','negative_adaptation_rate','recovery_rate']]
        lines.append(f"| {r['role']} | {r['condition']} | " + ' | '.join(values) + ' |')
    lines += ['', '* TENT BN使用query批统计量（仅前向、无梯度），属于显式transductive-forward对照；其余方法在独立reference适应后固定状态评分。不得把该权限差异忽略。', '',
              '纯净源训练仅18,553行，source valid仅选模；本表所有目标日期均为已观察TemporalDrift开发数据，不是新确认集。主表no_pseudo_fixed是无伪标签、低学习率与分类头锁定的联合配方，不从此表单独归因某个组件。Proteus-default为wide_bn上的忠实机制移植，不称官方RF端到端复现。', '',
              '重载审计通过45个适应checkpoint和54条指标（含9个历史纯净frozen引用），支持集动力学probe不接触query，不改变BN状态或控制选模。机理图见配套run `'+RUN.name+'`；图表正向/阴性结果全部保留。']
    (CORE / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharex=True)
    norm_evidence = {}
    for axis, role in zip(axes, roles):
        norm_evidence[role] = {}
        for condition, color in [('tent_coadapt', '#bd3d3a'), ('tent_entropy_gate', '#226caa')]:
            ds = [json.loads((CORE / 'artifacts' / f'dynamics_{seed}_{role}_{condition}.json').read_text()) for seed in seeds]
            steps = [r['step'] for r in ds[0]]
            values = np.array([[r['logit_l2_mean'] for r in d] for d in ds])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            axis.plot(steps, mean, label=LABELS[condition], color=color)
            axis.fill_between(steps, mean - sd, mean + sd, alpha=.17, color=color)
            norm_evidence[role][condition] = {'step0_mean': float(mean[0]), 'step800_mean': float(mean[-1]),
                                            'max_over_steps_mean': float(mean.max()),
                                            'terminal_to_initial_ratio_by_seed': (values[:, -1] / values[:, 0]).tolist()}
        axis.set(title=role.replace('day', 'Day '), xlabel='Adaptation step', ylabel='Mean logit L2 on fixed reference')
        axis.set_xlim(0, 800)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    export(fig, 'fig1_logit_norm_dynamics')
    fig, axis = plt.subplots(figsize=(7.2, 4))
    bins = np.linspace(0, np.log(102), 51)
    entropy_evidence = {}
    for condition, color in [('frozen', '#777777'), ('tent_coadapt', '#bd3d3a'), ('tent_entropy_gate', '#226caa')]:
        values = []
        per_seed = []
        for seed in seeds:
            if condition == 'frozen':
                row = next(r for r in rows if r['seed'] == seed and r['role'] == 'day270' and r['condition'] == condition)
                path = ROOT / row['prediction_file']
            else:
                path = CORE / 'artifacts' / f'pred_{seed}_day270_{condition}.npz'
            with np.load(path) as data:
                z = data['logits'].astype(np.float64)
                z -= z.max(1, keepdims=True)
                logp = z - np.log(np.exp(z).sum(1, keepdims=True))
                h = -(np.exp(logp) * logp).sum(1)
            values.append(h)
            per_seed.append({'seed': seed, 'fraction_entropy_lt_0_2': float((h < .2).mean()),
                             'fraction_entropy_in_gate': float(((h >= .2) & (h <= .4)).mean()), 'mean_entropy': float(h.mean())})
        values = np.concatenate(values)
        axis.hist(values, bins=bins, weights=np.ones(len(values)) / len(values), histtype='step',
                  linewidth=1.8, label=LABELS[condition], color=color)
        entropy_evidence[condition] = per_seed
    axis.axvspan(.2, .4, color='#72ad72', alpha=.2, label='Update gate [0.2, 0.4] nats')
    axis.set(xlabel='Terminal Day 270 query entropy (nats)', ylabel='Fraction per bin', title='Observed entropy distribution; no assumed collapse rate')
    axis.legend(fontsize=8)
    fig.tight_layout()
    export(fig, 'fig2_day270_entropy_distribution')
    fig, axis = plt.subplots(figsize=(7, 5))
    colors = plt.get_cmap('tab10')
    quadrant = {}
    for index, condition in enumerate(conditions + ['tent_coadapt']):
        selected = [r for r in rows if r['condition'] == condition]
        x = np.array([r['negative_adaptation_rate'] for r in selected]) * 100
        y = np.array([r['recovery_rate'] for r in selected]) * 100
        axis.scatter(x, y, alpha=.2, s=24, color=colors(index))
        axis.scatter(x.mean(), y.mean(), color=colors(index), s=75, label=LABELS[condition], edgecolors='black', linewidths=.5)
        quadrant[condition] = {'nar_mean_pct': float(x.mean()), 'recovery_mean_pct': float(y.mean()),
                               'cells': len(selected), 'aggregation': 'equal seed/date cells'}
    axis.set(xlabel='Negative adaptation rate (%) — lower is better', ylabel='Recovery rate (%) — higher is better',
             title='Observed safety–recovery tradeoff; each faint point is one cell')
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    fig.tight_layout()
    export(fig, 'fig3_nar_recovery')
    write_json(RUN / 'artifacts/figure_data.json', {'norms': norm_evidence, 'day270_entropy': entropy_evidence,
               'quadrant': quadrant, 'low_entropy_threshold_nats': .2, 'pre_specified_false_confidence_fraction': None})
    result = ['# 机理图生成结果', '', '三张图均导出PDF/SVG/PNG，数值见artifacts/figure_data.json。正文五条件主表见核心矩阵artifacts/official_benchmark_table.csv。', '',
              '图1：仅固定1024条无标签reference的step0–800评估，阴影为三seed样本标准差；不因probe结果改变训练。图2：Day270终端query熵，0.2 nat为预设低熵阈值；不能从低熵本身推断错误，也不预设98%比例。图3：各seed/date单元的NAR与Recovery及等权均值；不预设校准方法达到最低NAR。', '']
    for condition, values in entropy_evidence.items():
        result.append(f"- {condition}：Day270 H<0.2占比均值 {np.mean([x['fraction_entropy_lt_0_2'] for x in values])*100:.3f}%。")
    result += ['', '解读边界：支持集probe与query终端指标角色分离。参数与统计量锁定只验证实现范围，不保证logit范数有数学上界、不自动证明物理因果或零误杀。该结果是已观察TemporalDrift上的开发机制诊断。']
    (RUN / 'RESULTS.md').write_text('\n'.join(result) + '\n')
    artifacts = list((RUN / 'artifacts').glob('fig*.*'))
    assert len([p for p in artifacts if p.suffix in ('.pdf', '.svg', '.png')]) == 9
    write_json(RUN / 'artifacts/render_audit.json', {'passed': True, 'figures': 3, 'formats': ['pdf', 'svg', 'png'],
              'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}})
    registry('completed', '三张实测机理图已导出PDF/SVG/PNG及数值；正文主表同步生成，不预设发散/98%低熵。')


if __name__ == '__main__':
    main()
