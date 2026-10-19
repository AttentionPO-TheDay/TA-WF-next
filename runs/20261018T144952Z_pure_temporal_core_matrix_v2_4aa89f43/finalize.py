"""Merge audited historical frozen cells, check every new cell, and publish results."""
import json
from collections import defaultdict

from runtime import RUN, ROOT, C, save, sha, setup, verify_freeze, registry


def main():
    setup()
    verify_freeze()
    baseline = ROOT / 'runs' / C['frozen_baseline_run']
    frozen = json.loads((baseline / 'artifacts/scores.json').read_text())
    assert len(frozen) == 9
    for row in frozen:
        row['prediction_file'] = str((baseline / 'artifacts' /
            f"pred_{row['seed']}_{row['role']}_frozen.npz").relative_to(ROOT))
        row['prediction_sha256'] = sha(ROOT / row['prediction_file'])
        row['delta_f1_pp'] = 0.0
        row['negative_adaptation_rate'] = 0.0
        row['recovery_rate'] = 0.0
        row['origin'] = 'audited_historical_pure_frozen'
    adapted = []
    for seed in C['seeds']:
        rows = json.loads((RUN / 'artifacts' / f'scores_seed_{seed}.json').read_text())
        assert len(rows) == 12
        adapted.extend(rows)
    scores = frozen + adapted
    assert len(scores) == 45
    assert len({(r['seed'], r['role'], r['condition']) for r in scores}) == 45
    save('scores.json', scores)
    from audit import main as audit
    audit()
    by_condition = defaultdict(list)
    for row in scores:
        by_condition[row['condition']].append(row)
    lines = ['# 实验结果', '', '纯净源域三 seed × Day14/90/270 × 五条件共 45 单元已完成；其中 9 个 frozen 单元引用独立审计通过的历史 run，本 run 新增 36 个适应单元。`postrun_audit.json` 已逐项重算并重载全部新 checkpoint。TemporalDrift 是已观察开发集。', '', '| 条件 | Macro-F1 (%) | Accuracy (%) | NAR (%) | Recovery (%) |', '|---|---:|---:|---:|---:|']
    for condition in C['conditions']:
        rows = by_condition[condition]
        assert len(rows) == 9
        mean = lambda key: 100 * sum(r[key] for r in rows) / len(rows)
        lines.append(f"| {condition} | {mean('macro_f1'):.3f} | {mean('accuracy'):.3f} | {mean('negative_adaptation_rate'):.3f} | {mean('recovery_rate'):.3f} |")
    lines += ['', '逐 seed、日期、条件结果在 `artifacts/scores.json`；800 步支持集动力学、预测和成本留在本 run 的 artifacts/checkpoints。TENT BN 的 query 前向使用批统计量，信息权限与另外四条件不同，单独解释。外部数据未用于此矩阵。', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))
    registry('completed', '纯净源域 TemporalDrift 五条件 45 单元完成（9个冻结引用+36个新适应）；独立重载审计通过，详见 RESULTS.md。')


if __name__ == '__main__':
    main()
