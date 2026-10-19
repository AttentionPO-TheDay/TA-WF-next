"""Independent checkpoint reload and sklearn metric recomputation; no selection."""
import json
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, accuracy_score
from data import load_cached
from runner import RUN, C, sha, save, clone_source, predict, registry
from token_alignment import AdaINToken, ResidualTokenProjector


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for relative, expected in json.loads((RUN / 'artifacts/freeze.json').read_text()).items():
        assert sha(RUN.parents[1] / relative) == expected, relative
    rows = json.loads((RUN / 'artifacts/scores.json').read_text())
    assert len(rows) == C['budget']['total_cells']
    keys = {(r['dataset'], r['seed'], r['slice'], r['condition']) for r in rows}
    assert len(keys) == len(rows)
    checked = 0
    for name in ('wtt_time', 'awf'):
        bundle = load_cached(name)
        classes = bundle['classes']
        expected = {(name, seed, date, condition) for seed in C['seeds'] for date in bundle['targets'] for condition in C['conditions']}
        assert expected == {key for key in keys if key[0] == name}
        for seed in C['seeds']:
            source = clone_source(name, classes, seed)
            for date, target in bundle['targets'].items():
                baseline = predict(source, target['query']).argmax(1)
                y = target['query']['y']
                base_f1 = f1_score(y, baseline, labels=list(range(classes)), average='macro', zero_division=0)
                for condition in C['conditions']:
                    row = next(r for r in rows if (r['dataset'], r['seed'], r['slice'], r['condition']) == (name, seed, date, condition))
                    stem = f'{name}_{seed}_{date}_{condition}'
                    path = RUN / 'artifacts' / f'pred_{stem}.npz'
                    assert sha(path) == row['prediction_sha256']
                    with np.load(path) as pred:
                        saved_logits = pred['logits']
                        assert np.array_equal(pred['labels'], y) and np.array_equal(pred['uids'], target['query']['uids'])
                        assert np.array_equal(pred['predictions'], saved_logits.argmax(1))
                    model = clone_source(name, classes, seed)
                    aligner = nn.Identity()
                    if condition != 'frozen':
                        checkpoint = torch.load(RUN / 'checkpoints' / f'{stem}.pt', map_location='cpu', weights_only=True)
                        assert checkpoint['condition'] == condition and checkpoint['config_sha256'] == sha(RUN / 'config.json')
                        model.load_state_dict(checkpoint['model'], strict=True)
                        if condition in ('token_adain', 'token_projector'):
                            for key, value in model.state_dict().items():
                                assert torch.equal(value, source.state_dict()[key]), key
                        if condition == 'token_adain':
                            dummy = {'mean': torch.zeros(110), 'variance': torch.ones(110)}
                            aligner = AdaINToken(dummy, dummy).cuda()
                        elif condition == 'token_projector':
                            aligner = ResidualTokenProjector(**C['token_alignment']['projector']).cuda()
                        aligner.load_state_dict(checkpoint['aligner'], strict=True)
                        for key, value in model.named_buffers():
                            assert torch.equal(value, dict(source.named_buffers())[key]), key
                    logits = predict(model, target['query'], aligner)
                    assert np.allclose(logits, saved_logits, atol=1e-6, rtol=1e-5)
                    p = logits.argmax(1)
                    f1 = f1_score(y, p, labels=list(range(classes)), average='macro', zero_division=0)
                    assert abs(f1 - row['macro_f1']) < 1e-12
                    assert abs(accuracy_score(y, p) - row['accuracy']) < 1e-12
                    assert abs(100 * (f1 - base_f1) - row['delta_f1_pp']) < 1e-10
                    correct = baseline == y
                    nar = float(np.sum(correct & (p != y)) / np.sum(correct)) if correct.any() else None
                    assert nar == row['negative_adaptation_rate']
                    checked += 1
                    del model, aligner
            del source
        del bundle
    summary = defaultdict(list)
    for row in rows:
        summary[(row['dataset'], row['slice'], row['condition'])].append(row)
    lines = ['# 外部独立方向模型方法迁移 v4', '', '144 个单元完成；全部预测文件、checkpoint/对齐层重载及 sklearn 指标复算通过。此结果检验冻结方法迁移，不是 Temporal 唯一模型的零样本跨类别分类。AWF 是固定文件诊断评价；WTT 不在未经完整暴露审计时宣称严格未见确认。', '',
             '| Dataset | Slice | Condition | F1 mean±sd (%) | Accuracy (%) | ΔF1 (pp) | NAR (%) |',
             '|---|---|---|---:|---:|---:|---:|']
    for (name, date, condition), group in sorted(summary.items()):
        assert len(group) == len(C['seeds'])
        f = np.array([r['macro_f1'] for r in group]) * 100
        acc = np.mean([r['accuracy'] for r in group]) * 100
        delta = np.mean([r['delta_f1_pp'] for r in group])
        nars = [r['negative_adaptation_rate'] for r in group if r['negative_adaptation_rate'] is not None]
        nar_text = f'{np.mean(nars)*100:.3f}' if nars else 'NA'
        lines.append(f'| {name} | {date} | {condition} | {f.mean():.3f}±{f.std(ddof=1):.3f} | {acc:.3f} | {delta:+.3f} | {nar_text} |')
    (RUN / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    save('postrun_audit.json', {'passed': True, 'cells_reloaded': checked, 'metrics_recomputed': checked,
                              'strict_confirmation_claimed': False, 'query_used_for_selection': False})
    registry('completed', '外部144单元完成，checkpoint/对齐层重载与独立指标复算通过；仅方法迁移评价。')


if __name__ == '__main__':
    main()
