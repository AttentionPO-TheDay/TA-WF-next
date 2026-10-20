"""Reload all sensitivity checkpoints and independently recompute full-query metrics."""
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from runtime import RUN, ROOT, C, setup, verify_freeze, load_model, load_target, predict, configure, sha, save, registry


def main():
    setup()
    verify_freeze()
    base = ROOT / 'runs' / C['base_matrix_run']
    assert json.loads((base / 'artifacts/postrun_audit.json').read_text())['passed']
    control_rows = json.loads((base / 'artifacts/scores.json').read_text())
    controls = {(r['seed'],r['role'],r['condition']):r for r in control_rows
                if r['condition'] in C['adaptation_conditions']}
    assert len(controls)==18
    fresh = []
    for seed in C['seeds']:
        rows = json.loads((RUN/'artifacts'/f'scores_seed_{seed}.json').read_text())
        assert len(rows)==21
        fresh.extend(rows)
    expected = {(s,d,v) for s in C['seeds'] for d in C['roles'] for v in C['variants']}
    assert len(fresh)==len(expected)==63
    assert {(r['seed'],r['role'],r['variant']) for r in fresh}==expected
    lookup = {(r['seed'],r['role'],r['variant']):r for r in fresh}
    reload_count=0
    for role in C['roles']:
        _, query=load_target(role)
        y=query['labels'].numpy()
        for seed in C['seeds']:
            pure=load_model(seed)
            original={k:v.detach().cpu().clone() for k,v in pure.state_dict().items()}
            frozen_path=ROOT/'runs'/C['frozen_baseline_run']/'artifacts'/f'pred_{seed}_{role}_frozen.npz'
            with np.load(frozen_path) as z:
                assert np.array_equal(z['rows'],query['rows'].numpy()) and np.array_equal(z['labels'],y)
                frozen=z['predictions'].copy()
            for variant,settings in C['variants'].items():
                row=lookup[(seed,role,variant)]
                condition=settings['condition']
                control=controls[(seed,role,condition)]
                assert row['condition']==condition and row['candidate_updates']==settings['candidate_updates']
                assert row['learning_rate']==settings['learning_rate'] and row['gate_nats']==settings.get('gate_nats')
                assert sha(base/'artifacts'/f'pred_{seed}_{role}_{condition}.npz')==control['prediction_sha256']
                key=f'{seed}_{role}_{variant}'
                pred_path=RUN/'artifacts'/f'pred_{key}.npz'
                ck_path=RUN/'checkpoints'/f'{key}.pt'
                assert sha(pred_path)==row['prediction_sha256'] and sha(ck_path)==row['checkpoint_sha256']
                ck=torch.load(ck_path,map_location='cpu',weights_only=True)
                assert ck['seed']==seed and ck['role']==role and ck['variant']==variant
                assert ck['config_sha256']==sha(RUN/'config.json')
                model=load_model(seed)
                names=set(configure(model,condition))
                model.load_state_dict(ck['state_dict'],strict=True)
                for name,value in model.state_dict().items():
                    if condition=='tent_entropy_gate' and name not in names:
                        assert torch.equal(value.cpu(),original[name]),name
                    if condition=='no_pseudo_fixed' and name.startswith('classifier.'):
                        assert torch.equal(value.cpu(),original[name]),name
                logits=predict(model,query)
                with np.load(pred_path) as z:
                    assert np.array_equal(z['rows'],query['rows'].numpy()) and np.array_equal(z['labels'],y)
                    assert np.allclose(logits,z['logits'],atol=1e-6,rtol=1e-5)
                    prediction=z['predictions'].copy()
                    assert np.array_equal(prediction,logits.argmax(1))
                f1=f1_score(y,prediction,labels=list(range(102)),average='macro',zero_division=0)
                acc=accuracy_score(y,prediction)
                assert abs(f1-row['macro_f1'])<1e-12 and abs(acc-row['accuracy'])<1e-12
                assert abs(100*(f1-control['macro_f1'])-row['delta_f1_vs_control_pp'])<1e-10
                correct=frozen==y
                nar=float((correct & (prediction!=y)).sum()/correct.sum())
                recovery=float(((~correct)&(prediction==y)).sum()/(~correct).sum())
                assert nar==row['negative_adaptation_rate'] and recovery==row['recovery_rate']
                history=json.loads((RUN/'artifacts'/f'updates_{key}.json').read_text())
                dynamics=json.loads((RUN/'artifacts'/f'dynamics_{key}.json').read_text())
                assert len(history)==settings['candidate_updates']
                assert [x['step'] for x in dynamics]==list(range(0,settings['candidate_updates']+1,10))
                if settings['candidate_updates']==800:
                    assert row['streams']['target']==control['streams']['target']
                    if condition=='no_pseudo_fixed':
                        assert row['streams']['source']==control['streams']['source']
                reload_count+=1
                del model
            del pure
    assert reload_count==63
    save('audit.json',{'passed':True,'new_checkpoints_reloaded':63,
                       'total_metric_cells':81,'reused_control_cells':18,
                       'query_labels_used_for_selection_or_adaptation':False})
    all_rows=[]
    for control in controls.values():
        all_rows.append({'seed':control['seed'],'role':control['role'],
                         'variant':'control_'+control['condition'],
                         'condition':control['condition'],
                         'macro_f1':control['macro_f1'],'accuracy':control['accuracy'],
                         'negative_adaptation_rate':control['negative_adaptation_rate'],
                         'recovery_rate':control['recovery_rate'],
                         'origin':'audited_historical_control'})
    all_rows += fresh
    save('scores.json',all_rows)
    by_variant=defaultdict(list)
    for row in all_rows:by_variant[row['variant']].append(row)
    lines=['# 轻量超参数敏感性结果','',
           '63个新适应单元及18个历史控制单元共81条指标已全query重算；63个新checkpoint独立重载。所有结果是已观察TemporalDrift开发证据。', '',
           '| 配方 | 配置 | Macro-F1 (%) | 相对本配方控制 ΔF1 (pp) | NAR (%) | Recovery (%) | 配对正向/9 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for condition in C['adaptation_conditions']:
        control_key='control_'+condition
        ordering=[control_key]+[v for v,x in C['variants'].items() if x['condition']==condition]
        for variant in ordering:
            rows=by_variant[variant]
            assert len(rows)==9
            mean=lambda key:sum(r[key] for r in rows)/9
            delta=0 if variant==control_key else mean('delta_f1_vs_control_pp')
            positive='—' if variant==control_key else str(sum(r['delta_f1_vs_control_pp']>0 for r in rows))
            setting='原800步/lr0.00025'+('/门控[0.2,0.4]' if condition=='tent_entropy_gate' else '') if variant==control_key else str(C['variants'][variant])
            lines.append(f"| {condition} | {variant}: {setting} | {100*mean('macro_f1'):.3f} | {delta:+.3f} | {100*mean('negative_adaptation_rate'):.3f} | {100*mean('recovery_rate'):.3f} | {positive} |")
    lines += ['', '逐 seed×日期预测、配对指标、成本与动态日志在 artifacts。两组配方分别对照，不跨配方把一个超参数变化归因为方法优劣。TENT门控样本数会随区间变化，属于本敏感性变量。三日期和三训练seed共享既有开发query，不作为独立确认或显著性样本。外部WTT/AWF未重新评分。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    registry('completed','双协议超参数敏感性完成：63新适应+18历史控制，63checkpoint全query重载、81指标复算通过；详见RESULTS.md。')


if __name__=='__main__':main()
