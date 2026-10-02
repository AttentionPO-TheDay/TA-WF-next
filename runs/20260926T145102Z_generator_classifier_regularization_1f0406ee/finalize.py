"""Aggregate all predefined comparisons, including failed and incomplete jobs."""
from __future__ import annotations
import subprocess
import statistics
from common import *


def close():
    config=config_read();names=[c['id'] for c in config['conditions']];reports=[];errors=[];checked=0
    config_sha=sha(RUN/'config.json');initials={}
    for seed in config['seeds']:
        for name in names:
            key=f'{name}_{seed}';path=RUN/'artifacts'/f'metrics_{key}.json'
            if path.exists():
                report=json.loads(path.read_text());reports.append(report)
                if report.get('config_sha256')!=config_sha:errors.append(key+': config mismatch')
                initials[key]=report.get('initial_state_sha256')
            verification=RUN/'artifacts'/f'verify_{key}.json'
            if not verification.exists():errors.append(key+': verification missing');continue
            result=json.loads(verification.read_text());checked+=result['prediction_sets_checked']
            if not result['complete']:errors.append(key+': '+str(result['errors']))
            for relative,digest in result['artifact_sha256'].items():
                if sha(ROOT/relative)!=digest:errors.append(key+': artifact changed after verification')
    for seed in config['seeds']:
        values=[initials.get(f'{name}_{seed}') for name in names]
        if None in values or len(set(values))!=1:errors.append(f'seed{seed}: missing or differing initial states')
    complete=len(reports)==12 and checked==24 and not errors
    integrity={'complete':complete,'prediction_sets_checked':checked,'expected_prediction_sets':24,'errors':errors,'device':'cpu','future_access':False}
    atomic_json(RUN/'artifacts/integrity.json',integrity)
    rows=['# 实验结果','',f"状态：{'completed' if complete else 'stopped / incomplete'}；CPU四条件×三seed，source2040/valid510，100轮上限。",'',
          '| 条件 | seed | 状态 | best epoch | train accuracy | valid accuracy | valid Macro-F1 | 秒 |',
          '|---|---:|---|---:|---:|---:|---:|---:|']
    for name in names:
        for seed in config['seeds']:
            r=next((x for x in reports if x['condition']==name and x['seed']==seed),None)
            if r is None or 'valid' not in r:rows.append(f'| {name} | {seed} | missing/no score | — | — | — | — | — |');continue
            rows.append(f"| {name} | {seed} | {r['status']} | {r['best_epoch']} | {100*r['source']['accuracy']:.3f}% | {100*r['valid']['accuracy']:.3f}% | {100*r['valid']['macro_f1']:.3f}% | {r['elapsed_seconds']:.1f} |")
    if complete:
        means={name:{m:statistics.mean(r['valid'][m] for r in reports if r['condition']==name) for m in ('accuracy','macro_f1')} for name in names}
        comparisons=[]
        rows+=['','| 条件 | mean valid accuracy | mean valid Macro-F1 |','|---|---:|---:|']
        for name,m in means.items():rows.append(f"| {name} | {100*m['accuracy']:.3f}% | {100*m['macro_f1']:.3f}% |")
        for pair in config['comparisons']:
            left=pair['left'];right=pair['right'];deltas=[]
            for seed in config['seeds']:
                a=next(r for r in reports if r['condition']==left and r['seed']==seed)
                b=next(r for r in reports if r['condition']==right and r['seed']==seed)
                deltas.append(100*(a['valid']['macro_f1']-b['valid']['macro_f1']))
            df=100*(means[left]['macro_f1']-means[right]['macro_f1']);da=100*(means[left]['accuracy']-means[right]['accuracy'])
            passed=df>=1 and all(d>0 for d in deltas) and da>=0
            comparison={**pair,'mean_f1_delta_pp':df,'mean_accuracy_delta_pp':da,'paired_f1_delta_pp':deltas,'gate_passed':passed}
            comparisons.append(comparison)
            rows+=['',f"{left} − {right}：F1 {df:+.3f}pp，accuracy {da:+.3f}pp；逐seed F1差 "+', '.join(f'{x:+.3f}' for x in deltas)+f"pp；预定门槛 {'PASS' if passed else 'FAIL'}。"]
        factorial=[]
        for seed in config['seeds']:
            v={r['condition']:100*r['valid']['macro_f1'] for r in reports if r['seed']==seed}
            factorial.append({'seed':seed,
                'generator_main_effect_pp':((v['R10_generator_anchor']-v['R00_baseline'])+(v['R11_both']-v['R01_classifier_reg']))/2,
                'classifier_main_effect_pp':((v['R01_classifier_reg']-v['R00_baseline'])+(v['R11_both']-v['R10_generator_anchor']))/2,
                'interaction_pp':v['R11_both']-v['R10_generator_anchor']-v['R01_classifier_reg']+v['R00_baseline']})
        diagnostics={}
        for name in names:
            rr=[r for r in reports if r['condition']==name]
            histories=[json.loads((RUN/'artifacts'/f"history_{name}_{r['seed']}.json").read_text())['history'] for r in rr]
            chosen=[next(h for h in hs if h['epoch']==r['best_epoch']) for hs,r in zip(histories,rr)]
            diagnostics[name]={
                'best_source_accuracy':statistics.mean(r['source']['accuracy'] for r in rr),
                'best_source_macro_f1':statistics.mean(r['source']['macro_f1'] for r in rr),
                'best_source_valid_accuracy_gap_pp':statistics.mean(100*(r['source']['accuracy']-r['valid']['accuracy']) for r in rr),
                'best_generator_parameter_delta_l2':statistics.mean(h['generator_parameter_delta_l2'] for h in chosen),
                'best_generator_token_delta_rms':statistics.mean(h['generator_token_delta_rms'] for h in chosen),
                'last_source_accuracy':statistics.mean(hs[-1]['source']['accuracy'] for hs in histories),
                'last_valid_macro_f1':statistics.mean(hs[-1]['valid']['macro_f1'] for hs in histories),
                'last_generator_parameter_delta_l2':statistics.mean(hs[-1]['generator_parameter_delta_l2'] for hs in histories)}
        atomic_json(RUN/'artifacts/summary.json',{'means':means,'comparisons':comparisons,'factorial_by_seed':factorial,'diagnostics':diagnostics})
        rows+=['','2×2效应（valid F1，描述性，单位pp）：']
        for field in ('generator_main_effect_pp','classifier_main_effect_pp','interaction_pp'):
            values=[v[field] for v in factorial]
            rows.append(f'{field}: {values}，均值 {statistics.mean(values):+.3f}。')
        rows+=['','| 条件 | best source accuracy | best source F1 | source−valid accuracy gap pp | best G偏移L2 | last G偏移L2 |', '|---|---:|---:|---:|---:|---:|']
        for n,d in diagnostics.items():
            rows.append(f"| {n} | {d['best_source_accuracy']*100:.3f}% | {d['best_source_macro_f1']*100:.3f}% | {d['best_source_valid_accuracy_gap_pp']:.3f} | {d['best_generator_parameter_delta_l2']:.3f} | {d['last_generator_parameter_delta_l2']:.3f} |")
        values='/'.join(f"{100*means[n]['macro_f1']:.3f}" for n in names)
        passed=[x['left']+'−'+x['right'] for x in comparisons if x['gate_passed']]
        summary=f'2×2正则化完成：R00/R01/R10/R11 valid F1 {values}%；通过比较{passed}；24/24预测与约束核验通过。'
        rows+=['','解释：分类器正则化是dropout和weight_decay的组合干预；锚定只限制生成器相对其随机初始化的偏移。训练拟合或差距下降本身不是成功，须看valid是否稳定上升；相关主效应不证明唯一过拟合来源。三seed共享已观察valid，结果仅为固定强度的开发筛查，未搜索超参或访问外部测试。']
        state='completed'
    else:
        summary=f'CPU正则化实验不完整：{checked}/24预测核验，{len(errors)}项缺失/错误；不作完整条件优劣裁决，不追加预算。'
        rows+=['',summary];state='stopped'
    rows+=['',f'独立核验：{checked}/24；errors={errors}。',
           '', '参数量、可训练参数、生成器反馈梯度/token变化见各seed metrics/history。代码与输入散列见artifacts/freeze.json、input_audit.json；检查、合成吞吐、训练及验证分开留存。每epoch保存完整恢复状态。',
           '', '仅使用TemporalDrift已观察source/valid；所有新模型从头初始化。无GPU、未来日期或外部数据访问。未追加机制或超参数搜索。']
    (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    p=ROOT/'STATUS.md';text=p.read_text();marker='当前正则化实验（2026-09-26）：`'+RUN.name+'`。'
    text='\n'.join(marker+summary if line.startswith(marker) else line for line in text.splitlines())+'\n';p.write_text(text)
    print(summary,flush=True)
    return 0 if complete else 1

if __name__=='__main__':raise SystemExit(close())
