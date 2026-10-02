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
        reference=config['practical_reference'];baseline_rows=[]
        prior_root=ROOT/'runs'/reference['run']
        for name in names:
            deltas=[]
            for seed in config['seeds']:
                previous=json.loads((prior_root/'artifacts'/f'metrics_summary_{seed}.json').read_text())
                actual=next(r for r in reports if r['condition']==name and r['seed']==seed)
                deltas.append(100*(actual['valid']['macro_f1']-previous['valid']['macro_f1']))
            baseline_rows.append({'condition':name,'historical_summary_f1_delta_pp':100*(means[name]['macro_f1']-reference['summary_valid_macro_f1']),
                                  'historical_summary_accuracy_delta_pp':100*(means[name]['accuracy']-reference['summary_valid_accuracy']),
                                  'paired_f1_delta_pp':deltas})
        atomic_json(RUN/'artifacts/summary.json',{'means':means,'comparisons':comparisons,'historical_summary_reference':baseline_rows})
        fvalues='/'.join(f"{100*means[n]['macro_f1']:.3f}" for n in names)
        summary=f'CPU生成器四条件完成：A/B/C/D valid F1 {fvalues}%；24/24预测与反馈边界核验通过。'
        rows+=['','历史摘要参照为accuracy26.471% / F1 24.655%，明确复用，不是新的独立重复。']
        for b in baseline_rows:rows.append(f"{b['condition']} 相对历史摘要：F1 {b['historical_summary_f1_delta_pp']:+.3f}pp，accuracy {b['historical_summary_accuracy_delta_pp']:+.3f}pp；三seed F1差 {b['paired_f1_delta_pp']}。")
        rows+=['','解释：B−C仅检验分类反馈相对随机冻结生成器的作用，胜过C不能证明实际收益；B−A检验可学习汇聚；D−B检验较小生成器更新尺度。结果仅为预定开发筛查，不是统计显著性或独立确认。采用候选仍需结合历史摘要的实际性能及全部seed一致性，不能只挑有利比较。']
        state='completed'
    else:
        summary=f'CPU生成器实验不完整：{checked}/24预测核验，{len(errors)}项缺失/错误；不作完整条件优劣裁决，不追加预算。'
        rows+=['',summary];state='stopped'
    rows+=['',f'独立核验：{checked}/24；errors={errors}。',
           '', '参数量、可训练参数、生成器反馈梯度/token变化见各seed metrics/history。代码与输入散列见artifacts/freeze.json、input_audit.json；检查、合成吞吐、训练及验证分开留存。每epoch保存完整恢复状态。',
           '', '仅使用TemporalDrift已观察source/valid；所有新模型从头初始化。无GPU、未来日期或外部数据访问。未追加机制或超参数搜索。']
    (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    p=ROOT/'STATUS.md';text=p.read_text();marker='当前并行实验设计（2026-09-26）：`'+RUN.name+'`。'
    text='\n'.join(marker+summary if line.startswith(marker) else line for line in text.splitlines())+'\n';p.write_text(text)
    print(summary,flush=True)
    return 0 if complete else 1

if __name__=='__main__':raise SystemExit(close())
