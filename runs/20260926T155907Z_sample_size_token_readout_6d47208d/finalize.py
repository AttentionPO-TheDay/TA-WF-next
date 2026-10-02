"""Summarize every prespecified paired contrast without extra selection."""
import subprocess,statistics
from common import *

def main():
    c=config_read();reports={};errors=[];checked=0;names=[v['id'] for v in c['conditions']]
    for seed in c['seeds']:
        for name in names:
            key=f'{name}_{seed}';out=RUN/'artifacts'/key
            try:
                r=json.loads((out/'report.json').read_text());v=json.loads((out/'verification.json').read_text())
                assert r['status']=='completed' and v['complete'] and r['config_sha256']==sha(RUN/'config.json')
                for p,h in v['artifact_sha256'].items():assert sha(ROOT/p)==h,p
                reports[key]=r;checked+=v['prediction_sets_checked']
            except Exception as e:errors.append(f'{key}: {type(e).__name__} {e}')
    for seed in c['seeds']:
        rr=[reports.get(f'{n}_{seed}') for n in names]
        if any(r is None for r in rr) or len({r['initial_state_sha256'] for r in rr})!=1:errors.append(f'{seed}: initial pairing missing/mismatch')
    complete=len(reports)==12 and checked==36 and not errors
    atomic_json(RUN/'artifacts/integrity.json',{'complete':complete,'prediction_sets_checked':checked,'expected_prediction_sets':36,'errors':errors,'future_access':False})
    lines=['# 样本规模 × 汇聚方式结果','',f'状态：{"completed" if complete else "incomplete"}。固定3200更新、20次选模；source20/80每类、固定valid510。','', '| 条件 | seed | best step | source accuracy | common source accuracy | valid accuracy | valid F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        for seed in c['seeds']:
            r=reports.get(f'{name}_{seed}')
            if not r:lines.append(f'| {name} | {seed} | incomplete | — | — | — | — |');continue
            lines.append(f"| {name} | {seed} | {r['best_block']*32} | {r['source']['accuracy']*100:.3f}% | {r['common_source']['accuracy']*100:.3f}% | {r['valid']['accuracy']*100:.3f}% | {r['valid']['macro_f1']*100:.3f}% |")
    summary={'complete':complete,'errors':errors}
    if complete:
        means={n:{role:{m:statistics.mean(reports[f'{n}_{s}'][role][m] for s in c['seeds']) for m in ['accuracy','macro_f1']} for role in ['source','common_source','valid']} for n in names}
        comparisons=[]
        for left,right in c['comparisons']:
            ds=[100*(reports[f'{left}_{s}']['valid']['macro_f1']-reports[f'{right}_{s}']['valid']['macro_f1']) for s in c['seeds']]
            da=100*(means[left]['valid']['accuracy']-means[right]['valid']['accuracy'])
            passed=statistics.mean(ds)>=1 and all(d>0 for d in ds) and da>=0
            comparisons.append({'left':left,'right':right,'f1_by_seed_pp':ds,'mean_f1_pp':statistics.mean(ds),'mean_accuracy_pp':da,'passed':passed})
            lines.extend(['',f'{left}−{right}: F1平均{statistics.mean(ds):+.3f}pp，逐seed{ds}，accuracy {da:+.3f}pp；门槛{"PASS" if passed else "FAIL"}。'])
        factorial=[]
        for s in c['seeds']:
            a,b,cc,d=[100*reports[f'{n}_{s}']['valid']['macro_f1'] for n in names]
            factorial.append({'seed':s,'sample_main_effect_pp':((cc-a)+(d-b))/2,'readout_main_effect_pp':((b-a)+(d-cc))/2,'interaction_pp':d-cc-b+a})
        summary.update(means=means,comparisons=comparisons,factorial=factorial)
        lines.extend(['','| 条件 | mean valid accuracy | mean valid F1 | common source accuracy |','|---|---:|---:|---:|'])
        for n,v in means.items():lines.append(f"| {n} | {v['valid']['accuracy']*100:.3f}% | {v['valid']['macro_f1']*100:.3f}% | {v['common_source']['accuracy']*100:.3f}% |")
        for f in ['sample_main_effect_pp','readout_main_effect_pp','interaction_pp']:
            vs=[v[f] for v in factorial];lines.extend(['',f'{f}: {vs}，均值{statistics.mean(vs):+.3f}pp（描述性）。'])
        vals='/'.join(f"{means[n]['valid']['macro_f1']*100:.3f}" for n in names)
        passed=[x['left']+'−'+x['right'] for x in comparisons if x['passed']]
        message=f'样本规模×汇聚完成：A/B/C/D valid F1 {vals}%；通过比较{passed}；36/36预测核验。'
    else:message=f'样本规模×汇聚实验不完整：{checked}/36核验，详见日志/完整性记录，不追加预算。'
    lines.extend(['',message,'',f'核验错误：{errors}。','', '限制：扩大source增加标签预算，固定204800样本呈现使20/80每类的平均数据遍历次数为100.39/25.10，不是相同epoch数。源训练指标比较请同时查看共同2040条的common_source；新模型seed没有提供新的数据抽样重复。masked mean同时改变汇聚规则并使用上下文化普通token，不能唯一归因于关系学习。固定valid已观察，仅开发证据，不是独立确认。未访问未来/外部数据，无预训练/TTA。','', '输入/代码/配置及审计散列见artifacts/freeze.json；每个job保存index_stream、sample_visits、历史、checkpoint与独立预测核验。'])
    atomic_json(RUN/'artifacts/summary.json',summary);(RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state','completed' if complete else 'stopped','--summary',message],check=True)
    p=ROOT/'STATUS.md';marker='当前样本规模与汇聚实验：`'+RUN.name+'`。'
    p.write_text('\n'.join(marker+message if line.startswith(marker) else line for line in p.read_text().splitlines())+'\n')
    print(message,flush=True)
    if not complete:raise SystemExit(2)
if __name__=='__main__':main()
