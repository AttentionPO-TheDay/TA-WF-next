"""All factorial cells, paired effects and complete/partial outcome reporting."""
import statistics,subprocess
from common import *

def main():
    c=config_read();names=[v['id'] for v in c['conditions']];reports={};errors=[];checked=0
    for seed in c['seeds']:
        for name in names:
            key=f'{name}_{seed}';out=RUN/'artifacts'/key
            try:
                r=json.loads((out/'report.json').read_text());v=json.loads((out/'verification.json').read_text())
                assert r['status']=='completed' and v['complete'] and r['config_sha256']==sha(RUN/'config.json')
                assert r['historical_reuse']==(name=='A_80_l1')
                for p,h in v['artifact_sha256'].items():assert sha(ROOT/p)==h,p
                reports[key]=r;checked+=v['prediction_sets_checked']
            except Exception as e:errors.append(f'{key}: {type(e).__name__}: {e}')
    for seed in c['seeds']:
        rr=[reports.get(f'{n}_{seed}') for n in names]
        if any(r is None for r in rr):continue
        if len({r['shared_initial_sha256'] for r in rr})!=1:errors.append(f'{seed}: common initialization mismatch')
        for left,right in [('A_80_l1','C_80_l2'),('B_150_l1','D_150_l2')]:
            a=torch.load(RUN/'artifacts'/f'{left}_{seed}'/'index_stream.pt',weights_only=True)
            b=torch.load(RUN/'artifacts'/f'{right}_{seed}'/'index_stream.pt',weights_only=True)
            if not torch.equal(a,b):errors.append(f'{seed}: sample stream mismatch {left}/{right}')
    complete=len(reports)==12 and checked==72 and not errors
    atomic_json(RUN/'artifacts/integrity.json',{'complete':complete,'prediction_sets_checked':checked,'expected_prediction_sets':72,'historical_prediction_sets':18,'errors':errors,'future_access':False})
    lines=['# 源期样本规模 × 局部深度结果','',f'状态：{"completed" if complete else "incomplete"}。A为明确复用的历史对照，其余9任务为新训练。','',
           '| 条件 | seed | best step | source accuracy | common80 accuracy | valid accuracy | valid F1 | last valid F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        for seed in c['seeds']:
            r=reports.get(f'{name}_{seed}')
            if r:lines.append(f"| {name} | {seed} | {r['best_block']*32} | {r['source']['accuracy']*100:.3f}% | {r['common_source']['accuracy']*100:.3f}% | {r['valid']['accuracy']*100:.3f}% | {r['valid']['macro_f1']*100:.3f}% | {r['last']['valid']['macro_f1']*100:.3f}% |")
            else:lines.append(f'| {name} | {seed} | incomplete | — | — | — | — | — |')
    summary={'complete':complete,'errors':errors,'historical_baseline':c['prior_run']}
    if complete:
        means={name:{stage:{role:{m:statistics.mean((reports[f'{name}_{s}'] if stage=='best' else reports[f'{name}_{s}']['last'])[role][m] for s in c['seeds']) for m in ['accuracy','macro_f1']} for role in ['source','common_source','valid']} for stage in ['best','last']} for name in names}
        comparisons={}
        for left,right in c['comparisons']:
            df=[100*(reports[f'{left}_{s}']['valid']['macro_f1']-reports[f'{right}_{s}']['valid']['macro_f1']) for s in c['seeds']]
            da=[100*(reports[f'{left}_{s}']['valid']['accuracy']-reports[f'{right}_{s}']['valid']['accuracy']) for s in c['seeds']]
            dt=[100*(reports[f'{left}_{s}']['common_source']['accuracy']-reports[f'{right}_{s}']['common_source']['accuracy']) for s in c['seeds']]
            passed=statistics.mean(df)>=1 and all(x>0 for x in df) and statistics.mean(da)>=0
            key=left+'−'+right;comparisons[key]={'f1_by_seed_pp':df,'mean_f1_pp':statistics.mean(df),'accuracy_by_seed_pp':da,'mean_accuracy_pp':statistics.mean(da),'common_source_accuracy_by_seed_pp':dt,'passed':passed,'joint_improvement':passed and statistics.mean(dt)>0}
            lines.extend(['',f'{key}：valid F1平均{statistics.mean(df):+.3f}pp，逐seed{df}；accuracy平均{statistics.mean(da):+.3f}pp；候选PASS={passed}。'])
        effects={}
        for metric_name in ['accuracy','macro_f1']:
            vals=[]
            for seed in c['seeds']:
                a,b,cc,d=[100*reports[f'{n}_{seed}']['valid'][metric_name] for n in names]
                vals.append({'seed':seed,'sample_main_pp':((b-a)+(d-cc))/2,'depth_main_pp':((cc-a)+(d-b))/2,'interaction_pp':(d-cc)-(b-a)})
            effects[metric_name]={'by_seed':vals,'means':{k:statistics.mean(v[k] for v in vals) for k in ['sample_main_pp','depth_main_pp','interaction_pp']}}
        summary.update(means=means,comparisons=comparisons,effects=effects)
        lines.extend(['','| 条件 | mean source accuracy | mean common80 accuracy | mean valid accuracy | mean valid F1 | parameters |','|---|---:|---:|---:|---:|---:|'])
        for n in names:
            v=means[n]['best'];p=reports[f'{n}_{c["seeds"][0]}']['parameters']['total']
            lines.append(f"| {n} | {100*v['source']['accuracy']:.3f}% | {100*v['common_source']['accuracy']:.3f}% | {100*v['valid']['accuracy']:.3f}% | {100*v['valid']['macro_f1']:.3f}% | {p} |")
        lines.extend(['','主效应与交互（描述性，非显著性检验）：',json.dumps(effects,ensure_ascii=False,indent=2)])
        values='/'.join(f"{100*means[n]['best']['valid']['macro_f1']:.3f}" for n in names);passed=[k for k,v in comparisons.items() if v['passed']]
        message=f'80/150条×1/2层完成：A/B/C/D valid F1 {values}%；通过比较{passed}；72/72预测核验（含18历史复核）。'
    else:message=f'80/150条×1/2层未完整通过：{len(reports)}/12模型、{checked}/72预测核验；保留产物，不追加预算。'
    lines.extend(['',message,'',f'错误：{errors}。','',
        '限制：A为历史训练结果，其他模型为本轮从头训练；初始化共同部分、总步数、LR与20次选模匹配，但已有验证反馈不消失。80/150条分别约100.39/53.54遍，相同步数不等于相同epoch或FLOPs。两层同时增加参数与计算，不能唯一归因于token交互。只一次嵌套样本抽样和固定valid510，不是独立确认。Day14及其他未来/外部数据未访问，无预训练/TTA。',
        '', '下一步先按源期结果决定候选；若需要增加预算须另立计划，不由本pipeline自动追加。'])
    atomic_json(RUN/'artifacts/summary.json',summary);(RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state','completed' if complete else 'stopped','--summary',message],check=True)
    p=ROOT/'STATUS.md';marker='当前样本规模×深度实验：`'+RUN.name+'`。'
    p.write_text('\n'.join(marker+message if line.startswith(marker) else line for line in p.read_text().splitlines())+'\n')
    print(message,flush=True)
    if not complete:raise SystemExit(2)
if __name__=='__main__':main()
