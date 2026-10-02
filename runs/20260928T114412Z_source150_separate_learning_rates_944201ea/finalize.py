import statistics,subprocess,json
from common import *

def main():
    c=config_read();names=[v['id'] for v in c['conditions']];reports={};errors=[];checked=0
    for seed in c['seeds']:
        for name in names:
            key=f'{name}_{seed}';out=RUN/'artifacts'/key
            try:
                r=json.loads((out/'report.json').read_text());v=json.loads((out/'verification.json').read_text());assert r['status']=='completed' and v['complete'] and r['config_sha256']==sha(RUN/'config.json') and not r['historical_reuse']
                for p,h in v['artifact_sha256'].items(): assert sha(ROOT/p)==h,p
                reports[key]=r;checked+=v['prediction_sets_checked']
            except Exception as e: errors.append(f'{key}: {type(e).__name__}: {e}')
    for seed in c['seeds']:
        rr=[reports.get(f'{n}_{seed}') for n in names]
        if any(r is None for r in rr): continue
        if len({r['initial_state_sha256'] for r in rr})!=1: errors.append(f'{seed}: common initialization mismatch')
        streams=[torch.load(RUN/'artifacts'/f'{n}_{seed}'/'index_stream.pt',weights_only=True) for n in names]
        if any(not torch.equal(streams[0],x) for x in streams[1:]): errors.append(f'{seed}: shared batch index mismatch')
    complete=len(reports)==12 and checked==48 and not errors
    atomic_json(RUN/'artifacts/integrity.json',{'complete':complete,'prediction_sets_checked':checked,'expected_prediction_sets':48,'errors':errors,'future_access':False})
    lines=['# 生成器/分类器分离学习率结果','',f'状态：{"completed" if complete else "incomplete"}。四条件×三 seed 均为本轮新训练。','', '| 条件 | seed | best step | source acc | source F1 | valid acc | valid F1 | last valid F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        for seed in c['seeds']:
            r=reports.get(f'{name}_{seed}')
            if r: lines.append(f"| {name} | {seed} | {r['best_block']*32} | {r['source']['accuracy']*100:.3f}% | {r['source']['macro_f1']*100:.3f}% | {r['valid']['accuracy']*100:.3f}% | {r['valid']['macro_f1']*100:.3f}% | {r['last']['valid']['macro_f1']*100:.3f}% |")
            else: lines.append(f'| {name} | {seed} | incomplete | — | — | — | — | — |')
    summary={'complete':complete,'errors':errors}
    if complete:
        means={n:{role:{m:statistics.mean(100*reports[f'{n}_{s}'][role][m] for s in c['seeds']) for m in ['accuracy','macro_f1']} for role in ['source','valid']} for n in names}
        comparisons={}
        for left,right in c['comparisons']:
            f1=[100*(reports[f'{left}_{s}']['valid']['macro_f1']-reports[f'{right}_{s}']['valid']['macro_f1']) for s in c['seeds']];acc=[100*(reports[f'{left}_{s}']['valid']['accuracy']-reports[f'{right}_{s}']['valid']['accuracy']) for s in c['seeds']]
            passed=statistics.mean(f1)>=c['gate']['mean_f1_delta_min_pp'] and all(x>0 for x in f1) and statistics.mean(acc)>=c['gate']['mean_accuracy_delta_min_pp'];comparisons[left+'−'+right]={'f1_by_seed_pp':f1,'mean_f1_pp':statistics.mean(f1),'accuracy_by_seed_pp':acc,'mean_accuracy_pp':statistics.mean(acc),'passed':passed}
            lines.append(f"\n{left} 相对 {right}：valid F1 平均 {statistics.mean(f1):+.3f}pp，逐 seed {f1}；accuracy 平均 {statistics.mean(acc):+.3f}pp；候选 PASS={passed}。")
        summary.update(means=means,comparisons=comparisons)
        lines += ['', '| 条件 | mean source acc | mean source F1 | mean valid acc | mean valid F1 |','|---|---:|---:|---:|---:|']
        for n in names: lines.append(f"| {n} | {means[n]['source']['accuracy']:.3f}% | {means[n]['source']['macro_f1']:.3f}% | {means[n]['valid']['accuracy']:.3f}% | {means[n]['valid']['macro_f1']:.3f}% |")
        passed=[k for k,v in comparisons.items() if v['passed']];values='/'.join(f"{means[n]['valid']['macro_f1']:.3f}" for n in names);message=f"分离学习率完成：A/B/C/D valid F1 {values}%；通过比较{passed}；48/48预测核验。"
    else: message=f'分离学习率未完整通过：{len(reports)}/12模型、{checked}/48预测核验；保留产物。'
    lines += ['',message,'','错误：'+json.dumps(errors,ensure_ascii=False),'','限制：固定 valid 已用于开发选模；所有条件共享初始化与 batch 流，但不同学习率改变两个模块的联合优化轨迹，不能把结果归因于单独一个模块。仅 source/valid，Day14及外部未来数据未访问。']
    atomic_json(RUN/'artifacts/summary.json',summary);(RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n');subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state','completed' if complete else 'stopped','--summary',message],check=True)
    p=ROOT/'STATUS.md';marker='当前分离学习率实验：`'+RUN.name+'`。';lines_status=p.read_text().splitlines();lines_status=[(marker+message) if line.startswith(marker) else line for line in lines_status];
    if not any(line.startswith(marker) for line in p.read_text().splitlines()): lines_status.insert(1,marker+message)
    p.write_text('\n'.join(lines_status)+'\n');print(message,flush=True)
    if not complete: raise SystemExit(2)
if __name__=='__main__':main()
