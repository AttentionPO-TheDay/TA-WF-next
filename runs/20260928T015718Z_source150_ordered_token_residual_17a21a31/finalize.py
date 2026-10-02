"""Aggregate all seeds, deployment and ordered-information comparisons."""
import statistics,subprocess
from common import *

def main():
 c=config_read();names=[v['id'] for v in c['conditions']];reports={};errors=[];new_checked=historical_checked=0
 for seed in c['seeds']:
  for name in names:
   key=f'{name}_{seed}';out=RUN/'artifacts'/key
   try:
    r=json.loads((out/'report.json').read_text());v=json.loads((out/'verification.json').read_text())
    assert r['status']=='completed' and v['complete'] and r['config_sha256']==sha(RUN/'config.json')
    assert r['historical_reuse']==(name=='A_base')
    for p,h in v['artifact_sha256'].items():assert sha(ROOT/p)==h
    reports[key]=r
    if name=='A_base':historical_checked+=v['prediction_sets_checked']
    else:new_checked+=v['prediction_sets_checked']
   except Exception as e:errors.append(f'{key}: {type(e).__name__}: {e}')
 for seed in c['seeds']:
  rr=[reports.get(f'{n}_{seed}') for n in names]
  if any(r is None for r in rr):continue
  if len({r['shared_initial_sha256'] for r in rr})!=1:errors.append(f'{seed}: shared initialization mismatch')
  if rr[1]['initial_state_sha256']!=rr[2]['initial_state_sha256']:errors.append(f'{seed}: residual initialization mismatch')
  ref=torch.load(RUN/'artifacts'/f'A_base_{seed}'/'index_stream.pt',weights_only=True)
  for n in names[1:]:
   if not torch.equal(ref,torch.load(RUN/'artifacts'/f'{n}_{seed}'/'index_stream.pt',weights_only=True)):errors.append(f'{seed}: sampling mismatch {n}')
 complete=len(reports)==9 and new_checked==36 and historical_checked==18 and not errors
 atomic_json(RUN/'artifacts/integrity.json',{'complete':complete,'new_prediction_sets_replayed':new_checked,'expected_new_prediction_sets':36,'historical_prediction_sets_recalculated':historical_checked,'expected_historical_prediction_sets':18,'errors':errors,'future_access':False})
 lines=['# 生成器有序方向残差结果','',f'状态：{"completed" if complete else "incomplete"}。A_base为历史基线，B/C共6个新训练任务。','',
 '| 条件 | seed | best step | source accuracy | valid accuracy | valid F1 | last source accuracy | last valid F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name in names:
  for seed in c['seeds']:
   r=reports.get(f'{name}_{seed}')
   if r:lines.append(f"| {name} | {seed} | {r['best_block']*32} | {r['source']['accuracy']*100:.3f}% | {r['valid']['accuracy']*100:.3f}% | {r['valid']['macro_f1']*100:.3f}% | {r['last']['source']['accuracy']*100:.3f}% | {r['last']['valid']['macro_f1']*100:.3f}% |")
   else:lines.append(f'| {name} | {seed} | incomplete | — | — | — | — | — |')
 summary={'complete':complete,'errors':errors}
 if complete:
  means={name:{stage:{role:{metric:statistics.mean((reports[f'{name}_{s}'] if stage=='best' else reports[f'{name}_{s}']['last'])[role][metric] for s in c['seeds']) for metric in ['accuracy','macro_f1']} for role in ['source','common_source','valid']} for stage in ['best','last']} for name in names}
  comparisons={}
  for left,right in c['comparisons']:
   da=[100*(reports[f'{left}_{s}']['valid']['accuracy']-reports[f'{right}_{s}']['valid']['accuracy']) for s in c['seeds']]
   df=[100*(reports[f'{left}_{s}']['valid']['macro_f1']-reports[f'{right}_{s}']['valid']['macro_f1']) for s in c['seeds']]
   passed=statistics.mean(da)>=1 and all(v>0 for v in da) and statistics.mean(df)>=0
   key=left+'−'+right;comparisons[key]={'accuracy_by_seed_pp':da,'mean_accuracy_pp':statistics.mean(da),'f1_by_seed_pp':df,'mean_f1_pp':statistics.mean(df),'passed':passed}
   lines.extend(['',f'{key}：accuracy平均{statistics.mean(da):+.3f}pp、逐seed{da}；F1平均{statistics.mean(df):+.3f}pp、逐seed{df}；候选PASS={passed}。'])
  candidate=all(comparisons[k]['passed'] for k in ['C_ordered−A_base','C_ordered−B_counts'])
  summary.update(means=means,comparisons=comparisons,ordered_candidate_pass=candidate)
  lines.extend(['','| 条件 | mean source accuracy | mean valid accuracy | mean valid F1 | parameters |','|---|---:|---:|---:|---:|'])
  for n in names:
   v=means[n]['best'];p=reports[f'{n}_{c["seeds"][0]}']['parameters']['total']
   lines.append(f"| {n} | {100*v['source']['accuracy']:.3f}% | {100*v['valid']['accuracy']:.3f}% | {100*v['valid']['macro_f1']:.3f}% | {p} |")
  branch={}
  for n in names[1:]:
   branch[n]=[]
   for seed in c['seeds']:
    hist=json.loads((RUN/'artifacts'/f'{n}_{seed}'/'history.json').read_text());best=next(h for h in hist if h['block']==reports[f'{n}_{seed}']['best_block'])
    branch[n].append({'seed':seed,'best_residual_l2':best['residual_parameter_delta_l2'],'last_residual_l2':hist[-1]['residual_parameter_delta_l2'],'best_residual_token_rms':best['residual_token_rms']})
  summary['branch_diagnostics']=branch
  acc='/'.join(f"{means[n]['best']['valid']['accuracy']*100:.3f}" for n in names);f1='/'.join(f"{means[n]['best']['valid']['macro_f1']*100:.3f}" for n in names)
  msg=f'150条有序残差完成：A/B/C valid accuracy {acc}%，F1 {f1}%；有序候选PASS={candidate}；36/36新预测重载、18/18历史指标复核。'
 else:msg=f'150条有序残差未完整通过：{len(reports)}/9模型、{new_checked}/36新预测重载、{historical_checked}/18历史复核；不追加预算。'
 lines.extend(['',msg,'',f'错误：{errors}。','',
 '限制：A为原B150×1层历史结果，其重载证据来自原实验；本次只检查原产物hash、独立重算保存预测指标及state加载。B/C同参数但统计控制输入有效秩较低，有序路径比较不能唯一因果定位顺序损失。旧卷积路径已有局部顺序信息。固定valid510已用于多轮开发，不是独立确认；本轮不访问Day14、其他未来、外部数据，不做预训练/TTA。'])
 atomic_json(RUN/'artifacts/summary.json',summary);(RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state','completed' if complete else 'stopped','--summary',msg],check=True)
 p=ROOT/'STATUS.md';marker='当前有序残差实验：`'+RUN.name+'`。';p.write_text('\n'.join(marker+msg if line.startswith(marker) else line for line in p.read_text().splitlines())+'\n')
 print(msg,flush=True)
 if not complete:raise SystemExit(2)
if __name__=='__main__':main()
