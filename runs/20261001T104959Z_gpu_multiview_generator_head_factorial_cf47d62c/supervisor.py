"""Bounded queue and full factorial reporting. No extra experiments on feedback."""
from pathlib import Path
import subprocess,sys,os,time,signal,json,hashlib,traceback
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)

def update(state,summary):
 subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
 p=ROOT/'STATUS.md'
 lines=[x for x in p.read_text().splitlines() if not x.startswith('当前多视角生成器×分类器实验：')]
 lines.insert(1,'当前多视角生成器×分类器实验：`'+RUN.name+'`。'+summary)
 p.write_text('\n'.join(lines)+'\n')

def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def finalize(c):
 import numpy as np
 import torch
 from sklearn.metrics import accuracy_score,f1_score
 torch.set_num_threads(2)
 reports=[json.loads((RUN/'artifacts'/f'{k}_{s}'/'report.json').read_text()) for s in c['seeds'] for k in c['conditions']]
 raw=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
 verified=0
 for r in reports:
  assert r['steps_completed']==c['optimizer_steps'] and r['validation_opportunities']==len(c['eval_steps'])
  assert r['verified_roles']==['source','valid']
  assert (r['generator_parameter_delta_l2']==0) if r['kind'].startswith('fixed') else (r['generator_parameter_delta_l2']>0)
  for checkpoint in ('best','last'):
   with np.load(RUN/'artifacts'/r['task']/f'predictions_{checkpoint}.npz') as preds:
    for role in c['roles']:
     y=raw[role]['labels'].numpy();p=preds[role]
     assert p.shape==y.shape and ((p>=0)&(p<102)).all()
     assert abs(accuracy_score(y,p)-r[checkpoint][role]['accuracy'])<1e-12
     assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-r[checkpoint][role]['macro_f1'])<1e-12
     verified+=1
 for seed in c['seeds']:
  states=[torch.load(RUN/'artifacts'/f'{k}_{seed}'/'initial_state.pt',weights_only=True) for k in c['conditions']]
  indices=[torch.load(RUN/'artifacts'/f'{k}_{seed}'/'index_stream.pt',weights_only=True) for k in c['conditions']]
  assert all(torch.equal(x,indices[0]) for x in indices[1:])
  for state in states[1:]:
   for key,value in state.items():
    if not key.startswith('head.'):assert torch.equal(value,states[0][key]),key
  # The generator factor does not alter initialization of either classifier.
  for left,right in ((0,2),(1,3)):
   assert states[left].keys()==states[right].keys()
   assert all(torch.equal(v,states[right][k]) for k,v in states[left].items())
 means={k:{m:float(np.mean([r['best']['valid'][m] for r in reports if r['kind']==k])) for m in ('accuracy','macro_f1')} for k in c['conditions']}
 def score(kind,seed,metric):
  return next(r for r in reports if r['kind']==kind and r['seed']==seed)['best']['valid'][metric]
 comparisons={}
 for a,b in (('learned_mlp','fixed_mlp'),('learned_transformer','fixed_transformer'),('fixed_transformer','fixed_mlp'),('learned_transformer','learned_mlp'),('learned_transformer','fixed_mlp')):
  delta=[score(a,s,'accuracy')-score(b,s,'accuracy') for s in c['seeds']]
  fdelta=[score(a,s,'macro_f1')-score(b,s,'macro_f1') for s in c['seeds']]
  comparisons[a+' - '+b]={'accuracy_mean_delta_pp':float(np.mean(delta)*100),'accuracy_per_seed_pp':[float(v*100) for v in delta],
   'macro_f1_mean_delta_pp':float(np.mean(fdelta)*100),'pass':bool(min(delta)>0 and np.mean(delta)>=c['decision_gate']['accuracy_gain'] and np.mean(fdelta)>=0)}
 interaction={m:[(score('learned_transformer',s,m)-score('learned_mlp',s,m))-(score('fixed_transformer',s,m)-score('fixed_mlp',s,m)) for s in c['seeds']] for m in ('accuracy','macro_f1')}
 interaction={m:{'mean_pp':float(np.mean(d)*100),'per_seed_pp':[float(v*100) for v in d]} for m,d in interaction.items()}
 generator_mlp=comparisons['learned_mlp - fixed_mlp']['pass']
 generator_transformer=comparisons['learned_transformer - fixed_transformer']['pass']
 transformer_learned=comparisons['learned_transformer - learned_mlp']['pass']
 best=max(c['conditions'],key=lambda k:means[k]['accuracy'])
 if generator_transformer and transformer_learned and best=='learned_transformer':
  verdict='保留当前可学习生成器+Transformer为优先候选；仍需独立设计漂移阶段与对照。'
 elif generator_mlp and not transformer_learned:
  verdict='可学习生成器有MLP侧增量证据；优先保留生成器接口，分类器转向MLP候选。未通过门槛不等于Transformer完全无效。'
 elif not generator_mlp and not generator_transformer:
  verdict='本版可学习局部特征未过归因门槛；重新设计生成器的信息保留/汇聚是候选，不应继续直接叠加漂移模块。'
 else:
  verdict='证据混合；保留逐因素结论，不能确认生成器+Transformer协同或据此决定唯一整体框架。'
 target={k:v['accuracy']>=c['decision_gate']['absolute_accuracy'] for k,v in means.items()}
 summary='12任务完成；valid accuracy '+ '/'.join(f"{k}={100*v['accuracy']:.3f}%" for k,v in means.items())+f'；{verified}份预测独立指标核验、24份best预测重载通过。'
 payload={'reports':reports,'means':means,'comparisons':comparisons,'interaction':interaction,'absolute_target_pass':target,
          'highest_mean_condition':best,'framework_verdict':verdict,'independent_prediction_metric_checks':verified,
          'shared_initialization_and_index_stream_checks':'passed','historical_references':c['historical_references']}
 save(RUN/'artifacts/summary.json',payload)
 rows=['# 多视角生成器×分类器结果','',''+summary,'',verdict,'','|条件|seed|best步数|source accuracy|valid accuracy|valid Macro-F1|','|---|---:|---:|---:|---:|---:|']
 for r in reports:
  rows.append(f"|{r['kind']}|{r['seed']}|{r['best_step']}|{100*r['best']['source']['accuracy']:.3f}%|{100*r['best']['valid']['accuracy']:.3f}%|{100*r['best']['valid']['macro_f1']:.3f}%|")
 rows.extend(['','三seed均值：','','|条件|accuracy|Macro-F1|≥90%|','|---|---:|---:|---|'])
 for k,v in means.items():rows.append(f"|{k}|{100*v['accuracy']:.3f}%|{100*v['macro_f1']:.3f}%|{target[k]}|")
 rows.extend(['','配对增量门槛：accuracy均值≥1pp、3/3seed为正、平均Macro-F1不下降。交互只报告数值，不宣称统计确认。','', '```json',json.dumps({'comparisons':comparisons,'interaction':interaction},ensure_ascii=False,indent=2),'```','',
  '历史RF 85.948%、方向CNN+MLP+mask 73.987%仅为已观察性能参照，不是本轮相同输入表示、训练配方及预算的归因对照。',
  '本轮固定/可学习共享统计及mask；可学习条件额外80维CNN特征改变有效特征自由度。新增特征整体增益不能单独归因于学习本身。此模型是自有系统的新版本，不是旧版原封不动或官方RF生成器移植；独立实现不等于研究新颖性。',
  '仅source/valid监督训练及预定选模；固定510 valid已多轮开发，三seed不是三个独立验证集。无未来/WTT/AWF评分、预训练、TTA或漂移调整。当前时间输入版本不能自动声称适用于缺时间戳的数据集。',
  '代码、配置和数据hash见artifacts/freeze.json及manifest.json；本轮结果不自动授权额外训练。'])
 (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n')
 update('completed',summary)

def main():
 c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen'
 for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 todo=[(kind,seed) for seed in c['seeds'] for kind in c['conditions']]
 active=[];done=[];start=time.monotonic()
 update('running','固定/可学习生成器×MLP/Transformer共12任务，GPU0最多2并发；每任务12800步、20次valid选模；仅source/valid，未来关闭。')
 try:
  while todo or active:
   assert time.monotonic()-start<c['pipeline_seconds'],'pipeline budget exceeded'
   while todo and len(active)<c['max_parallel']:
    k,s=todo.pop(0);key=f'{k}_{s}';log=(RUN/'logs'/f'{key}.log').open('x')
    p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind',k,'--seed',str(s)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active.append((key,p,log,time.monotonic()))
   for row in list(active):
    key,p,log,started=row
    if p.poll() is not None:
     log.close();active.remove(row);assert p.returncode==0,f'{key} exited {p.returncode}'
     assert (RUN/'artifacts'/key/'report.json').exists();done.append(key)
    elif time.monotonic()-started>c['job_seconds']+60:raise TimeoutError(key+' job watchdog')
   save(RUN/'artifacts/progress.json',{'status':'running','active':[{'task':k,'pid':p.pid} for k,p,_,_ in active],
     'completed':done,'pending':[f'{k}_{s}' for k,s in todo],'elapsed_seconds':time.monotonic()-start})
   time.sleep(5)
  finalize(c)
  save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'elapsed_seconds':time.monotonic()-start})
 except BaseException:
  for _,p,log,_ in active:stop(p);log.close()
  err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err)
  with (RUN/'RESULTS.md').open('a') as f:f.write('\n执行中止，已完成产物保留：\n'+err)
  save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err,'elapsed_seconds':time.monotonic()-start})
  update('failed','执行中止，详见logs/error.txt；保留部分结果，未追加预算或未来评价。')
  raise

if __name__=='__main__':main()
