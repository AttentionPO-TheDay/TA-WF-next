"""Six new TAM encoders, three explicitly reused local baseline seeds; bounded GPU queue."""
from pathlib import Path
import subprocess,sys,os,time,signal,json,hashlib,traceback
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)

def update(state,summary):
 subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
 p=ROOT/'STATUS.md';lines=[x for x in p.read_text().splitlines() if not x.startswith('当前TAM多尺度生成器实验：')]
 lines.insert(1,'当前TAM多尺度生成器实验：`'+RUN.name+'`。'+summary);p.write_text('\n'.join(lines)+'\n')

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
 kinds=c['reported_conditions'];reports=[];verified=0
 raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
 for seed in c['seeds']:
  for kind in kinds:
   out=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((out/'report.json').read_text())
   assert rep['steps_completed']==c['optimizer_steps'] and rep['validation_opportunities']==len(c['eval_steps'])
   assert rep['parameters_total']==350006 and rep['parameters_trainable']==322726
   assert rep['verified_roles']==c['roles'] and rep['inactive_parameters_unchanged']
   if kind=='local_d1':assert rep['historical_reuse'] and rep['reloaded_with_new_local_d1_code']
   else:assert rep['generator_parameter_delta_l2']>0
   for ck in ('best','last'):
    with np.load(out/f'predictions_{ck}.npz') as pred:
     for role in c['roles']:
      y=raw[role]['labels'].numpy();p=pred[role]
      assert p.shape==y.shape and ((p>=0)&(p<102)).all()
      assert abs(accuracy_score(y,p)-rep[ck][role]['accuracy'])<1e-12
      assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-rep[ck][role]['macro_f1'])<1e-12
      verified+=1
   reports.append(rep)
  states=[torch.load(RUN/'artifacts'/f'{kind}_{seed}'/'initial_state.pt',weights_only=True) for kind in kinds]
  streams=[torch.load(RUN/'artifacts'/f'{kind}_{seed}'/'index_stream.pt',weights_only=True) for kind in kinds]
  assert all(torch.equal(v,streams[0]) for v in streams[1:])
  for state in states[1:]:
   assert state.keys()==states[0].keys()
   assert all(torch.equal(v,states[0][k]) for k,v in state.items())
 means={kind:{m:float(np.mean([rep['best']['valid'][m] for rep in reports if rep['kind']==kind])) for m in ('accuracy','macro_f1')} for kind in kinds}
 source_means={kind:float(np.mean([rep['best']['source']['accuracy'] for rep in reports if rep['kind']==kind])) for kind in kinds}
 def score(kind,seed,metric):return next(rep for rep in reports if rep['kind']==kind and rep['seed']==seed)['best']['valid'][metric]
 comparisons={}
 for a,b in (('context_d9','local_d1'),('multiscale_d139','local_d1'),('multiscale_d139','context_d9')):
  delta=[score(a,seed,'accuracy')-score(b,seed,'accuracy') for seed in c['seeds']]
  fd=[score(a,seed,'macro_f1')-score(b,seed,'macro_f1') for seed in c['seeds']]
  comparisons[a+' - '+b]={'accuracy_mean_delta_pp':float(np.mean(delta)*100),'accuracy_per_seed_pp':[float(v*100) for v in delta],
   'macro_f1_mean_delta_pp':float(np.mean(fd)*100),'pass':bool(min(delta)>0 and np.mean(delta)>=c['decision_gate']['accuracy_gain'] and np.mean(fd)>=0)}
 improved_context=comparisons['context_d9 - local_d1']['pass']
 improved_multi=comparisons['multiscale_d139 - local_d1']['pass']
 beyond_context=comparisons['multiscale_d139 - context_d9']['pass']
 if improved_multi and beyond_context:
  verdict='共享权重多尺度编码相对原局部编码及相同最大感受野单尺度均过门槛，保留此生成器候选；增加计算/采样密度仍是解释边界。'
 elif improved_context and not beyond_context:
  verdict='扩大单尺度上下文过增量门槛；未确认多尺度组合额外收益，优先保留较低计算的上下文候选。'
 elif improved_multi:
  verdict='多尺度编码相对原编码过门槛，但相对大范围单尺度未过；可保留候选，不能确认收益特异来自多尺度。'
 else:
  verdict='本轮多尺度候选未过原编码增量门槛；报告阴性/混合证据，不直接叠加漂移机制或继续扩大同一valid搜索。'
 target={kind:v['accuracy']>=c['decision_gate']['absolute_accuracy'] for kind,v in means.items()}
 y=raw['valid']['labels'].numpy();transitions={}
 for seed in c['seeds']:
  with np.load(RUN/'artifacts'/f'local_d1_{seed}'/'predictions_best.npz') as f:base=f['valid']==y
  transitions[str(seed)]={}
  for kind in c['conditions']:
   with np.load(RUN/'artifacts'/f'{kind}_{seed}'/'predictions_best.npz') as f:changed=f['valid']==y
   transitions[str(seed)][kind]={'candidate_correct_base_wrong':int((changed&~base).sum()),'candidate_wrong_base_correct':int((~changed&base).sum()),'both_wrong':int((~changed&~base).sum())}
 summary='6项新训练完成、3项原TAM编码历史复用；valid accuracy '+ '/'.join(f"{kind}={100*v['accuracy']:.3f}%" for kind,v in means.items())+f'；{verified}组预测指标核验通过。'
 result={'reports':reports,'means':means,'source_accuracy_means':source_means,'comparisons':comparisons,'error_transitions':transitions,
  'generator_verdict':verdict,'absolute_target_pass':target,'independent_prediction_metric_checks':verified,
  'new_training_tasks':6,'historical_reused_tasks':3,'new_best_reload_prediction_checks':12,'historical_reload_prediction_checks':6,
  'parameter_counts_initialization_and_index_stream_checks':'passed'}
 save(RUN/'artifacts/summary.json',result)
 rows=['# TAM上下文与多尺度生成器结果','',summary,'',verdict,'','|编码|seed|来源|best步数|source accuracy|valid accuracy|valid Macro-F1|耗时秒|','|---|---:|---|---:|---:|---:|---:|---:|']
 for rep in reports:
  rows.append(f"|{rep['kind']}|{rep['seed']}|{'历史复用' if rep['kind']=='local_d1' else '新训练'}|{rep['best_step']}|{100*rep['best']['source']['accuracy']:.3f}%|{100*rep['best']['valid']['accuracy']:.3f}%|{100*rep['best']['valid']['macro_f1']:.3f}%|{rep['elapsed_seconds']:.1f}|")
 rows.extend(['','三seed均值：','','|编码|accuracy|Macro-F1|≥90%|','|---|---:|---:|---|'])
 for kind,v in means.items():rows.append(f"|{kind}|{100*v['accuracy']:.3f}%|{100*v['macro_f1']:.3f}%|{target[kind]}|")
 rows.extend(['','```json',json.dumps(comparisons,indent=2,ensure_ascii=False),'```','',
  '三个编码总参数350006/可训练322726/生成器可训练1472完全相同，共享初始权重、120×110时间token、固定30维计数、mask、Transformer、训练索引/步数及20次选模机会。',
  'local_d1=dilation1，context_d9=dilation9，multiscale_d139=dilation1/3/9共享同一套两层卷积、同位置均匀平均。最大理论感受野分别9/73/73bin（subbin均值前），不引入额外权重、生成器dropout或可学习scale gate。',
  '多尺度约三倍局部卷积调用，固定步数不代表GPU计算完全匹配；dilation9稀疏采样可能有栅格效应。多尺度与d9差值同时包括尺度覆盖、密度与计算，不能唯一归因于多尺度概念。序列位置及patch内原始计数保留不证明表示可逆或漂移不变。',
  '原编码三seed仅明确历史复用，不新增独立重复。全新训练从头初始化，无旧checkpoint warm start；历史checkpoint仅用于原source/valid复现。',
  '固定510 valid已观察并多轮开发，三seed不是三个独立数据集。源期达到90%与相对编码门槛分开裁决；没有未来日期/WTT/AWF/漂移适应或最终泛化确认。时间输入版本不自动适用于无时间戳数据集。',
  '来源、权限、预算与停止条件见PLAN.md/SOURCES.md，冻结hash见artifacts/freeze.json。保留所有seed、阴性和停止结果，不自动追加训练。'])
 (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');update('completed',summary)

def main():
 c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen'
 for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 todo=[(k,s) for s in c['seeds'] for k in c['conditions']];active=[];done=[];start=time.monotonic()
 update('running','大范围单尺度/共享权重多尺度各3seed，共6项新训练；原TAM编码3seed已审计复用。GPU0最多2并发，12800步/20次选模，仅source/valid。')
 def terminated(signum,frame):raise RuntimeError('supervisor received signal '+str(signum))
 signal.signal(signal.SIGTERM,terminated)
 try:
  while todo or active:
   if time.monotonic()-start>c['pipeline_seconds']:raise TimeoutError('pipeline budget exceeded')
   while todo and len(active)<c['max_parallel']:
    k,s=todo.pop(0);key=f'{k}_{s}';log=(RUN/'logs'/f'{key}.log').open('x')
    p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind',k,'--seed',str(s)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active.append((key,p,log,time.monotonic()))
   for row in list(active):
    key,p,log,started=row
    if p.poll() is not None:
     log.close();active.remove(row)
     if p.returncode!=0:raise RuntimeError(f'{key} exited {p.returncode}')
     assert (RUN/'artifacts'/key/'report.json').exists();done.append(key)
    elif time.monotonic()-started>c['job_seconds']+60:raise TimeoutError(key+' job watchdog')
   save(RUN/'artifacts/progress.json',{'status':'running','active':[{'task':k,'pid':p.pid} for k,p,_,_ in active],
    'completed':done,'pending':[f'{k}_{s}' for k,s in todo],'historical_local_seeds':c['seeds'],'elapsed_seconds':time.monotonic()-start})
   time.sleep(5)
  finalize(c);save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'historical_reused_tasks':3,'elapsed_seconds':time.monotonic()-start})
 except BaseException:
  for _,p,log,_ in active:stop(p);log.close()
  err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err)
  with (RUN/'RESULTS.md').open('a') as f:f.write('\n执行中止，已完成产物保留：\n'+err)
  save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err,'elapsed_seconds':time.monotonic()-start})
  update('failed','执行中止，见logs/error.txt；部分结果保留，未追加预算或未来评价。');raise

if __name__=='__main__':main()
