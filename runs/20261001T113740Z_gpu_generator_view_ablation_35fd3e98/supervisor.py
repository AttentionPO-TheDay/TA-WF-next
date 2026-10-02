"""Nine new ablations, three explicitly reused fusion seeds; bounded GPU queue."""
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
 p=ROOT/'STATUS.md';lines=[x for x in p.read_text().splitlines() if not x.startswith('当前生成器视角消融实验：')]
 lines.insert(1,'当前生成器视角消融实验：`'+RUN.name+'`。'+summary);p.write_text('\n'.join(lines)+'\n')

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
 kinds=c['reported_conditions'];reports=[]
 raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
 verified=0
 for s in c['seeds']:
  for k in kinds:
   out=RUN/'artifacts'/f'{k}_{s}';rep=json.loads((out/'report.json').read_text())
   assert rep['steps_completed']==c['optimizer_steps'] and rep['validation_opportunities']==len(c['eval_steps'])
   assert rep['verified_roles']==c['roles']
   if k=='fusion':assert rep['historical_reuse'] and rep['reloaded_with_new_fusion_code']
   else:assert rep['inactive_parameters_unchanged'] and rep['generator_parameter_delta_l2']>0
   for ck in ('best','last'):
    with np.load(out/f'predictions_{ck}.npz') as pred:
     for role in c['roles']:
      y=raw[role]['labels'].numpy();p=pred[role]
      assert p.shape==y.shape and ((p>=0)&(p<102)).all()
      assert abs(accuracy_score(y,p)-rep[ck][role]['accuracy'])<1e-12
      assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-rep[ck][role]['macro_f1'])<1e-12
      verified+=1
   reports.append(rep)
  states=[torch.load(RUN/'artifacts'/f'{k}_{s}'/'initial_state.pt',weights_only=True) for k in kinds]
  streams=[torch.load(RUN/'artifacts'/f'{k}_{s}'/'index_stream.pt',weights_only=True) for k in kinds]
  assert all(torch.equal(v,streams[0]) for v in streams[1:])
  for state in states[1:]:
   assert state.keys()==states[0].keys()
   assert all(torch.equal(v,states[0][k]) for k,v in state.items())
 means={k:{m:float(np.mean([r['best']['valid'][m] for r in reports if r['kind']==k])) for m in ('accuracy','macro_f1')} for k in kinds}
 source_means={k:float(np.mean([r['best']['source']['accuracy'] for r in reports if r['kind']==k])) for k in kinds}
 def score(k,s,m):return next(r for r in reports if r['kind']==k and r['seed']==s)['best']['valid'][m]
 comparisons={}
 for a,b in [('fusion',k) for k in c['conditions']]+[('packet_native','packet_direction'),('tam_only','packet_direction'),('packet_native','fusion'),('tam_only','fusion')]:
  delta=[score(a,s,'accuracy')-score(b,s,'accuracy') for s in c['seeds']]
  fd=[score(a,s,'macro_f1')-score(b,s,'macro_f1') for s in c['seeds']]
  comparisons[a+' - '+b]={'accuracy_mean_delta_pp':float(np.mean(delta)*100),'accuracy_per_seed_pp':[float(x*100) for x in delta],
   'macro_f1_mean_delta_pp':float(np.mean(fd)*100),'pass':bool(min(delta)>0 and np.mean(delta)>=c['decision_gate']['accuracy_gain'] and np.mean(fd)>=0)}
 transitions={};y=raw['valid']['labels'].numpy()
 for s in c['seeds']:
  with np.load(RUN/'artifacts'/f'fusion_{s}'/'predictions_best.npz') as f:fc=f['valid']==y
  transitions[str(s)]={}
  for k in c['conditions']:
   with np.load(RUN/'artifacts'/f'{k}_{s}'/'predictions_best.npz') as f:sc=f['valid']==y
   transitions[str(s)][k]={'fusion_correct_single_wrong':int((fc&~sc).sum()),'fusion_wrong_single_correct':int((~fc&sc).sum()),'both_wrong':int((~fc&~sc).sum())}
 if comparisons['fusion - packet_native']['pass'] and comparisons['fusion - tam_only']['pass']:
  verdict='融合相对原packet和TAM两个单视角均过增量门槛，保留双视角候选；90%及漂移有效性另行判断。'
 elif comparisons['packet_native - fusion']['pass'] or comparisons['tam_only - fusion']['pass']:
  verdict='至少一个单视角稳定胜过本版融合；优先审查跨视角交互或汇聚接法，不把融合固定为必要结构。'
 else:
  verdict='本版融合未稳定超过两个单视角，证据混合；按配对数值选择下一个受控编码/融合问题，不宣称互补性已成立。'
 target={k:v['accuracy']>=c['decision_gate']['absolute_accuracy'] for k,v in means.items()}
 summary='9项新训练完成、3项融合历史复用；valid accuracy '+ '/'.join(f"{k}={100*v['accuracy']:.3f}%" for k,v in means.items())+f'；{verified}组预测指标核验通过。'
 result={'reports':reports,'means':means,'source_accuracy_means':source_means,'comparisons':comparisons,'error_transitions':transitions,
         'framework_verdict':verdict,'absolute_target_pass':target,'independent_prediction_metric_checks':verified,
         'new_training_tasks':9,'historical_reused_tasks':3,'new_best_reload_prediction_checks':18,'historical_reload_prediction_checks':6,
         'initialization_and_index_stream_checks':'passed'}
 save(RUN/'artifacts/summary.json',result)
 rows=['# 生成器视角消融结果','',summary,'',verdict,'','|条件|seed|来源|best步数|source accuracy|valid accuracy|valid Macro-F1|','|---|---:|---|---:|---:|---:|---:|']
 for r in reports:
  rows.append(f"|{r['kind']}|{r['seed']}|{'历史复用' if r['kind']=='fusion' else '新训练'}|{r['best_step']}|{100*r['best']['source']['accuracy']:.3f}%|{100*r['best']['valid']['accuracy']:.3f}%|{100*r['best']['valid']['macro_f1']:.3f}%|")
 rows.extend(['','三seed均值：','','|条件|accuracy|Macro-F1|≥90%|','|---|---:|---:|---|'])
 for k,v in means.items():rows.append(f"|{k}|{100*v['accuracy']:.3f}%|{100*v['macro_f1']:.3f}%|{target[k]}|")
 rows.extend(['','```json',json.dumps(comparisons,indent=2,ensure_ascii=False),'```','',
  'packet_direction仅方向/顺序/观测，时间幅值通道零；packet_native保留原packet逐包时间；tam_only为按方向分通道的时间计数而非不含方向的纯时间。融合是同版已完成seed，不是新增重复。',
  '所有新训练从头初始化，GPU预算、步数、样本顺序、20次accuracy选模与原融合匹配。移除视角时冻结独立参数并屏蔽attention/readout；有效信息、参数参与容量及计算仍不同，不能把差值唯一归因于信息本身。共享分类器的未使用半侧及view embedding行没有有效信号。',
  '固定510 valid已经参与多轮开发，三seed不是三个独立数据集。无未来日期、WTT/AWF、TTA或漂移调整，源期90%即便达成也不是独立泛化或抗漂移确认。',
  'packet_native−packet_direction诊断当前接法中逐包时间幅值的增量；fusion−packet_native诊断加入TAM整体增量；fusion−tam_only诊断加入原packet整体增量。模型参数/表示可用自由度随消融改变。',
  '复用来源见SOURCES.md、manifest.json、artifacts/historical_audit.json；冻结代码/数据/配置hash见artifacts/freeze.json。阴性与全部seed均报告，未自动追加训练或改法。'])
 (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');update('completed',summary)

def main():
 c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen'
 for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 todo=[(k,s) for s in c['seeds'] for k in c['conditions']];active=[];done=[];start=time.monotonic()
 update('running','纯方向packet/原packet/TAM-only各3seed，共9项新训练；融合三seed已审计复用。GPU0最多2并发，12800步/20次选模，仅source/valid。')
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
    'completed':done,'pending':[f'{k}_{s}' for k,s in todo],'historical_fusion_seeds':c['seeds'],'elapsed_seconds':time.monotonic()-start})
   time.sleep(5)
  finalize(c);save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'historical_reused_tasks':3,'elapsed_seconds':time.monotonic()-start})
 except BaseException:
  for _,p,log,_ in active:stop(p);log.close()
  err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err)
  with (RUN/'RESULTS.md').open('a') as f:f.write('\n执行中止，已完成产物保留：\n'+err)
  save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err,'elapsed_seconds':time.monotonic()-start})
  update('failed','执行中止，见logs/error.txt；部分结果保留，未追加预算或未来评价。');raise

if __name__=='__main__':main()
