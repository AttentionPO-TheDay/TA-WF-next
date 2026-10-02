"""Nine depth/width tasks, three reused multiscale baseline seeds; bounded GPU queue."""
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
 p=ROOT/'STATUS.md';lines=[x for x in p.read_text().splitlines() if not x.startswith('当前TAM生成器深度容量实验：')]
 lines.insert(1,'当前TAM生成器深度容量实验：`'+RUN.name+'`。'+summary);p.write_text('\n'.join(lines)+'\n')

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
   expected=c['parameter_counts'][kind]
   assert rep['steps_completed']==c['optimizer_steps'] and rep['validation_opportunities']==len(c['eval_steps'])
   assert rep['parameters_total']==expected['total'] and rep['parameters_trainable']==expected['trainable']
   assert rep['verified_roles']==c['roles'] and rep['inactive_parameters_unchanged']
   if kind=='d2_w16':assert rep['historical_reuse'] and rep['reloaded_with_new_d2_w16_code']
   else:
    assert rep['generator_parameter_delta_l2']>0
    assert all(delta>0 for key,delta in rep['generator_parameter_deltas_squared'].items() if not key.startswith('packet_conv.'))
   for checkpoint in ('best','last'):
    with np.load(out/f'predictions_{checkpoint}.npz') as pred:
     for role in c['roles']:
      y=raw[role]['labels'].numpy();p=pred[role]
      assert p.shape==y.shape and ((p>=0)&(p<102)).all()
      assert abs(accuracy_score(y,p)-rep[checkpoint][role]['accuracy'])<1e-12
      assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-rep[checkpoint][role]['macro_f1'])<1e-12
      verified+=1
   reports.append(rep)
  states={kind:torch.load(RUN/'artifacts'/f'{kind}_{seed}'/'initial_state.pt',weights_only=True) for kind in kinds}
  streams=[torch.load(RUN/'artifacts'/f'{kind}_{seed}'/'index_stream.pt',weights_only=True) for kind in kinds]
  assert all(torch.equal(v,streams[0]) for v in streams[1:])
  ref=states['d2_w16']
  for kind,state in states.items():
   for key,value in state.items():
    if not key.startswith('generator.'):assert torch.equal(value,ref[key]),key
    if key.startswith('generator.depth_mix.'):assert not value.any(),key
  for shallow,deep in [('d2_w16','d4_w16'),('d2_w32','d4_w32')]:
   for key,value in states[shallow].items():
    if key.startswith('generator.time_conv.'):assert torch.equal(value,states[deep][key]),key
 means={kind:{metric:float(np.mean([rep['best']['valid'][metric] for rep in reports if rep['kind']==kind])) for metric in ('accuracy','macro_f1')} for kind in kinds}
 def score(kind,seed,metric):return next(rep for rep in reports if rep['kind']==kind and rep['seed']==seed)['best']['valid'][metric]
 comparisons={}
 for a,b in [('d2_w32','d2_w16'),('d4_w16','d2_w16'),('d4_w32','d2_w32'),('d4_w32','d4_w16'),('d4_w32','d2_w16')]:
  delta=[score(a,seed,'accuracy')-score(b,seed,'accuracy') for seed in c['seeds']]
  fd=[score(a,seed,'macro_f1')-score(b,seed,'macro_f1') for seed in c['seeds']]
  comparisons[a+' - '+b]={'accuracy_mean_delta_pp':float(np.mean(delta)*100),'accuracy_per_seed_pp':[float(v*100) for v in delta],
   'macro_f1_mean_delta_pp':float(np.mean(fd)*100),'pass':bool(min(delta)>0 and np.mean(delta)>=c['decision_gate']['accuracy_gain'] and np.mean(fd)>=0)}
 interaction={}
 for metric in ('accuracy','macro_f1'):
  values=[(score('d4_w32',seed,metric)-score('d2_w32',seed,metric))-(score('d4_w16',seed,metric)-score('d2_w16',seed,metric)) for seed in c['seeds']]
  interaction[metric]={'mean_pp':float(np.mean(values)*100),'per_seed_pp':[float(v*100) for v in values]}
 passes=[kind for kind in c['conditions'] if comparisons[kind+' - d2_w16']['pass']]
 # d4_w32 has its combined baseline comparison; d2_w32/d4_w16 direct ones.
 preferred=max(passes,key=lambda kind:(means[kind]['accuracy'],-c['parameter_counts'][kind]['generator_trainable'])) if passes else 'd2_w16'
 if passes:
  verdict='至少一个深度/容量候选稳定超过原生成器；优先候选为'+preferred+'，仍须分别解释深度、宽度及组合增量。'
 else:verdict='本轮深度/容量候选均未通过原生成器增量门槛；保留d2_w16开发基线，不自动继续扩大网络或叠加漂移模块。'
 target={kind:v['accuracy']>=c['decision_gate']['absolute_accuracy'] for kind,v in means.items()}
 transitions={};y=raw['valid']['labels'].numpy()
 for seed in c['seeds']:
  with np.load(RUN/'artifacts'/f'd2_w16_{seed}'/'predictions_best.npz') as f:base=f['valid']==y
  transitions[str(seed)]={}
  for kind in c['conditions']:
   with np.load(RUN/'artifacts'/f'{kind}_{seed}'/'predictions_best.npz') as f:changed=f['valid']==y
   transitions[str(seed)][kind]={'candidate_correct_base_wrong':int((changed&~base).sum()),'candidate_wrong_base_correct':int((~changed&base).sum()),'both_wrong':int((~changed&~base).sum())}
 summary='9项新训练完成、3项2层16通道多尺度基线历史复用；valid accuracy '+ '/'.join(f"{kind}={100*v['accuracy']:.3f}%" for kind,v in means.items())+f'；{verified}组预测指标核验通过。'
 result={'reports':reports,'means':means,'comparisons':comparisons,'interaction':interaction,'error_transitions':transitions,
  'generator_verdict':verdict,'preferred_candidate':preferred,'absolute_target_pass':target,'parameter_counts':c['parameter_counts'],
  'independent_prediction_metric_checks':verified,'new_training_tasks':9,'historical_reused_tasks':3,
  'new_best_reload_prediction_checks':18,'historical_reload_prediction_checks':6,'shared_initialization_and_index_stream_checks':'passed'}
 save(RUN/'artifacts/summary.json',result)
 rows=['# TAM生成器深度×宽度结果','',summary,'',verdict,'','|生成器|seed|来源|best步数|source accuracy|valid accuracy|valid Macro-F1|耗时秒|','|---|---:|---|---:|---:|---:|---:|---:|']
 for rep in reports:
  rows.append(f"|{rep['kind']}|{rep['seed']}|{'历史复用' if rep['kind']=='d2_w16' else '新训练'}|{rep['best_step']}|{100*rep['best']['source']['accuracy']:.3f}%|{100*rep['best']['valid']['accuracy']:.3f}%|{100*rep['best']['valid']['macro_f1']:.3f}%|{rep['elapsed_seconds']:.1f}|")
 rows.extend(['','三seed均值：','','|生成器|accuracy|Macro-F1|生成器可训练参数|≥90%|','|---|---:|---:|---:|---|'])
 for kind,v in means.items():rows.append(f"|{kind}|{100*v['accuracy']:.3f}%|{100*v['macro_f1']:.3f}%|{c['parameter_counts'][kind]['generator_trainable']}|{target[kind]}|")
 rows.extend(['','```json',json.dumps({'comparisons':comparisons,'interaction':interaction},indent=2,ensure_ascii=False),'```','',
  '所有条件仅TAM，固定dilation1/3/9、两层kernel5时间卷积、30维原计数统计、120×110 token、mask、Transformer和读出。深度4指在两层时间卷积中间增加两个kernel1点位残差层，初始化为identity；最大时间覆盖9/25/73bin不变。宽度改变第一层到中间残差层的通道16/32，最后输出均16。',
  '同seed classifier初始权重一致；同width两种depth的时间卷积初始化一致，4层初始输出精确等于对应2层。增加深度包括新增非线性、残差结构、参数及计算，不唯一识别抽象层数作用；宽度也改变函数自由度与参数。',
  '三份d2_w16仅明确历史复用，不是新独立重复；新模型从头初始化，无旧checkpoint warm start。梯度、状态、完整曲线、重载与指标审核通过。固定优化步数和选模次数，不代表GPU算力完全匹配。',
  '主指标accuracy；增量门槛均值≥1pp、3/3seed正、平均Macro-F1不下降。交互报告数值不声称统计确认。90%目标单独判断。',
  '固定510 valid已经多轮开发，不是新确认集；无未来日期/WTT/AWF/漂移调整。源期增益尚不是抗漂移或外部泛化证据，依赖时间的版本不能直接推广到缺时间戳数据。',
  '方案、来源、停止边界见PLAN.md/SOURCES.md；冻结hash见artifacts/freeze.json，不自动追加网络规模、预算或搜索。'])
 (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');update('completed',summary)

def main():
 c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen'
 for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 todo=[(k,s) for s in c['seeds'] for k in c['conditions']];active=[];done=[];start=time.monotonic()
 update('running','固定多尺度，2/4层×16/32通道；9项新训练、3项原多尺度基线复用。GPU0最多2并发，12800步/20次选模，仅source/valid。')
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
    'completed':done,'pending':[f'{k}_{s}' for k,s in todo],'historical_baseline_seeds':c['seeds'],'elapsed_seconds':time.monotonic()-start})
   time.sleep(5)
  finalize(c);save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'historical_reused_tasks':3,'elapsed_seconds':time.monotonic()-start})
 except BaseException:
  for _,p,log,_ in active:stop(p);log.close()
  err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err)
  with (RUN/'RESULTS.md').open('a') as f:f.write('\n执行中止，已完成产物保留：\n'+err)
  save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err,'elapsed_seconds':time.monotonic()-start})
  update('failed','执行中止，见logs/error.txt；部分结果保留，未追加预算或未来评价。');raise

if __name__=='__main__':main()
