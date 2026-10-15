import json,os,signal,subprocess,sys,time,traceback
import numpy as np
from common import RUN,ROOT,config,verify_freeze,save

def update(state,summary): subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
def main():
 c=config();verify_freeze();assert __import__('torch').cuda.is_available();todo=[(k,s) for s in c['seeds'] for k in c['conditions']];active=[];done=[];start=time.monotonic();update('running','时间顺序掩码组件 GPU0 实验启动；CE、硬/软标量模板、硬/软细粒度模板×3seed，未来关闭。')
 try:
  while todo or active:
   if time.monotonic()-start>c['pipeline_seconds']:raise TimeoutError('whole pipeline budget exceeded')
   while todo and len(active)<c['max_gpu_parallel']:
    kind,seed=todo.pop(0);log=(RUN/'logs'/f'{kind}_{seed}.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind',kind,'--seed',str(seed),'--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append((kind,seed,p,log,time.monotonic()))
   for item in list(active):
    kind,seed,p,log,launched=item
    if p.poll() is not None:
     log.close();active.remove(item)
     if p.returncode!=0:raise RuntimeError(f'{kind}_{seed} failed with {p.returncode}')
     assert (RUN/'artifacts'/f'{kind}_{seed}'/'report.json').exists();done.append(f'{kind}_{seed}')
    elif time.monotonic()-launched>c['gpu_job_seconds']+300:raise TimeoutError(f'{kind}_{seed} watchdog exceeded')
   save(RUN/'artifacts/progress.json',{'status':'running','active':[x[0]+'_'+str(x[1]) for x in active],'completed':done,'pending':[x[0]+'_'+str(x[1]) for x in todo],'elapsed_seconds':time.monotonic()-start});time.sleep(5)
  reports=[json.loads((RUN/'artifacts'/f'{k}_{s}'/'report.json').read_text()) for s in c['seeds'] for k in c['conditions']];by={k:[r for r in reports if r['kind']==k] for k in c['conditions']};means={k:{m:float(np.mean([r['best']['valid'][m] for r in by[k]])) for m in ('accuracy','macro_f1')} for k in c['conditions']};base=[r for r in reports if r['kind']=='ce_control'];comparisons={}
  for kind in c['conditions'][1:]:
   cand=[r for r in reports if r['kind']==kind];da=[100*(x['best']['valid']['accuracy']-y['best']['valid']['accuracy']) for x,y in zip(cand,base)];df=[100*(x['best']['valid']['macro_f1']-y['best']['valid']['macro_f1']) for x,y in zip(cand,base)];comparisons[kind]={'accuracy_mean_delta_pp':float(np.mean(da)),'accuracy_per_seed_pp':da,'macro_f1_mean_delta_pp':float(np.mean(df)),'positive_seed_count':int(sum(x>0 for x in da)),'pass':bool(np.mean(da)>=1 and min(da)>0 and np.mean(df)>=0)}
  summary={'means':means,'comparisons':comparisons,'reports':reports,'future_access':False,'training_tasks_completed':len(reports)};save(RUN/'artifacts/summary.json',summary)
  rows=['# 时间顺序掩码组件结果','','CE 与四种顺序掩码组件各三 seed 完成；历史 CE 为只读参考。','','| 条件 | valid accuracy | valid Macro-F1 |','|---|---:|---:|']+[f"| {k} | {means[k]['accuracy']*100:.3f}% | {means[k]['macro_f1']*100:.3f}% |" for k in c['conditions']];rows += ['', '```json',json.dumps(comparisons,ensure_ascii=False,indent=2),'```','','组件掩码仅用于 source 训练；valid 使用原始输入；未访问未来日期、WTT-Time、AWF 或 TTA。'];(RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'elapsed_seconds':time.monotonic()-start});update('completed','15项时间顺序掩码组件GPU0训练完成；结果见RESULTS.md，未来关闭。')
 except BaseException:
  for _,_,p,log,_ in active:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   log.close()
  err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err);save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err});update('failed','时间顺序掩码组件GPU0训练中止，部分产物保留，见logs/error.txt。');raise
if __name__=='__main__':main()
