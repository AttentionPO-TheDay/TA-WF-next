from pathlib import Path
import subprocess,sys,os,time,signal,json,hashlib,traceback
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def update(state,summary):
 subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
 p=ROOT/'STATUS.md';lines=p.read_text().splitlines();lines=[x for x in lines if not x.startswith('当前原生基线实验：')];lines.insert(1,'当前原生基线实验：`'+RUN.name+'`。'+summary);p.write_text('\n'.join(lines)+'\n')
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def finalize(c):
 import numpy as np
 reports=[json.loads((RUN/'artifacts'/f'{k}_{s}'/'report.json').read_text()) for s in c['seeds'] for k in c['conditions']]
 means={k:{m:float(np.mean([r['best']['valid'][m] for r in reports if r['kind']==k])) for m in ['accuracy','macro_f1']} for k in c['conditions']}
 rows=['# 原生Var-CNN与RF源期比较','','|模型|seed|best epoch|总epoch|训练accuracy|valid accuracy|valid F1|','|---|---:|---:|---:|---:|---:|---:|']
 for r in reports:rows.append(f"|{r['kind']}|{r['seed']}|{r['best_epoch']}|{r['epochs_completed']}|{100*r['best']['source']['accuracy']:.3f}%|{100*r['best']['valid']['accuracy']:.3f}%|{100*r['best']['valid']['macro_f1']:.3f}%|")
 gates={k:v['accuracy']>=.90 for k,v in means.items()}
 summary='6任务完成；valid accuracy '+ '/'.join(f"{k}={100*v['accuracy']:.3f}%" for k,v in means.items())+'；12份best预测重载与独立指标核验通过。'
 rows.extend(['',summary,'',json.dumps({'means':means,'target_90_percent':gates},ensure_ascii=False,indent=2),'','历史参照：DF70.784% accuracy（方向、30轮）；CNN-MLP+遮挡73.987%（方向、原F1选模）。本轮统一accuracy选模，RF加入时间TAM；Var-CNN的验证驱动训练最多150次查看valid，而RF20次，历史20次，不能声称同输入信息/优化/选模预算公平。只比较各自冻结原生配方的源期开发性能。Keras移植仍有BN/RNG框架差异，无精确跨框架数值复现。无未来评价，不证明抗漂移；固定valid已观察。'])
 (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');save(RUN/'artifacts/summary.json',{'reports':reports,'means':means,'target_90_percent':gates});update('completed',summary)

def main():
 c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen'
 for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 todo=[(kind,seed) for seed in c['seeds'] for kind in c['conditions']];active=[];done=[];start=time.monotonic()
 update('running','GPU0最多2任务并发；Var-CNN方向/RF时间TAM各3seed；原生配方分列比较；source/valid限定，未来关闭。')
 try:
  while todo or active:
   assert time.monotonic()-start<c['pipeline_seconds'],'pipeline budget exceeded'
   while todo and len(active)<c['max_parallel']:
    allowed=next((i for i,(kind,seed) in enumerate(todo) if kind!='varcnn' or not any(row[0].startswith('varcnn_') for row in active)),None)
    if allowed is None:break
    k,s=todo.pop(allowed);key=f'{k}_{s}';log=(RUN/'logs'/f'{key}.log').open('x')
    p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind',k,'--seed',str(s)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append((key,p,log,time.monotonic()))
   for row in list(active):
    key,p,log,started=row
    if p.poll() is not None:
     log.close();active.remove(row);assert p.returncode==0,f'{key} exited {p.returncode}'
     assert (RUN/'artifacts'/key/'report.json').exists();done.append(key)
    elif time.monotonic()-started>c['job_seconds']+60:raise TimeoutError(key+' job watchdog')
   save(RUN/'artifacts/progress.json',{'status':'running','active':[{'task':k,'pid':p.pid} for k,p,_,_ in active],'completed':done,'pending':[f'{k}_{s}' for k,s in todo],'elapsed_seconds':time.monotonic()-start})
   time.sleep(5)
  finalize(c);save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'elapsed_seconds':time.monotonic()-start})
 except BaseException:
  for key,p,log,_ in active:stop(p);log.close()
  err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err)
  with (RUN/'RESULTS.md').open('a') as f:f.write('\n执行中止，已完成产物保留：\n'+err)
  update('failed','执行中止，详见logs/error.txt；已完成任务保留，未追加预算或未来评价。');raise
if __name__=='__main__':main()
