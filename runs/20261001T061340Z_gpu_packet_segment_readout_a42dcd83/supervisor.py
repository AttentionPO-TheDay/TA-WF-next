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
 p=ROOT/'STATUS.md';lines=p.read_text().splitlines();lines=[x for x in lines if not x.startswith('当前分段汇聚实验：')];lines.insert(1,'当前分段汇聚实验：`'+RUN.name+'`。'+summary);p.write_text('\n'.join(lines)+'\n')
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def finalize(c):
 import numpy as np
 import torch
 from sklearn.metrics import accuracy_score,f1_score
 reports=[json.loads((RUN/'artifacts'/f'{k}_{s}'/'report.json').read_text()) for s in c['seeds'] for k in c['conditions']]
 raw=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False);hist=ROOT/c['historical_run'];historical=[]
 for seed in c['seeds']:
  p=hist/'artifacts'/f's150_mask_{seed}';rep=json.loads((p/'report.json').read_text());pred=np.load(p/'predictions_best.npz')
  for role,entry in [('source','source150'),('valid','valid')]:
   y=raw[entry]['labels'].numpy();pr=pred[role]
   assert abs(accuracy_score(y,pr)-rep['best'][role]['accuracy'])<1e-12
   assert abs(f1_score(y,pr,average='macro',labels=np.arange(102),zero_division=0)-rep['best'][role]['macro_f1'])<1e-12
  historical.append({'kind':'historical_mean','seed':seed,'best':rep['best'],'last':rep['last'],'historical_reuse':True})
 for seed in c['seeds']:
  states=[torch.load(RUN/'artifacts'/f'{k}_{seed}'/'initial_state.pt',weights_only=True) for k in c['conditions']]
  previous=torch.load(hist/'artifacts'/f's150_mask_{seed}'/'initial_state.pt',weights_only=True)
  for state in states:
   assert set(state)==set(previous)
   for k in state:
    if k.startswith('readout.'):continue
    assert torch.equal(state[k],previous[k]),k
  for k in states[0]:assert torch.equal(states[0][k],states[1][k]),k
  assert sha(RUN/'artifacts'/f'global_repeat_{seed}'/'index_stream.pt')==sha(RUN/'artifacts'/f'segments_{seed}'/'index_stream.pt')
  assert sha(RUN/'artifacts'/f'segments_{seed}'/'index_stream.pt')==sha(hist/'artifacts'/f's150_mask_{seed}'/'index_stream.pt')
 allrep=reports+historical
 means={k:{m:float(np.mean([r['best']['valid'][m] for r in allrep if r['kind']==k])) for m in ['accuracy','macro_f1']} for k in ['historical_mean']+c['conditions']}
 comparisons={}
 for a,b in [('segments','global_repeat'),('segments','historical_mean'),('global_repeat','historical_mean')]:
  delta=[next(r for r in allrep if r['kind']==a and r['seed']==s)['best']['valid']['macro_f1']-next(r for r in allrep if r['kind']==b and r['seed']==s)['best']['valid']['macro_f1'] for s in c['seeds']]
  comparisons[a+' - '+b]={'mean_delta_pp':float(np.mean(delta)*100),'per_seed_pp':list(np.array(delta)*100),'pass':bool(min(delta)>0 and np.mean(delta)>=.01 and means[a]['accuracy']>=means[b]['accuracy'])}
 rows=['# 分段汇聚结果','','|条件|seed|source accuracy|valid accuracy|valid F1|','|---|---:|---:|---:|---:|']
 for x in allrep:rows.append(f"|{x['kind']}|{x['seed']}|{100*x['best']['source']['accuracy']:.3f}%|{100*x['best']['valid']['accuracy']:.3f}%|{100*x['best']['valid']['macro_f1']:.3f}%|")
 summary='6任务完成；valid F1 '+ '/'.join(f"{k}={100*v['macro_f1']:.3f}%" for k,v in means.items())+'；12份新预测与6份历史预测指标核验通过。'
 rows.extend(['',summary,'',json.dumps({'means':means,'comparisons':comparisons},ensure_ascii=False,indent=2),'','限制：global_repeat与segments参数总量相同，但重复输入有效秩与表达能力不同，不能称有效容量完全匹配；全局重复扩大线性头改变优化参数化，历史mean为部署门槛。相对四分段按有效token分配，不是已验证资源或burst边界。固定valid已观察，不证明抗漂移或达到90%。'])
 (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');save(RUN/'artifacts/summary.json',{'reports':reports,'historical':historical,'means':means,'comparisons':comparisons});update('completed',summary)

def main():
 c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen'
 for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 todo=[(kind,seed) for seed in c['seeds'] for kind in c['conditions']];active=[];done=[];start=time.monotonic()
 update('running','GPU0最多2任务并发；全局重复/相对四段汇聚各3seed；source/valid限定，未来关闭。')
 try:
  while todo or active:
   assert time.monotonic()-start<c['pipeline_seconds'],'pipeline budget exceeded'
   while todo and len(active)<c['max_parallel']:
    k,s=todo.pop(0);key=f'{k}_{s}';log=(RUN/'logs'/f'{key}.log').open('x')
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
