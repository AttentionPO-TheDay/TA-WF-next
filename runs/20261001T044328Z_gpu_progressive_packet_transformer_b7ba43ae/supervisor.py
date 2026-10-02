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
 p=ROOT/'STATUS.md';lines=p.read_text().splitlines();lines=[x for x in lines if not x.startswith('当前渐进packet实验：')];lines.insert(1,'当前渐进packet实验：`'+RUN.name+'`。'+summary);p.write_text('\n'.join(lines)+'\n')
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
 # Source labels are loaded only for independently verifying saved predictions.
 raw=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False)
 hist=ROOT/c['historical_run'];historical=[];verified=0
 for kind in ['df','transformer']:
  for seed in c['seeds']:
   p=hist/'artifacts'/f'{kind}_{seed}';rep=json.loads((p/'report.json').read_text());pred=np.load(p/'predictions_best.npz')
   for role,entry in [('source','source150'),('valid','valid')]:
    y=raw[entry]['labels'].numpy();pr=pred[role]
    assert abs(accuracy_score(y,pr)-rep['best'][role]['accuracy'])<1e-12
    assert abs(f1_score(y,pr,average='macro',labels=np.arange(102),zero_division=0)-rep['best'][role]['macro_f1'])<1e-12;verified+=1
   historical.append({'kind':'historical_'+kind,'seed':seed,**{k:rep[k] for k in ['best','last','parameters','elapsed_seconds']},'historical_reuse':True})
 for seed in c['seeds']:
  a=RUN/'artifacts'/f'transformer_{seed}';b=RUN/'artifacts'/f'mlp_{seed}'
  assert sha(a/'index_stream.pt')==sha(b/'index_stream.pt')
  aa=torch.load(a/'initial_state.pt',weights_only=True);bb=torch.load(b/'initial_state.pt',weights_only=True)
  common=[k for k in aa if k in bb and (k.startswith('generator.') or k.startswith('readout.') or k.startswith('final_norm.') or 'position' in k)]
  assert common and any(k.startswith('generator.') for k in common)
  assert {'position','readout.weight','readout.bias','final_norm.weight','final_norm.bias'} <= set(common)
  for k in common:assert torch.equal(aa[k],bb[k]),k
 rows=['# 渐进packet CNN实验结果','','|条件|seed|训练accuracy|valid accuracy|valid F1|','|---|---:|---:|---:|---:|']
 for r in reports+historical:
  rows.append(f"|{r['kind']}|{r['seed']}|{100*r['best']['source']['accuracy']:.3f}%|{100*r['best']['valid']['accuracy']:.3f}%|{100*r['best']['valid']['macro_f1']:.3f}%|")
 allrep=reports+historical;means={k:{metric:float(np.mean([r['best']['valid'][metric] for r in allrep if r['kind']==k])) for metric in ['accuracy','macro_f1']} for k in ['transformer','mlp','historical_transformer','historical_df']}
 gates={}
 for candidate,ref in [('transformer','mlp'),('transformer','historical_transformer'),('transformer','historical_df'),('mlp','historical_transformer')]:
  delta=[next(r for r in allrep if r['kind']==candidate and r['seed']==s)['best']['valid']['macro_f1']-next(r for r in allrep if r['kind']==ref and r['seed']==s)['best']['valid']['macro_f1'] for s in c['seeds']]
  gates[candidate+' vs '+ref]={'f1_delta_pp':list(np.array(delta)*100),'mean_delta_pp':float(np.mean(delta)*100),'pass':bool(min(delta)>0 and np.mean(delta)>=.01 and means[candidate]['accuracy']>=means[ref]['accuracy'])}
 summary=f"6任务完成；CNN-Transformer/MLP valid F1={100*means['transformer']['macro_f1']:.3f}/{100*means['mlp']['macro_f1']:.3f}%；重载核验12份新预测、历史指标核验{verified}份通过。"
 rows.extend(['',summary,'',json.dumps({'means':means,'comparisons':gates},ensure_ascii=False,indent=2),'','解释限制：两新条件共享CNN、位置编码和读出初始化、批次流、训练预算；Transformer与MLP参数量不同，不是严格容量匹配。历史多视角与新packet输入信息一致但表示/架构不同；DF训练配方、参数量和呈现次数不同。历史结果仅复用，不算新重复。valid已观察、只有一次训练样本抽样、三seed非独立数据。无未来评价，不证明抗漂移。'])
 (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');save(RUN/'artifacts/summary.json',{'reports':reports,'historical':historical,'means':means,'comparisons':gates});update('completed',summary)

def main():
 c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen'
 for p,h in json.loads((RUN/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 todo=[(kind,seed) for seed in c['seeds'] for kind in c['conditions']];active=[];done=[];start=time.monotonic()
 update('running','GPU0最多2任务并发；packet CNN-Transformer/MLP各3seed；source/valid限定，未来关闭。')
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
