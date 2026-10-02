import json,hashlib
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import accuracy_score,f1_score
from common import RUN,ROOT,save

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((RUN/'config.json').read_text());freeze=json.loads((RUN/'artifacts/freeze.json').read_text())
 for p,h in freeze.items():assert sha(ROOT/p)==h,p
 raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');checks=0;rows=[]
 for seed in c['seeds']:
  stream=torch.load(RUN/'artifacts'/f'baseline_{seed}'/'index_stream.pt',weights_only=True)
  ref=torch.load(RUN/'artifacts'/f'baseline_{seed}'/'initial_state.pt',weights_only=True)
  for kind in c['reported_conditions']:
   d=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((d/'report.json').read_text());hist=json.loads((d/'history.json').read_text());assert rep['steps_completed']==12800 and len(hist)==20 and [x['step'] for x in hist]==c['eval_steps'];assert rep['best_step']==max(hist,key=lambda x:x['valid']['accuracy'])['step'];assert torch.equal(stream,torch.load(d/'index_stream.pt',weights_only=True))
   state=torch.load(d/'initial_state.pt',weights_only=True)
   for k,v in ref.items():
    if c['factors'][kind]['width']==1 or not k.startswith('generator.'):assert torch.equal(v,state[k]),(kind,k)
   if kind=='wide_bn':
    wide=torch.load(RUN/'artifacts'/f'wide_{seed}'/'initial_state.pt',weights_only=True);assert all(torch.equal(v,state[k]) for k,v in wide.items())
   for phase in ['best','last']:
    with np.load(d/f'predictions_{phase}.npz') as f:
     for role in c['roles']:
      y=raw[role]['labels'].numpy();p=f[role];assert p.shape==y.shape and np.isin(p,np.arange(102)).all();assert abs(accuracy_score(y,p)-rep[phase][role]['accuracy'])<1e-12;assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-rep[phase][role]['macro_f1'])<1e-12;checks+=1
   rows.append((kind,seed,rep['best']['valid']['accuracy'],rep['best']['valid']['macro_f1'],rep['last']['valid']['accuracy']))
   if kind in ['bn','wide_bn']:
    assert rep['bn_eval_buffers_unchanged'] and len(rep['bn_best_counters'])==6 and all(v==rep['best_step'] for v in rep['bn_best_counters'].values())
 summary=json.loads((RUN/'artifacts/summary.json').read_text());means={k:float(np.mean([x[2] for x in rows if x[0]==k])) for k in c['reported_conditions']};assert all(abs(means[k]-summary['means'][k]['accuracy'])<1e-12 for k in means)
 result={'passed':True,'freeze_hashes_verified':len(freeze),'metric_groups_verified':checks,'initialization_index_selection_verified':True,'bn_eval_statistics_verified':True,'rows':[{'kind':k,'seed':s,'valid_best':a,'macro_f1':f,'valid_last':l} for k,s,a,f,l in rows],'last_below_best_count':sum(x[4]<x[2] for x in rows),'future_access':False}
 save(RUN/'artifacts/postrun_audit.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
