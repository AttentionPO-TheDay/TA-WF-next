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
 raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');rows=[];checks=0
 for seed in c['seeds']:
  stream=torch.load(RUN/'artifacts'/f'baseline_{seed}'/'index_stream.pt',weights_only=True);ref=torch.load(RUN/'artifacts'/f'baseline_{seed}'/'initial_state.pt',weights_only=True)
  for kind in c['reported_conditions']:
   d=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((d/'report.json').read_text());hist=json.loads((d/'history.json').read_text());assert rep['steps_completed']==12800 and len(hist)==20 and [v['step'] for v in hist]==c['eval_steps'];assert rep['best_step']==max(hist,key=lambda v:v['valid']['accuracy'])['step'];assert torch.equal(stream,torch.load(d/'index_stream.pt',weights_only=True))
   state=torch.load(d/'initial_state.pt',weights_only=True);assert set(state)==set(ref)
   for k,v in ref.items():assert torch.equal(v,state[k]),(kind,k)
   for phase in ['best','last']:
    with np.load(d/f'predictions_{phase}.npz') as f:
     for role in c['roles']:
      y=raw[role]['labels'].numpy();p=f[role];assert p.shape==y.shape and np.isin(p,np.arange(102)).all();assert abs(accuracy_score(y,p)-rep[phase][role]['accuracy'])<1e-12;assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-rep[phase][role]['macro_f1'])<1e-12;checks+=1
   if kind!='baseline':assert rep['recipe']==c['optimizer_refinement'][kind]
   rows.append({'kind':kind,'seed':seed,'valid_accuracy':rep['best']['valid']['accuracy'],'macro_f1':rep['best']['valid']['macro_f1'],'last':rep['last']['valid']['accuracy'],'best_step':rep['best_step']})
 means={k:float(np.mean([x['valid_accuracy'] for x in rows if x['kind']==k])) for k in c['reported_conditions']};result={'passed':True,'freeze_hashes_verified':len(freeze),'metric_groups_verified':checks,'initialization_index_selection_verified':True,'rows':rows,'means':means,'last_below_best_count':sum(x['last']<x['valid_accuracy'] for x in rows),'aggregation_bug':'supervisor finalize referenced previous run comparison keys wide_bn/wide/bn/baseline; training artifacts unaffected','future_access':False};save(RUN/'artifacts/postrun_audit.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
