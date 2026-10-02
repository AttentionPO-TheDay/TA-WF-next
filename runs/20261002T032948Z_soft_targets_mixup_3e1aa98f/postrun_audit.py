"""Post-run independent artifact audit; no training or future input."""
import json,hashlib
from pathlib import Path
import torch,numpy as np
from sklearn.metrics import accuracy_score,f1_score
from common import RUN,ROOT,save

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def main():
 torch.set_num_threads(2)
 frozen=json.loads((RUN/'artifacts/freeze.json').read_text())
 for p,h in frozen.items():assert digest(ROOT/p)==h,p
 c=json.loads((RUN/'config.json').read_text());raw=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False);checks=0;rows=[];shuffle=[]
 for seed in c['seeds']:
  base=RUN/'artifacts'/f'ce_only_{seed}'
  ref=torch.load(base/'initial_state.pt',weights_only=True);stream=torch.load(base/'index_stream.pt',weights_only=True)
  t=torch.load(RUN/'artifacts'/f'teacher_source_{seed}.pt',weights_only=True)
  assert set(t)=={'source_logits','source_rows','source_labels'}
  assert torch.equal(t['source_rows'],raw['source']['rows']) and torch.equal(t['source_labels'],raw['source']['labels'])
  perm=torch.randperm(len(t['source_logits']),generator=torch.Generator().manual_seed(seed+70000))
  shuffle.append({'seed':seed,'fixed_row_fraction':float((perm==torch.arange(len(perm))).float().mean()),'same_true_class_fraction':float((t['source_labels'][perm]==t['source_labels']).float().mean())})
  for kind in c['reported_conditions']:
   out=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((out/'report.json').read_text());hist=json.loads((out/'history.json').read_text())
   assert rep['steps_completed']==12800 and len(hist)==20 and [x['step'] for x in hist]==c['eval_steps']
   assert max(hist,key=lambda x:x['valid']['accuracy'])['step']==rep['best_step']
   assert torch.equal(stream,torch.load(out/'index_stream.pt',weights_only=True))
   state=torch.load(out/'initial_state.pt',weights_only=True);assert state.keys()==ref.keys() and all(torch.equal(state[k],v) for k,v in ref.items())
   for phase in ['best','last']:
    with np.load(out/f'predictions_{phase}.npz') as f:
     for role in ['source','valid']:
      y=raw[role]['labels'].numpy();p=f[role];assert p.shape==y.shape and np.isin(p,np.arange(102)).all()
      assert abs(accuracy_score(y,p)-rep[phase][role]['accuracy'])<1e-12
      assert abs(f1_score(y,p,average='macro',labels=np.arange(102),zero_division=0)-rep[phase][role]['macro_f1'])<1e-12;checks+=1
   if kind in c['conditions']:
    ck=torch.load(RUN/'checkpoints'/f'{kind}_{seed}_best.pt',weights_only=True,map_location='cpu');assert ck['step']==rep['best_step'] and ck['config_sha256']==digest(RUN/'config.json')
    spec=c['regularization']['conditions'][kind]
    assert rep['target_type']==spec['target'] and rep['mixup']==spec['mixup']
    assert rep['regularization_weight']==(.5 if spec['target'] else 0.)
    assert rep['inactive_parameters_unchanged'] and rep['generator_parameter_delta_l2']>0
    assert all(np.isfinite(x['train_aux']) and x['train_aux']>=-1e-6 for x in hist)
    if spec['target'] is None:assert all(x['train_aux']==0 for x in hist)
   rows.append({'kind':kind,'seed':seed,'best':rep['best']['valid']['accuracy'],'last':rep['last']['valid']['accuracy']})
 result={'passed':True,'freeze_hashes_verified':len(frozen),'metric_groups_verified':checks,'initialization_index_and_selection_verified':True,'teacher_source_identity_verified':True,'shuffled_mapping':shuffle,'last_below_best_count':sum(r['last']<r['best'] for r in rows),'conditions_seeds':rows,'new_training_tasks':12,'historical_tasks':6,'future_access':False}
 save(RUN/'artifacts/postrun_audit.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
