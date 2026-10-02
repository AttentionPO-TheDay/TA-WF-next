"""Independent post-run checks, no fitting or future data."""
import json,hashlib
import torch,numpy as np
from sklearn.metrics import accuracy_score,f1_score
from common import RUN,ROOT,save

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def main():
 torch.set_num_threads(2);c=json.loads((RUN/'config.json').read_text());freeze=json.loads((RUN/'artifacts/freeze.json').read_text())
 for path,h in freeze.items():assert digest(ROOT/path)==h,path
 raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');summary=json.loads((RUN/'artifacts/summary.json').read_text());groups=0;rows=[];checks=[]
 for seed in c['seeds']:
  base=RUN/'artifacts'/f'shallow_transformer_{seed}';ref=torch.load(base/'initial_state.pt',weights_only=True);stream=torch.load(base/'index_stream.pt',weights_only=True);states={}
  for kind in c['reported_conditions']:
   out=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((out/'report.json').read_text());hist=json.loads((out/'history.json').read_text());state=torch.load(out/'initial_state.pt',weights_only=True);states[kind]=state
   assert rep['steps_completed']==12800 and rep['validation_opportunities']==20 and [v['step'] for v in hist]==c['eval_steps']
   assert rep['best_step']==max(hist,key=lambda v:v['valid']['accuracy'])['step']
   assert torch.equal(stream,torch.load(out/'index_stream.pt',weights_only=True))
   if kind!='rf_matched':
    for k,v in ref.items():
     if not k.startswith(('generator.','head.')):assert torch.equal(v,state[k]),(kind,k)
     if k.startswith('generator.') and kind.startswith('shallow'):assert torch.equal(v,state[k])
     if k.startswith('head.') and kind.endswith('transformer'):assert torch.equal(v,state[k])
   for phase in ['best','last']:
    with np.load(out/f'predictions_{phase}.npz') as f:
     for role in c['roles']:
      y=raw[role]['labels'].numpy();p=f[role];assert p.shape==y.shape and np.isin(p,np.arange(102)).all()
      assert abs(accuracy_score(y,p)-rep[phase][role]['accuracy'])<1e-12
      assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-rep[phase][role]['macro_f1'])<1e-12;groups+=1
   with np.load(out/'logits_best.npz') as z,np.load(out/'predictions_best.npz') as p:
    for role in c['roles']:assert np.isfinite(z[role]).all() and np.array_equal(z[role].argmax(1),p[role])
   if kind in c['conditions']:
    ck=torch.load(RUN/'checkpoints'/f'{kind}_{seed}_best.pt',weights_only=True,map_location='cpu');assert ck['step']==rep['best_step'] and ck['config_sha256']==digest(RUN/'config.json')
    assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
    assert rep['inactive_parameters_unchanged'] and rep['verified_roles']==['source','valid']
    assert rep['parameters_total']==c['parameter_counts'][kind]['total'] and rep['parameters_trainable']==c['parameter_counts'][kind]['trainable']
    grads=json.loads((out/'first_gradients.json').read_text())
    if kind!='rf_matched':
     assert rep['generator_parameter_delta_l2']>0
     assert all(v is not None and v>0 for k,v in grads.items() if k.startswith('generator.') and not k.startswith('generator.packet_conv.'))
    else:assert all(v is not None and np.isfinite(v) for v in grads.values())
   rows.append({'kind':kind,'seed':seed,'source_best':rep['best']['source']['accuracy'],'best':rep['best']['valid']['accuracy'],'last':rep['last']['valid']['accuracy'],'best_step':rep['best_step'],'elapsed_seconds':rep['elapsed_seconds']})
  assert all(torch.equal(v,states['progressive_mlp'][k]) for k,v in states['progressive_transformer'].items() if k.startswith('generator.'))
  assert all(torch.equal(v,states['progressive_mlp'][k]) for k,v in states['shallow_mlp'].items() if k.startswith('head.'))
 for kind in c['reported_conditions']:assert abs(np.mean([v['best'] for v in rows if v['kind']==kind])-summary['means'][kind]['accuracy'])<1e-12
 for a,b in c['comparisons']:
  ds=[next(v['best'] for v in rows if v['kind']==a and v['seed']==s)-next(v['best'] for v in rows if v['kind']==b and v['seed']==s) for s in c['seeds']]
  assert np.allclose(np.array(ds)*100,summary['comparisons'][a+' - '+b]['accuracy_per_seed_pp'],atol=1e-12,rtol=0)
 # Already scored predictions: descriptive error overlap only, no ensemble/oracle score.
 y=raw['valid']['labels'].numpy();pred={}
 for kind in ['shallow_transformer','progressive_transformer','progressive_mlp','rf_matched']:
  pred[kind]=np.stack([np.load(RUN/'artifacts'/f'{kind}_{s}'/'predictions_best.npz')['valid']==y for s in c['seeds']])
 overlap={}
 for kind,correct in pred.items():
  wrong=~correct.any(0);overlap[kind]={'common_wrong':int(wrong.sum()),'rf_all3_correct_among_common_wrong':int((wrong&pred['rf_matched'].all(0)).sum())}
 result={'passed':True,'freeze_hashes_verified':len(freeze),'metric_groups_verified':groups,'shared_initialization_indices_selection_checked':True,'logit_prediction_and_checkpoint_metadata_checked':True,'rows':rows,'last_below_best_count':sum(x['last']<x['best'] for x in rows),'error_overlap_descriptive':overlap,'cross_head_architecture_gate':summary['generator_cross_head_gate'],'own_model_accuracy_target_met':False,'official_rf_reference_accuracy_target_met':True,'future_access':False}
 save(RUN/'artifacts/postrun_audit.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
