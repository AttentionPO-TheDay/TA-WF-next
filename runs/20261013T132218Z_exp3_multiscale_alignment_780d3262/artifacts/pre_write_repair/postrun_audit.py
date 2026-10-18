import json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import accuracy_score,f1_score
import port_base as B
R,ROOT,C=B.R,B.ROOT,B.C
scores=json.loads((R/'artifacts/scores.json').read_text())
assert len(scores)==36
hist=[x for x in scores if x['origin']=='historical_reference']; new=[x for x in scores if x['origin']=='new']
assert len(hist)==27 and len(new)==9
assert {x['condition'] for x in hist}=={'frozen','no_pseudo_fixed','dynamic_lr_only'}
assert {x['condition'] for x in new}==set(C['conditions'])
assert len(list((R/'checkpoints').glob('*.pt')))==9
assert len(list((R/'artifacts').glob('update_*.json')))==9
assert len(list((R/'artifacts').glob('entropy_gate_*.npz')))==9
assert not list((R/'artifacts').glob('gmm_*.npz'))
for p,v in json.loads((R/'artifacts/freeze.json').read_text()).items(): assert B.sha(ROOT/p)==v,p
for r in scores:
 p=R/'artifacts'/f"pred_{r['seed']}_{r['role']}_{r['condition']}.npz"
 with np.load(p) as z:
  y=z['labels'];pred=z['predictions'];logits=z['logits'];rows=z['rows']
  assert len(y)==r['rows'] and logits.shape==(len(y),102) and np.array_equal(pred,logits.argmax(1)) and np.isfinite(logits).all()
  assert abs(accuracy_score(y,pred)-r['accuracy'])<1e-12
  assert abs(f1_score(y,pred,labels=np.arange(102),average='macro',zero_division=0)-r['macro_f1'])<1e-12
  with np.load(R/'artifacts'/f"pred_{r['seed']}_{r['role']}_frozen.npz") as b:
   assert np.array_equal(rows,b['rows']) and np.array_equal(y,b['labels']);base=b['predictions']
  bc=base==y;bw=~bc
  for k,val in {'negative_adaptation_rate':float((bc&(pred!=y)).sum()/max(1,bc.sum())),'recovery_rate':float((bw&(pred==y)).sum()/max(1,bw.sum())),'changed_rate':float((base!=pred).mean())}.items():assert abs(val-r[k])<1e-12
for r in new:
 assert r['candidate_updates']==800 and r['gmm_calls']==0 and r['pseudo_label_calls']==0
 assert r['executed_updates']+r['skipped_updates']==800 and r['executed_updates']>0
 assert r['target_backward_trace_exposures']>=r['executed_updates']*1
 assert r['query_inference_seconds']>0 and np.isfinite(r['query_latency_ms_per_trace'])
 assert r['peak_allocated_bytes']>0
# Exact source/target streams across all five methods per paired cell.
for seed in C['seeds']:
 for role in C['roles']:
  cells=[r for r in new if r['seed']==seed and r['role']==role]
  assert len({r['streams']['source'] for r in cells})==1 and len({r['streams']['target'] for r in cells})==1
# Reload one representative query batch from every new checkpoint.
for r in new:
 target=B.load_prepared(C['prepared'][r['role']]);
 with np.load(R/'artifacts'/f"split_{r['role']}.npz") as s:q=s['query_indices'][:128]
 model,_=B.load_model(r['seed']);ck=torch.load(R/'checkpoints'/f"{r['seed']}_{r['role']}_{r['condition']}.pt",map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'],strict=True);model.eval()
 z=B.predict(model,target['timestamps'][q],target['tam'][q],128)
 with np.load(R/'artifacts'/f"pred_{r['seed']}_{r['role']}_{r['condition']}.npz") as p:assert np.allclose(z,p['logits'][:128],atol=1e-5,rtol=1e-5)
 del model;torch.cuda.empty_cache()
B.save_json('audit.json',{'passed':True,'total_rows':36,'historical_rows':27,'new_tasks':9,'metrics_recomputed':36,'checkpoints':9,'checkpoint_reloads':9,'gmm_calls':0,'pseudo_label_calls':0,'cost_fields_verified':True,'paired_stream_cells':9})
print('audit passed')
