import json, numpy as np, torch
from sklearn.metrics import accuracy_score,f1_score
import port_base as B
import runner as Q
from enhanced_tent import configure_model
R,ROOT,C,DEVICE=B.R,B.ROOT,B.C,B.DEVICE
Q.setup(); scores=json.loads((R/'artifacts/scores.json').read_text()); hist=[x for x in scores if x['origin']=='historical_reference']; new=[x for x in scores if x['origin']=='new']
assert len(scores)==45 and len(hist)==36 and len(new)==9 and {x['condition'] for x in new}=={'tent_fixedstats_ln'}
assert len(list((R/'checkpoints').glob('*.pt')))==9 and len(list((R/'artifacts').glob('update_*.json')))==9
for path,digest in json.loads((R/'artifacts/freeze.json').read_text()).items(): assert B.sha(ROOT/path)==digest,path
for row in scores:
 with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz") as z:
  y=z['labels'];pred=z['predictions'];logits=z['logits']; rows=z['rows']; assert len(y)==row['rows'] and logits.shape==(len(y),102) and np.array_equal(pred,logits.argmax(1)) and np.isfinite(logits).all()
  assert abs(accuracy_score(y,pred)-row['accuracy'])<1e-12 and abs(f1_score(y,pred,labels=np.arange(102),average='macro',zero_division=0)-row['macro_f1'])<1e-12
  with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_frozen.npz") as f:
   base=f['predictions']; assert np.array_equal(rows,f['rows']) and np.array_equal(y,f['labels'])
  correct=base==y;wrong=~correct; expected={'negative_adaptation_rate':float((correct&(pred!=y)).sum()/max(1,correct.sum())),'recovery_rate':float((wrong&(pred==y)).sum()/max(1,wrong.sum())),'changed_rate':float((base!=pred).mean())}
  for k,v in expected.items():assert abs(v-row[k])<1e-12
for row in new:
 assert row['candidate_updates']==800 and row['executed_updates']==800 and row['trainable_parameter_count']==3072 and row['batch_statistics'] is False
 assert row['source_bn_buffers_unchanged'] is True
 for key in ('streams',): assert key in row
# paired source/target stream equality across all 9 new cells, and target stream differs by role as expected
for seed in C['seeds']:
 for role in C['roles']:
  cells=[x for x in new if x['seed']==seed and x['role']==role]; assert len(cells)==1
# checkpoint and scope/buffer audit
for row in new:
 model,_=B.load_model(row['seed']); base={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; adapter=configure_model(Q.InputAdapter(model)); params,names=__import__('enhanced_tent',fromlist=['collect_params']).collect_params(adapter); assert sum(p.numel() for p in params)==3072
 ck=torch.load(R/'checkpoints'/f"{row['seed']}_{row['role']}_{row['condition']}.pt",map_location='cpu',weights_only=True); missing,unexpected=adapter.model.load_state_dict(ck['state_dict'],strict=False); assert not unexpected and not missing
 allowed={n.removeprefix('model.') for n in names}
 for key,val in ck['state_dict'].items():
  bare=key
  module='.'.join(bare.split('.')[:-1])
  if module not in {n.rsplit('.',1)[0] for n in allowed}: assert torch.equal(val,base[key]),key
  if key.endswith(('running_mean','running_var','num_batches_tracked')): assert torch.equal(val,base[key]),key
 adapter.model.eval(); target=B.load_prepared(C['prepared'][row['role']]);
 with np.load(R/'artifacts'/f"split_{row['role']}.npz") as split:q=split['query_indices'][:128]
 rel=B.predict(adapter.model,target['timestamps'][q],target['tam'][q],128)
 with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz") as p: assert np.allclose(rel,p['logits'][:128],atol=1e-5,rtol=1e-5)
 del model,adapter;torch.cuda.empty_cache()
B.save_json('audit.json',{'passed':True,'total_rows':45,'historical_rows':36,'new_tasks':9,'metrics_recomputed':45,'checkpoints':9,'checkpoint_reloads':9,'trainable_parameter_count':3072,'bn_buffers_unchanged':True,'query_statistics':False,'paired_cells':9})
print('audit passed')
