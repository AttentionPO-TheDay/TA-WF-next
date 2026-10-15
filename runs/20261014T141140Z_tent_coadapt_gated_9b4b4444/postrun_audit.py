import json,numpy as np,torch
from sklearn.metrics import accuracy_score,f1_score
import port_base as B
import runner as Q
from coadapt import configure_model,collect_params
R,ROOT,C,DEVICE=B.R,B.ROOT,B.C,B.DEVICE
Q.setup();scores=json.loads((R/'artifacts/scores.json').read_text());hist=[x for x in scores if x['origin']=='historical_reference'];new=[x for x in scores if x['origin']=='new']
assert len(scores)==72 and len(hist)==63 and len(new)==9 and {x['condition'] for x in new}=={'tent_coadapt_gated'}
assert len(list((R/'checkpoints').glob('*.pt')))==9 and len(list((R/'artifacts').glob('update_*.json')))==9 and len(list((R/'artifacts').glob('gate_*.npz')))==9
for path,digest in json.loads((R/'artifacts/freeze.json').read_text()).items():assert B.sha(ROOT/path)==digest,path
for row in scores:
 with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz") as z:
  y=z['labels'];pred=z['predictions'];logits=z['logits'];rows=z['rows'];assert len(y)==row['rows'] and logits.shape==(len(y),102) and np.array_equal(pred,logits.argmax(1)) and np.isfinite(logits).all();assert abs(accuracy_score(y,pred)-row['accuracy'])<1e-12;assert abs(f1_score(y,pred,labels=np.arange(102),average='macro',zero_division=0)-row['macro_f1'])<1e-12
  with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_frozen.npz") as f:base=f['predictions'];assert np.array_equal(rows,f['rows']) and np.array_equal(y,f['labels'])
  correct=base==y;wrong=~correct;exp={'negative_adaptation_rate':float((correct&(pred!=y)).sum()/max(1,correct.sum())),'recovery_rate':float((wrong&(pred==y)).sum()/max(1,wrong.sum())),'changed_rate':float((base!=pred).mean())}
  for k,v in exp.items():assert abs(v-row[k])<1e-12
for row in new:
 assert row['candidate_updates']==800 and row['trainable_parameter_count']==C['adaptation']['trainable_parameter_count'] and row['source_forward_trace_exposures']==0 and row['source_backward_trace_exposures']==0 and row['source_samples_used_for_adaptation'] is False
 assert row['bn_forward_batches']==800 and all(v==800 for v in row['bn_num_batches_delta'].values()) and row['bn_momentum']==.1
 updates=json.loads((R/'artifacts'/f"update_{row['seed']}_{row['role']}_{row['condition']}.json").read_text());gate=np.load(R/'artifacts'/f"gate_{row['seed']}_{row['role']}_{row['condition']}.npz");assert len(updates)==800 and gate['selected'].shape==(800,128)
 assert row['target_backward_trace_exposures']==sum(x['retained_samples'] for x in updates)
 assert row['retained_sample_fraction']==row['target_backward_trace_exposures']/102400
for row in new:
 model,_=B.load_model(row['seed']);base={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};adapter=Q.InputAdapter(model);adapter=configure_model(adapter,.1);params,names=collect_params(adapter);assert len(params)==len(list(model.parameters())) and sum(p.numel() for p in params)==C['adaptation']['trainable_parameter_count'];ck=torch.load(R/'checkpoints'/f"{row['seed']}_{row['role']}_{row['condition']}.pt",map_location='cpu',weights_only=True);missing,unexpected=adapter.model.load_state_dict(ck['state_dict'],strict=True);assert not missing and not unexpected
 for name,module in model.named_modules():
  if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):assert int(ck['state_dict'][name+'.num_batches_tracked'])-int(base[name+'.num_batches_tracked'])==800
 adapter.model.eval();target=B.load_prepared(C['prepared'][row['role']])
 with np.load(R/'artifacts'/f"split_{row['role']}.npz") as split:q=split['query_indices'][:128]
 rel=B.predict(adapter.model,target['timestamps'][q],target['tam'][q],128)
 with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz") as p:assert np.allclose(rel,p['logits'][:128],atol=1e-5,rtol=1e-5)
 del model,adapter;torch.cuda.empty_cache()
B.save_json('audit.json',{'passed':True,'total_rows':72,'historical_rows':63,'new_tasks':9,'metrics_recomputed':72,'checkpoints':9,'checkpoint_reloads':9,'all_model_parameters_trainable':True,'source_adaptation_exposures':0,'bn_momentum':.1,'bn_batches_checked':9,'gate_masks_checked':9,'query_statistics':False})
print('audit passed')
