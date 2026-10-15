import json, os, numpy as np, torch
import port_base as B
import runner as Q
R,ROOT,C,DEVICE=B.R,B.ROOT,B.C,B.DEVICE
Q.setup(); assert C['status']=='frozen'; assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available()
assert C['permissions']['support_labels'] is False and not C['permissions']['query_adaptation'] and not C['permissions']['query_forward_batch_statistics']
source=B.load_prepared(C['prepared']['source_valid'],'source'); valid=B.load_prepared(C['prepared']['source_valid'],'valid'); source_all={k:np.concatenate([source[k],valid[k]]) for k in source}
checked={}
for role in C['roles']:
 target=B.load_prepared(C['prepared'][role]); s,q,_=B.split_targets(source_all,role,target)
 with np.load(ROOT/C['comparison_run']/ 'artifacts'/f'split_{role}.npz') as old: assert np.array_equal(s,old['support_indices']) and np.array_equal(q,old['query_indices'])
 checked[role]={'support_rows':int(len(s)),'query_rows':int(len(q)),'same_historical_split':True}
historical_rows=0
for alias,path in C['historical_references'].items():
 old=json.loads((ROOT/path).read_text()); assert sum(x['condition']==alias for x in old)==9,alias; historical_rows+=9
assert historical_rows==C['budget']['historical_rows']
role=C['roles'][0]; target=B.load_prepared(C['prepared'][role]); s,_,_=B.split_targets(source_all,role,target); support={k:target[k][s] for k in ('timestamps','tam','rows')}
model,_=B.load_model(C['seeds'][0]); cost=Q.train_condition(model,support,C['conditions'][0],C['seeds'][0],role,max_steps=2)
assert cost['candidate_updates']==2 and cost['trainable_parameter_count']==C['adaptation']['trainable_parameter_count']
assert cost['source_forward_trace_exposures']==0 and cost['source_backward_trace_exposures']==0
assert cost['bn_forward_batches']==2 and all(v==2 for v in cost['bn_num_batches_delta'].values())
assert 0 < cost['target_backward_trace_exposures'] <= 256
assert len(cost['trainable_parameter_names'])==len(list(model.parameters()))
B.save_json('preflight.json',{'passed':True,'gpu':torch.cuda.get_device_name(0),'splits':checked,'historical_rows':historical_rows,'candidate_smoke_updates':2,'trainable_parameter_count':cost['trainable_parameter_count'],'all_model_parameters_trainable':True,'source_forward_trace_exposures':0,'bn_momentum':C['adaptation']['bn_momentum'],'bn_smoke_deltas':cost['bn_num_batches_delta'],'config_sha':B.sha(R/'config.json'),'code_sha':{p.name:B.sha(p) for p in [R/'runner.py',R/'coadapt.py',R/'port_base.py',R/'model.py',R/'reporting.py',R/'postrun_audit.py']}})
print('preflight passed',R.name,flush=True)
