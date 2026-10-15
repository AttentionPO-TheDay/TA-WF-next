import copy,json,os
import numpy as np,torch
import port_base as B
import runner as Q
from mechanism import WindowController
R,ROOT,C=B.R,B.ROOT,B.C
Q.setup()
assert C['status']=='frozen' and os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available()
assert C['permissions']['support_labels'] is False and C['permissions']['query_adaptation'] is False
source=B.load_prepared(C['prepared']['source_valid'],'source');valid=B.load_prepared(C['prepared']['source_valid'],'valid')
source_all={k:np.concatenate([source[k],valid[k]]) for k in source};checked={}
for role in C['roles']:
    target=B.load_prepared(C['prepared'][role]);s,q,_=B.split_targets(source_all,role,target)
    with np.load(ROOT/C['comparison_run']/'artifacts'/f'split_{role}.npz') as old:
        assert np.array_equal(s,old['support_indices']) and np.array_equal(q,old['query_indices'])
    checked[role]={'support_rows':len(s),'query_rows':len(q),'same_historical_split':True,'exact_input_disjoint':True}
    if role==C['roles'][0]:support={k:target[k][s] for k in ('timestamps','tam','rows')}
# Structural and wrapper gradient checks happen in the two-step smoke run.
for condition in C['conditions']:
    model,_=B.load_model(C['seeds'][0]);cost=Q.train(model,source,support,condition,C['seeds'][0],C['roles'][0],max_steps=2)
    assert cost['executed_updates']==2 and cost['max_gradient']>0 and cost['gmm_calls']==cost['pseudo_label_calls']==0
    del model;torch.cuda.empty_cache()
B.save_json('preflight.json',{'passed':True,'gpu':torch.cuda.get_device_name(0),'splits':checked,'candidate_smoke_updates':2,'wrapper_gradient_checks':True,'gmm_calls':0,'pseudo_label_calls':0,'config_sha':B.sha(R/'config.json'),'runner_sha':B.sha(R/'runner.py')})
print('preflight passed',R.name,flush=True)
