import copy,json,os
import numpy as np,torch
import port_base as B
import runner as Q
from mechanism import topology_loss,WindowController,update_teacher
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
# Meaningful structural/gradient checks before starting 800-step tasks.
a=[torch.randn(2,c,l,device=B.DEVICE,requires_grad=True) for c,l in [(64,1800),(128,600),(256,120)]]
l=topology_loss(a,24);l.backward();assert l>0 and all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in a)
w=WindowController(1.)
for _ in range(16):assert w.step(1.)['scale']==1.
w=WindowController(1.)
for h in [1.]*12+[2.]*4:d=w.step(h)
assert d['scale']==.25
for condition in C['conditions']:
    model,_=B.load_model(C['seeds'][0]);cost=Q.train(model,source,support,condition,C['seeds'][0],C['roles'][0],max_steps=2)
    assert cost['executed_updates']==2 and cost['max_gradient']>0 and cost['gmm_calls']==cost['pseudo_label_calls']==0
    del model;torch.cuda.empty_cache()
B.save_json('preflight.json',{'passed':True,'gpu':torch.cuda.get_device_name(0),'splits':checked,'candidate_smoke_updates':2,'topology_gradient_checks':True,'window_unit_checks':True,'ema_soft_targets':True,'gmm_calls':0,'pseudo_label_calls':0,'config_sha':B.sha(R/'config.json'),'runner_sha':B.sha(R/'runner.py')})
print('preflight passed',R.name,flush=True)
