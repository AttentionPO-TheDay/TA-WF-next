"""Independent audit for the matched-budget rollback experiment."""
import json, numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score
import port_base as B
from runner import setup, CONDITIONS
R,ROOT,C=B.R,B.ROOT,B.C; setup()
scores=json.loads((R/'artifacts/scores.json').read_text()); assert len(scores)==45
new=[r for r in scores if r['origin']=='new']; old=[r for r in scores if r['origin']=='historical_reference']
assert len(new)==36 and len(old)==9
expected={(s,d,c) for s in C['seeds'] for d in C['roles'] for c in CONDITIONS}
assert {(r['seed'],r['role'],r['condition']) for r in new}==expected
assert {(r['seed'],r['role']) for r in old}=={(s,d) for s in C['seeds'] for d in C['roles']}
assert len(list((R/'checkpoints').glob('*.pt')))==36
assert len(list((R/'artifacts').glob('gmm_*.npz')))==720
for p,v in json.loads((R/'artifacts/freeze.json').read_text()).items(): assert B.sha(ROOT/p)==v,p
for role in C['roles']:
    with np.load(R/'artifacts'/f'split_{role}.npz') as q:
        assert len(q['support'])==1024 and set(q['support']).isdisjoint(q['query'])
for s in C['seeds']:
  for d in C['roles']:
    cells=[r for r in new if (r['seed'],r['role'])==(s,d)]
    assert len({r['streams']['source'] for r in cells})==1 and len({r['streams']['target'] for r in cells})==1
for r in scores:
    with np.load(R/'artifacts'/f"pred_{r['seed']}_{r['role']}_{r['condition']}.npz") as p, np.load(R/'artifacts'/f"pred_{r['seed']}_{r['role']}_frozen.npz") as b:
        y,pred=p['labels'],p['predictions']; base=b['predictions']; assert np.array_equal(p['rows'],b['rows']) and np.array_equal(y,b['labels'])
        assert p['logits'].shape==(len(y),102) and np.isfinite(p['logits']).all() and np.array_equal(pred,p['logits'].argmax(1))
        assert abs(accuracy_score(y,pred)-r['accuracy'])<1e-12
        assert abs(f1_score(y,pred,labels=np.arange(102),average='macro',zero_division=0)-r['macro_f1'])<1e-12
        bc=base==y; bw=~bc
        assert abs((bc&(pred!=y)).sum()/max(1,bc.sum())-r['negative_adaptation_rate'])<1e-12
        assert abs((bw&(pred==y)).sum()/max(1,bw.sum())-r['recovery_rate'])<1e-12
        assert abs((base!=pred).mean()-r['changed_rate'])<1e-12
    if r['origin']=='new':
        assert r['updates']==800 and len(r['loss_history'])==100 and len(r['gmm_history'])==20
        assert len(r['acceptance_history'])==800//C['rollback_intervals'][r['condition']] if C['rollback_intervals'][r['condition']] else len(r['acceptance_history'])==0
        assert len(r['streams'])==3 and r['changed_parameter_tensors']>0
        for h in r['loss_history']: assert np.isfinite([h[k] for k in C['loss_order']]).all()
        for e in range(0,100,5):
            with np.load(R/'artifacts'/f"gmm_{r['seed']}_{r['role']}_{r['condition']}_{e}.npz") as g:
                assert len(g['rows'])==1024 and int(g['selected'].sum())>=128 and np.array_equal(g['selected'],g['probabilities']>=.6)
for role in C['roles']:
  t=B.load_prepared(C['prepared'][role]); q=np.load(R/'artifacts'/f'split_{role}.npz')['query_indices'][:128]
  for r in [x for x in new if x['role']==role]:
    m,_=B.load_model(r['seed']); ck=torch.load(R/'checkpoints'/f"{r['seed']}_{role}_{r['condition']}.pt",map_location='cpu',weights_only=True); m.load_state_dict(ck['state_dict'],strict=True); m.eval(); z=B.predict(m,t['timestamps'][q],t['tam'][q],128)
    with np.load(R/'artifacts'/f"pred_{r['seed']}_{role}_{r['condition']}.npz") as p: assert np.allclose(z,p['logits'][:128],atol=1e-5,rtol=1e-5) and np.array_equal(z.argmax(1),p['predictions'][:128])
    del m; torch.cuda.empty_cache()
B.save_json('audit.json',{'passed':True,'new_tasks':36,'historical_rows':9,'metrics_recomputed':45,'paired_stream_cells':9,'checkpoint_reloads':36,'gmm_files':720,'frozen_hashes_verified':True})
print('audit passed')
