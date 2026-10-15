"""Independent stored-prediction metrics, paired streams, reload and freeze checks."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
import port_base as B
from runner import setup, CONDITIONS

R,ROOT,C=B.R,B.ROOT,B.C
setup()
scores=json.loads((R/'artifacts/scores.json').read_text())
assert len(scores)==90
new=[r for r in scores if r['origin']=='new']
assert len(new)==54
expected={(s,d,c) for s in C['seeds'] for d in C['roles'] for c in CONDITIONS}
assert {(r['seed'],r['role'],r['condition']) for r in new}==expected
assert len(list((R/'checkpoints').glob('*.pt')))==54
assert len(list((R/'artifacts').glob('gmm_*.npz')))==1080
for p,value in json.loads((R/'artifacts/freeze.json').read_text()).items():
    assert B.sha(ROOT/p)==value, p
for role in C['roles']:
    with np.load(R/'artifacts'/f'split_{role}.npz') as split:
        assert len(split['support'])==1024
        assert set(split['support']).isdisjoint(split['query'])
for seed in C['seeds']:
    for role in C['roles']:
        cells=[r for r in new if (r['seed'],r['role'])==(seed,role)]
        assert len({r['streams']['source'] for r in cells})==1
        assert len({r['streams']['target'] for r in cells})==1
for r in scores:
    path=R/'artifacts'/f"pred_{r['seed']}_{r['role']}_{r['condition']}.npz"
    with np.load(path) as p, np.load(R/'artifacts'/f"pred_{r['seed']}_{r['role']}_frozen.npz") as frozen:
        y,pred=p['labels'],p['predictions']; base=frozen['predictions']
        assert np.array_equal(p['rows'],frozen['rows']) and np.array_equal(y,frozen['labels'])
        assert len(y)==r['rows'] and p['logits'].shape==(len(y),102)
        assert np.isfinite(p['logits']).all() and np.array_equal(pred,p['logits'].argmax(1))
        assert abs(accuracy_score(y,pred)-r['accuracy'])<1e-12
        assert abs(f1_score(y,pred,labels=np.arange(102),average='macro',zero_division=0)-r['macro_f1'])<1e-12
        bc=base==y; bw=~bc
        assert abs(((bc)&(pred!=y)).sum()/max(1,bc.sum())-r['negative_adaptation_rate'])<1e-12
        assert abs(((bw)&(pred==y)).sum()/max(1,bw.sum())-r['recovery_rate'])<1e-12
        assert abs((base!=pred).mean()-r['changed_rate'])<1e-12
    if r['origin']=='new':
        assert r['updates']==800 and len(r['loss_history'])==100 and len(r['gmm_history'])==20
        assert r['changed_parameter_tensors']>0 and r['relative_parameter_change']>0
        assert all(np.isfinite([h[k] for k in C['loss_order']]).all() for h in r['loss_history'])
        for h in r['loss_history']:
            assert abs(sum(w*h[k] for w,k in zip(C['loss_weights'][r['condition']],C['loss_order']))-h['weighted_total'])<1e-9
        for epoch in range(0,100,5):
            with np.load(R/'artifacts'/f"gmm_{r['seed']}_{r['role']}_{r['condition']}_{epoch}.npz") as g:
                assert len(g['rows'])==1024 and g['selected'].sum()>=128
                assert np.array_equal(g['selected'],g['probabilities']>=.6)

# Reload a representative batch for every model, same batch size as scoring.
for role in C['roles']:
    target=B.load_prepared(C['prepared'][role])
    with np.load(R/'artifacts'/f'split_{role}.npz') as split: q=split['query_indices'][:128]
    for row in [r for r in new if r['role']==role]:
        model,_=B.load_model(row['seed'])
        checkpoint=torch.load(R/'checkpoints'/f"{row['seed']}_{role}_{row['condition']}.pt",map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'],strict=True); model.eval()
        logits=B.predict(model,target['timestamps'][q],target['tam'][q],128)
        with np.load(R/'artifacts'/f"pred_{row['seed']}_{role}_{row['condition']}.npz") as p:
            assert np.allclose(logits,p['logits'][:128],atol=1e-5,rtol=1e-5)
            assert np.array_equal(logits.argmax(1),p['predictions'][:128])
        del model; torch.cuda.empty_cache()
B.save_json('audit.json',{'passed':True,'new_tasks':54,'history_rows':36,'metrics_recomputed':90,
    'paired_stream_cells':9,'checkpoint_reloads':54,'gmm_files':1080,'frozen_hashes_verified':True})
