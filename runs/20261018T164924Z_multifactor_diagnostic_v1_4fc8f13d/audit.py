"""Independent prediction reload, fixed scopes and paired metric recomputation."""
from contextlib import nullcontext
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from runtime import RUN,ROOT,C,setup,verify,sha,save,load_model,load_slice,predict,configure,metrics
from mechanisms import restore_aligner
from token_alignment import aligned_generator

def main():
    setup();verify();rows=json.loads((RUN/'artifacts/scores.json').read_text())
    expected={(s,n,c) for s in C['seeds'] for n in C['slices'] for c in C['conditions']}
    assert len(rows)==len(expected)==210 and {(r['seed'],r['slice'],r['condition']) for r in rows}==expected
    count=0
    for name in C['slices']:
        _,q=load_slice(name);y=q['labels'].numpy()
        for seed in C['seeds']:
            pure=load_model(seed);initial={k:v.detach().cpu().clone() for k,v in pure.state_dict().items()};del pure
            with np.load(RUN/'artifacts'/f'pred_{seed}_{name}_frozen.npz') as f:base=f['predictions'].copy()
            for condition in C['conditions']:
                r=next(r for r in rows if (r['seed'],r['slice'],r['condition'])==(seed,name,condition))
                key=f'{seed}_{name}_{condition}';pp=RUN/'artifacts'/f'pred_{key}.npz';cp=RUN/'checkpoints'/f'{key}.pt'
                assert sha(pp)==r['prediction_sha256'] and sha(cp)==r['checkpoint_sha256']
                ck=torch.load(cp,map_location='cpu',weights_only=True)
                assert ck['config_sha256']==sha(RUN/'config.json')
                m=load_model(seed);trainable=configure(m,condition);assert trainable==r['trainable_names']
                for k,v in ck['model'].items():
                    if k not in trainable:assert torch.equal(v,initial[k]),key+'/'+k
                m.load_state_dict(ck['model'],strict=True);a=restore_aligner(ck['aligner'],condition)
                with aligned_generator(m,a) if a is not None else nullcontext():z=predict(m,q)
                with np.load(pp) as f:
                    assert np.array_equal(f['rows'],q['rows'].numpy()) and np.array_equal(f['labels'],y)
                    assert np.allclose(z,f['logits'],rtol=1e-5,atol=1e-5),key
                    p=f['predictions'].copy();assert np.array_equal(p,z.argmax(1))
                assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-r['macro_f1'])<1e-12
                assert abs(accuracy_score(y,p)-r['accuracy'])<1e-12
                for k,v in metrics(y,p,base).items():assert v==r[k],key+'/'+k
                assert abs(r['delta_f1_pp']-100*(r['macro_f1']-metrics(y,base,base)['macro_f1']))<1e-12
                assert r['steps']==(0 if condition in ('frozen','token_adain') else 800)
                del m,a;count+=1
            print('audited',seed,name,flush=True)
    external=ROOT/C['external_run'];assert json.loads((external/'artifacts/postrun_audit.json').read_text())['passed']
    save('postrun_audit.json',{'passed':True,'prediction_cells_reloaded':count,'metrics_recomputed':count,'no_query_adaptation_or_selection':True,'diagnostic_only':True,'external_new_scoring':False})

if __name__=='__main__':main()
