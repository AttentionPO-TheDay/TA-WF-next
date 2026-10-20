"""Independent confusion matrix/cluster assignment, every model/data split reload."""
import os,time,json
import numpy as np
import torch
from sklearn.metrics import pairwise_distances_argmin
from runtime import RUN,ROOT,C,setup,verify,sha,save,load_model,load_slice
from methods import tokenize,evaluate,attach,assert_base_locked,standardize
from shared import inputs,canonical,old_adapter,signals,cluster,diagnostics,kl_gate
from scoring import read
from progressive import schedule
from view import probs


def mm(y,p,b):
    mat=np.zeros((102,102),dtype=np.int64);np.add.at(mat,(y,p),1);den=mat.sum(0)+mat.sum(1);f=np.divide(2*mat.diagonal(),den,out=np.zeros(102),where=den>0).mean();correct=b==y;wrong=~correct
    return dict(macro_f1=float(f),accuracy=float((p==y).mean()),nar=float((p[correct]!=y[correct]).mean()),recovery=float((p[wrong]==y[wrong]).mean()))


def main():
    setup();verify();seed=int(os.environ['CANDIDATE_SEED']);begin=time.monotonic();p,bb,prior,src=inputs(seed);vt=src['valid_tokens'].cuda();rows=json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text());records=[];view=torch.load(RUN/'artifacts/source_view.pt',map_location='cpu',weights_only=True)
    for name in C['slices']:
        ref,query=load_slice(name);m=load_model(seed);pure=canonical(m);rt=tokenize(m,ref);re0,rz0=evaluate(m,rt);qt=tokenize(m,query);qe0,qz0=evaluate(m,qt);ve0=prior['valid_embeddings'].cuda();vz0=src['valid_logits'].cuda();base=read(RUN/'artifacts'/f'base_{seed}_{name}.npz');bp=base['predictions'];y=base['labels'];gate,_=kl_gate(rz0,re0,p,vz0)
        viewfile=read(RUN/'artifacts'/f'view_{seed}_{name}.npz');assert np.array_equal(viewfile['reference_q'],probs(ref['tam'],view).numpy()) and np.array_equal(viewfile['query_predictions'],probs(query['tam'],view).argmax(-1).numpy())
        assert np.array_equal(base['logits'],qz0.cpu().numpy());gg=read(RUN/'artifacts'/f'kl_gate_{seed}_{name}.npz');assert np.array_equal(gg['eligible'],gate.cpu().numpy())
        for condition in ['frozen','v6_lora','v8_balance']+C['conditions']:
            row=next(r for r in rows if r['slice']==name and r['condition']==condition);m=load_model(seed);a=None
            if condition=='v6_lora':a=attach(m,'lora');a.load_state_dict(old_adapter(seed,name))
            elif condition=='v8_balance':
                a=attach(m,'lora');a.load_state_dict(torch.load(ROOT/C['v8_run']/'checkpoints'/f'{seed}_{name}_geometry_grad_balance.pt',map_location='cpu',weights_only=True)['adapter'])
            elif condition in C['conditions']:
                ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';assert sha(ck)==row['checkpoint_sha256'];s=torch.load(ck,map_location='cpu',weights_only=True);assert s['config_sha256']==sha(RUN/'config.json');a=attach(m,'lora');a.load_state_dict(s['adapter']);mem=s['reference_state'];assert mem['steps']==400 and mem['features'].shape==(1024,128) and mem['probabilities'].shape==(1024,102)
                assert all(torch.isfinite(mem[k]).all() for k in ['source','prototype','features','probabilities','mean','second'])
                h=json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{condition}.json').read_text());assert len(h)==400 and all(np.isfinite(v['loss']) and np.isfinite(v['gradient_norm']) and v['weight_min']>=.05-1e-7 for v in h)
                ema=None;ell=0.
                for i,v in enumerate(h,1):
                    weight,ema,ell,ratio=schedule(v['kl_gradient_norm'],v['geometry_gradient_norm'],i,ema,ell,condition!='ramp_balance');assert abs(weight-v['geometry_weight'])<1e-10 and abs(ell-v['ell'])<1e-12
                    if condition=='smooth_shuffled_view':assert abs(v['aux_weighted_mass']-v['unshuffled_aux_mass'])<1e-4
                    assert v['aux_enabled']==bool(view['enabled'] and condition in ['smooth_view','smooth_shuffled_view'])
                    for k in ['gradient_previous_cosine','update_previous_cosine','kl_geometry_cosine','kl_aux_cosine']:
                        assert v[k] is None or -1.00001<=v[k]<=1.00001
                gp=json.loads((RUN/'artifacts'/f'gradients_{seed}_{name}_{condition}.json').read_text());assert [v['step'] for v in gp]==[1,200,400]
            ds={};es={}
            for prefix,t,be,bz,ids in [('reference',rt,re0,rz0,ref['rows']),('query',qt,qe0,qz0,query['rows']),('valid',vt,ve0,vz0,src['valid_rows'])]:
                e,z=evaluate(m,t,a);d=read(RUN/'artifacts'/f'{prefix}_{seed}_{name}_{condition}.npz');assert np.array_equal(d['logits'],z.cpu().numpy()) and np.array_equal(d['embeddings'],e.cpu().numpy()) and np.array_equal(d['rows'],ids.numpy());assert np.isfinite(d['logits']).all();assert np.array_equal(d['base_logits'],bz.cpu().numpy())
                ss=signals(bz,z,be,e,p)
                for k,v in ss.items():assert np.array_equal(v,d[k])
                ds[prefix]=d;es[prefix]=e
            assert 'labels' not in ds['reference'];assert_base_locked(m,pure)
            d=ds['query'];met=mm(y,d['logits'].argmax(-1),bp);vm=mm(ds['valid']['labels'],ds['valid']['logits'].argmax(-1),ds['valid']['base_logits'].argmax(-1))
            for k in met:assert abs(met[k]-row[k])<1e-12
            for k in vm:assert abs(vm[k]-row['source_valid_'+k])<1e-12
            assert sha(RUN/'artifacts'/f'query_{seed}_{name}_{condition}.npz')==row['query_sha256']
            cl=read(RUN/'artifacts'/f'cluster_{seed}_{name}_{condition}.npz');qstd=standardize(es['query'],p).cpu().numpy().astype('float64');assign=pairwise_distances_argmin(qstd,cl['centers']);assert np.array_equal(assign,cl['query_clusters'])
            centers,expected=cluster(es['reference'],es['query'],p);assert np.array_equal(expected,assign) and np.allclose(centers,cl['centers'],atol=1e-12,rtol=1e-12)
            dg=diagnostics(y,bp,d['logits'].argmax(-1),assign,d['nearest'],base['common_wrong'])
            for k,v in dg.items():assert abs(v-row[k])<1e-12
            records.append(dict(slice=name,condition=condition,reference_query_valid_exact=True,bn_source_locked=True,cluster_reference_fit_reproduced=True,cluster_query_assignment_independent=True,metrics_independent=True));print(f'audit {seed} {name} {condition} {len(records)}/14',flush=True)
            del m,a,es,ds;torch.cuda.empty_cache()
    save(f'audit_{seed}.json',dict(passed=True,seed=seed,models_reloaded=14,new_checkpoints_reloaded=8,all_three_splits=True,seconds=time.monotonic()-begin,records=records))

if __name__=='__main__':main()
