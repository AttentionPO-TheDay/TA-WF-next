"""Independent confusion matrix/cluster assignment, every model/data split reload."""
import os,time,json
import numpy as np
import torch
from sklearn.metrics import pairwise_distances_argmin
from runtime import RUN,ROOT,C,setup,verify,sha,save,load_model,load_slice
from methods import tokenize,evaluate,attach,assert_base_locked,standardize
from shared import inputs,canonical,old_adapter,signals,cluster,diagnostics,kl_gate
from scoring import read,fit_policy,apply


def mm(y,p,b):
    mat=np.zeros((102,102),dtype=np.int64);np.add.at(mat,(y,p),1);den=mat.sum(0)+mat.sum(1);f=np.divide(2*mat.diagonal(),den,out=np.zeros(102),where=den>0).mean();correct=b==y;wrong=~correct
    return dict(macro_f1=float(f),accuracy=float((p==y).mean()),nar=float((p[correct]!=y[correct]).mean()),recovery=float((p[wrong]==y[wrong]).mean()))


def main():
    setup();verify();seed=int(os.environ['CANDIDATE_SEED']);begin=time.monotonic();p,bb,prior,src=inputs(seed);vt=src['valid_tokens'].cuda();rows=json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text());records=[]
    for name in C['slices']:
        ref,query=load_slice(name);m=load_model(seed);pure=canonical(m);rt=tokenize(m,ref);re0,rz0=evaluate(m,rt);qt=tokenize(m,query);qe0,qz0=evaluate(m,qt);ve0=prior['valid_embeddings'].cuda();vz0=src['valid_logits'].cuda();base=read(RUN/'artifacts'/f'base_{seed}_{name}.npz');bp=base['predictions'];y=base['labels'];gate,_=kl_gate(rz0,re0,p,vz0)
        assert np.array_equal(base['logits'],qz0.cpu().numpy());gg=read(RUN/'artifacts'/f'kl_gate_{seed}_{name}.npz');assert np.array_equal(gg['eligible'],gate.cpu().numpy())
        for condition in ['frozen','v6_lora']+C['conditions']:
            row=next(r for r in rows if r['slice']==name and r['condition']==condition);m=load_model(seed);a=None
            if condition=='v6_lora':a=attach(m,'lora');a.load_state_dict(old_adapter(seed,name))
            elif condition in C['conditions']:
                ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';assert sha(ck)==row['checkpoint_sha256'];s=torch.load(ck,map_location='cpu',weights_only=True);assert s['config_sha256']==sha(RUN/'config.json');a=attach(m,'lora');a.load_state_dict(s['adapter']);mem=s['reference_state'];assert mem['steps']==400 and mem['features'].shape==(1024,128) and mem['probabilities'].shape==(1024,102)
                assert all(torch.isfinite(mem[k]).all() for k in ['source','prototype','features','probabilities','mean','second'])
                h=json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{condition}.json').read_text());assert len(h)==400 and all(np.isfinite(v['loss']) and np.isfinite(v['gradient_norm']) and v['weight_min']>=.05-1e-7 for v in h)
                if condition=='no_teacher_kl':assert all(v['active_kl']==0 and v['geometry_weight']==1 for v in h)
                if condition=='confident_teacher_kl':assert all(0<=v['kl_gate_fraction']<=1 and v['geometry_weight']==1 for v in h)
                if condition=='geometry_grad_balance':
                    ema=None
                    for v in h:
                        kn,gn=v['kl_gradient_norm'],v['geometry_gradient_norm'];ema=[kn,gn] if ema is None else [.9*ema[0]+.1*kn,.9*ema[1]+.1*gn];factor=max(1.,min(50.,ema[0]/(ema[1]+1e-8))) if gn>1e-10 else 1.;assert abs(factor-v['geometry_weight'])<1e-12
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
            records.append(dict(slice=name,condition=condition,reference_query_valid_exact=True,bn_source_locked=True,cluster_reference_fit_reproduced=True,cluster_query_assignment_independent=True,metrics_independent=True));print(f'audit {seed} {name} {condition} {len(records)}/10',flush=True)
            del m,a,es,ds;torch.cuda.empty_cache()
    save(f'audit_{seed}.json',dict(passed=True,seed=seed,models_reloaded=10,new_checkpoints_reloaded=6,all_three_splits=True,seconds=time.monotonic()-begin,records=records))


def policy_audit():
    rows=json.loads((RUN/'artifacts/policy_scores.json').read_text());records=[]
    for seed in C['seeds']:
        for name in C['slices']:
            for condition in ['frozen','v6_lora']+C['conditions']:
                ds={k:read(RUN/'artifacts'/f'{k}_{seed}_{name}_{condition}.npz') for k in ['reference','valid','query']}
                for policy in C['v8']['score_policies']:
                    path=RUN/'artifacts'/f'fit_{seed}_{name}_{condition}_{policy}.json';state=json.loads(path.read_text());again=fit_policy(ds['valid'],ds['reference'],policy);assert state==again
                    if state['enabled'] and policy!='margin_ref_rank_source1':
                        r=ds['reference'];rp=r['logits'].argmax(-1);rb=r['base_logits'].argmax(-1);mask=rp!=rb
                        if policy=='agreement_q95':mask&=(r['margin_gain']>0)&(rp==r['nearest'])&(r['geometry_margin']>=r['frozen_geometry_margin'])
                        fit=np.sort(r['margin_gain'][mask]);q=.99 if policy=='extreme_q99' else .95;expected=float(fit[int(np.ceil(q*(len(fit)-1)))]);assert expected==state['threshold']
                    saved=read(RUN/'artifacts'/f'policy_{seed}_{name}_{condition}_{policy}.npz')
                    for role,d in ds.items():
                        pred,keep=apply(d,state,role);assert np.array_equal(pred,saved[role+'_predictions'])
                        # Independent policy reconstruction; fitting values never derive from query.
                        bp=d['base_logits'].argmax(-1);cp=d['logits'].argmax(-1);accept=cp!=bp
                        if not state['enabled']:accept[:]=False
                        elif policy=='margin_ref_rank_source1':
                            pool=state['source_fit'] if role=='valid' else state['reference_fit'];ranks=np.array([sum(x>=s for s in pool)/max(len(pool),1) for x in d['margin_gain']]);accept&=ranks>=state['threshold']
                        else:
                            accept&=(d['margin_gain']>0)&(d['margin_gain']>=state['threshold'])
                            if policy=='agreement_q95':accept&=(cp==d['nearest'])&(d['geometry_margin']>=d['frozen_geometry_margin'])
                        assert np.array_equal(np.where(accept,cp,bp),pred)
                        if role!='reference':
                            met=mm(d['labels'],pred,bp);row=next(v for v in rows if v['seed']==seed and v['slice']==name and v['condition']==condition and v['policy']==policy)
                            for k,v in met.items():assert abs(v-row[('source_valid_' if role=='valid' else '')+k])<1e-12
                    records.append(dict(seed=seed,slice=name,condition=condition,policy=policy,fit_reproduced=True,predictions_independent=True,metrics_independent=True))
    save('policy_audit.json',dict(passed=True,records=records))

if __name__=='__main__':main()
