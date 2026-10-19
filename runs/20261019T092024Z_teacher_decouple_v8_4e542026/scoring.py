"""No sigmoid/GMM. Reference order-statistic cutoffs are heuristics, not risk bounds."""
import numpy as np
from runtime import RUN,ROOT,C,save,metrics


def read(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}


def rank(fit,x):return np.searchsorted(np.sort(fit),x,side='right')/max(len(fit),1)


def select(y,bp,cp,s):
    change=bp!=cp;correct=bp==y;wrong=~correct;trials=[]
    if change.any():
        for q in C['v7']['calibration_quantiles']:
            th=float(np.quantile(s[change],q));keep=change&(s>=th);d=int((keep&correct).sum());r=int((keep&wrong&(cp==y)).sum())
            trials.append(dict(quantile=q,threshold=th,changes=int(keep.sum()),damaged=d,recovered=r,nar=d/max(int(correct.sum()),1),feasible=bool(keep.sum()>=10 and d/max(int(correct.sum()),1)<=.01 and r>d)))
    ff=[r for r in trials if r['feasible']];best=max(ff,key=lambda r:(r['recovered'],-r['damaged'],r['threshold'])) if ff else None
    return dict(enabled=best is not None,threshold=best['threshold'] if best else None,trials=trials)


def changed(d):return d['base_logits'].argmax(-1)!=d['logits'].argmax(-1)


def compatible(d):
    cp=d['logits'].argmax(-1)
    return changed(d)&(d['margin_gain']>0)&(cp==d['nearest'])&(d['geometry_margin']>=d['frozen_geometry_margin'])


def fit_policy(valid,reference,policy):
    # Label-bearing valid enters ONLY select. Reference API physically excludes labels.
    assert 'labels' not in reference
    vm=changed(valid);rm=changed(reference);out=dict(policy=policy,reference_changed=int(rm.sum()),fitted_from='unlabeled_reference',enabled=False,threshold=None)
    if policy=='margin_ref_rank_source1':
        sv=rank(valid['margin_gain'][vm],valid['margin_gain']);cal=select(valid['labels'],valid['base_logits'].argmax(-1),valid['logits'].argmax(-1),sv)
        out.update(calibration=cal,enabled=bool(cal['enabled'] and rm.sum()>=32),threshold=cal['threshold'],fitted_from='source_valid_labels_and_reference_ECDF',reference_fit=reference['margin_gain'][rm].tolist(),source_fit=valid['margin_gain'][vm].tolist())
    else:
        mask=compatible(reference) if policy=='agreement_q95' else rm
        q=.99 if policy=='extreme_q99' else .95
        if mask.sum()>=32:out.update(enabled=True,threshold=float(np.quantile(reference['margin_gain'][mask],q,method='higher')))
        out.update(quantile=q,reference_compatible=int(mask.sum()),quantile_method='higher')
    return out


def apply(d,state,role):
    cp=d['logits'].argmax(-1);bp=d['base_logits'].argmax(-1);mask=cp!=bp
    if not state['enabled']:return bp.copy(),np.zeros(len(bp),bool)
    if state['policy']=='margin_ref_rank_source1':
        fit=state['source_fit'] if role=='valid' else state['reference_fit'];value=rank(np.asarray(fit),d['margin_gain']);mask&=value>=state['threshold']
    else:
        mask&=(d['margin_gain']>0)&(d['margin_gain']>=state['threshold'])
        if state['policy']=='agreement_q95':mask&=compatible(d)
    return np.where(mask,cp,bp),mask


def main():
    rows=[]
    for seed in C['seeds']:
        for name in C['slices']:
            for condition in ['frozen','v6_lora']+C['conditions']:
                ds={role:read(RUN/'artifacts'/f'{prefix}_{seed}_{name}_{condition}.npz') for role,prefix in [('valid','valid'),('reference','reference'),('query','query')]}
                for policy in C['v8']['score_policies']:
                    state=fit_policy(ds['valid'],ds['reference'],policy);save(f'fit_{seed}_{name}_{condition}_{policy}.json',state);saved={};counts={};mets={}
                    for role,d in ds.items():
                        pred,keep=apply(d,state,role);saved[role+'_predictions']=pred;counts[role+'_accepted']=int(keep.sum());counts[role+'_changed_pool']=int(changed(d).sum())
                        if role!='reference':mets[role]=metrics(d['labels'],pred,d['base_logits'].argmax(-1))
                    q=ds['query'];bp=q['base_logits'].argmax(-1);basef=metrics(q['labels'],bp,bp)['macro_f1'];row=dict(seed=seed,slice=name,condition=condition,policy=policy,**mets['query'],delta_f1_pp=100*(mets['query']['macro_f1']-basef),**{'source_valid_'+k:v for k,v in mets['valid'].items()},**counts)
                    rows.append(row);np.savez_compressed(RUN/'artifacts'/f'policy_{seed}_{name}_{condition}_{policy}.npz',**saved)
                    if condition=='v6_lora' and policy=='margin_ref_rank_source1':
                        with np.load(ROOT/C['v7_run']/'artifacts'/f'policy_{seed}_{name}_{policy}.npz') as old:assert np.array_equal(old['predictions'],saved['query_predictions'])
    save('policy_scores.json',rows)

if __name__=='__main__':main()
