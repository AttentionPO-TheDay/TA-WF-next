"""Reference-only normalization/GMM; source-only threshold fitting; terminal metrics."""
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from runtime import RUN,ROOT,C,save,metrics

KEYS=['combined','normalized_margin_gain','candidate_geometric_margin','negative_frozen_maha','negative_frozen_entropy']

def read(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}

def rank(fit,x):return np.searchsorted(np.sort(fit),x,side='right')/max(len(fit),1)

def scaled(sig,mask,fit):
    mg=sig['normalized_margin_gain'];v=fit['normalized_margin_gain'];pool=v[mask]
    if len(pool)==0:return {k:np.zeros(len(mg)) for k in ('z','rank','fusion')}
    return dict(z=(mg-pool.mean())/max(pool.std(),1e-6),rank=rank(pool,mg),fusion=np.mean([rank(fit[k][mask],sig[k]) for k in ['normalized_margin_gain','candidate_geometric_margin','negative_frozen_maha']],axis=0))

def select(y,bp,cp,s):
    changed=cp!=bp;correct=bp==y;wrong=~correct;trials=[]
    if changed.any():
        for q in C['v7']['calibration_quantiles']:
            th=float(np.quantile(s[changed],q));keep=changed&(s>=th);d=int((keep&correct).sum());r=int((keep&wrong&(cp==y)).sum())
            trials.append(dict(quantile=q,threshold=th,changes=int(keep.sum()),damaged=d,recovered=r,nar=d/max(int(correct.sum()),1),feasible=bool(keep.sum()>=10 and d/max(int(correct.sum()),1)<=.01 and r>d)))
    feasible=[r for r in trials if r['feasible']];best=max(feasible,key=lambda r:(r['recovered'],-r['damaged'],r['threshold'])) if feasible else None
    return dict(enabled=best is not None,threshold=best['threshold'] if best else None,trials=trials)


def policies(v,q,r,vs,qs,rs):
    vb=v['base_logits'].argmax(-1);vc=v['candidate_logits'].argmax(-1);qb=q['base_logits'].argmax(-1);qc=q['candidate_logits'].argmax(-1);rb=r['base_logits'].argmax(-1);rc=r['candidate_logits'].argmax(-1)
    vm=vb!=vc;rm=rb!=rc;vloc=scaled(vs,vm,vs);qloc=scaled(qs,rm,rs);local_ok=int(rm.sum())>=32
    maps={'combined_source1':(vs['combined'],qs['combined']),'margin_source1':(vs['normalized_margin_gain'],qs['normalized_margin_gain']),
          'margin_ref_z_source1':(vloc['z'],qloc['z']),'margin_ref_rank_source1':(vloc['rank'],qloc['rank']),'rank_fusion_source1':(vloc['fusion'],qloc['fusion'])}
    result={};notes={}
    for k,(sv,sq) in maps.items():
        calibration=select(v['labels'],vb,vc,sv);enabled=calibration['enabled'] and (local_ok or k in ['combined_source1','margin_source1'])
        keep=(qc!=qb)&(sq>=calibration['threshold']) if enabled else np.zeros(len(qb),bool)
        result[k]=np.where(keep,qc,qb);notes[k]=dict(calibration=calibration,reference_changed=int(rm.sum()),local_enabled=local_ok,changes=int(keep.sum()))
    keep=(qc!=qb)&(qloc['rank']>=.95) if local_ok else np.zeros(len(qb),bool);result['ref_rank95']=np.where(keep,qc,qb);notes['ref_rank95']=dict(local_enabled=local_ok,changes=int(keep.sum()))
    keep=np.zeros(len(qb),bool);gn=dict(enabled=False,reference_changed=int(rm.sum()),posterior_not_safety_probability=True)
    if local_ok:
        fit=rs['normalized_margin_gain'][rm,None];args=dict(n_init=3,reg_covar=.001,random_state=20261019)
        one=GaussianMixture(n_components=1,**args).fit(fit);two=GaussianMixture(n_components=2,**args).fit(fit);enable=two.bic(fit)<one.bic(fit);top=int(two.means_[:,0].argmax())
        post=two.predict_proba(qs['normalized_margin_gain'][:,None])[:,top];keep=(qc!=qb)&(post>=.95)&(qs['normalized_margin_gain']>0) if enable else keep
        gn.update(enabled=bool(enable),bic1=float(one.bic(fit)),bic2=float(two.bic(fit)),means=two.means_[:,0].tolist(),weights=two.weights_.tolist(),high_component=top,changes=int(keep.sum()))
    result['ref_gmm2_tail']=np.where(keep,qc,qb);notes['ref_gmm2_tail']=gn
    return result,notes


def correlation(sig,mask,y=None,base=None,candidate=None):
    x=np.stack([sig[k][mask] for k in KEYS],1);valid=len(x)>2 and (x.std(0)>0).all()
    out=dict(n=len(x),keys=KEYS,pearson=np.corrcoef(x.T).tolist() if valid else None,spearman=spearmanr(x).statistic.tolist() if valid else None,components={})
    for i,k in enumerate(KEYS):
        a=x[:,i];d=dict(min=float(a.min()),max=float(a.max()),mean=float(a.mean()),std=float(a.std()),unique=int(len(np.unique(a)))) if len(a) else dict(n=0)
        if y is not None:
            yy=(candidate[mask]==y[mask]).astype(int)
            d['correction_auroc']=float(roc_auc_score(yy,a)) if len(np.unique(yy))==2 else None
        out['components'][k]=d
    return out


def main():
    rows=[]
    for seed in C['seeds']:
        import torch
        source=torch.load(ROOT/C['v4_run']/'artifacts'/f'source_{seed}.pt',map_location='cpu',weights_only=True)
        for name in C['slices']:
            old=ROOT/C['v6_run']/'artifacts';base=read(RUN/'artifacts'/f'base_{seed}_{name}.npz');raw=read(old/f'raw_{seed}_{name}_sharp_target_only.npz');vv=read(old/f'valid_{seed}_{name}_sharp_target_only.npz');r=read(RUN/'artifacts'/f'reference_signal_{seed}_{name}.npz')
            q=dict(base_logits=base['logits'],candidate_logits=raw['logits'],labels=base['labels']);v=dict(base_logits=source['valid_logits'].numpy(),candidate_logits=vv['logits'],labels=vv['labels'])
            qs=read(old/f'signal_{seed}_{name}_sharp_target_only.npz')
            # Valid geometry is fixed source geometry from the original frozen encoder.
            from routing import scores
            vg=source['valid_geometry'];vs={k:t.numpy() for k,t in scores(source['valid_logits'],torch.from_numpy(vv['logits']),vg,source['calibration']).items()};rs={k:r[k] for k in KEYS}
            predictions,notes=policies(v,q,r,vs,qs,rs);save(f'policy_fit_{seed}_{name}.json',notes)
            bp=q['base_logits'].argmax(-1);cp=q['candidate_logits'].argmax(-1);y=q['labels'];bf=metrics(y,bp,bp)['macro_f1']
            for policy,pred in predictions.items():
                np.savez_compressed(RUN/'artifacts'/f'policy_{seed}_{name}_{policy}.npz',predictions=pred,rows=base['rows'])
                met=metrics(y,pred,bp);rows.append(dict(seed=seed,slice=name,condition='v6_lora',policy=policy,**met,delta_f1_pp=100*(met['macro_f1']-bf),changes=int((pred!=bp).sum())))
            report=dict(reference=correlation(rs,r['base_logits'].argmax(-1)!=r['candidate_logits'].argmax(-1)),query_diagnostic=correlation(qs,bp!=cp,y,bp,cp))
            save(f'correlations_{seed}_{name}.json',report)
    save('policy_scores.json',rows)

if __name__=='__main__':main()
