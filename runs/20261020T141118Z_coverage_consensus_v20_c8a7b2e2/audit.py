"""Full model replay with independent nearest-class routing and metric checks."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score
from runner import RUN,ROOT,C,A,CK,F,FA,P,PA,V,VA,read,save,sha,states,support,verify,verify_fits
from maintenance_base import fit_head


def normalize(x):
    return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)


def nearest_class(sim,y,ids):
    score=np.full((len(sim),102),-np.inf,dtype=np.float32)
    for c in range(102):
        sel=ids[y[ids]==c]
        if len(sel):score[:,c]=sim[:,sel].max(axis=1)
    return score


def route(pred,base,scores):
    out=pred.copy();ii=np.arange(len(base))
    reject=(pred!=scores.argmax(axis=1))|(scores[ii,pred]<=scores[ii,base])
    out[reject]=base[reject]
    return out


def gain(z,original,p,base):
    a=np.take_along_axis(z,p[:,None],axis=1).ravel()-np.take_along_axis(z,base[:,None],axis=1).ravel()
    b=np.take_along_axis(original,p[:,None],axis=1).ravel()-np.take_along_axis(original,base[:,None],axis=1).ravel()
    return (a/np.linalg.norm(z,axis=1).clip(1e-12)-b/np.linalg.norm(original,axis=1).clip(1e-12)).astype(np.float32)


def independent_vote(base,ps,gs,k):
    # Strict majorities: only a proposed changed label with >=k votes can win.
    ps=np.stack(ps);gs=np.stack(gs);out=base.copy();score=np.zeros(len(base),np.float32)
    for p in ps:
        mask=ps==p[None,:];n=mask.sum(axis=0)
        accepted=(p!=base)&(n>=k)
        out[accepted]=p[accepted]
        score[accepted]=(np.where(mask,gs,0).sum(axis=0)/n.clip(1))[accepted]
    return out,score


def independent_metrics(y,p,base):
    good=base==y;changed=p!=base
    d=int((good&changed).sum());r=int((~good&(p==y)).sum());w=int((~good&changed&(p!=y)).sum())
    return {'macro_f1':float(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(y,p)),'R':r,'D':d,'W':w,'NAR':d/good.sum(),'Recovery':r/(~good).sum()}


def independent_frontier_at_D(y,p,base,g,budget):
    ids=np.flatnonzero(p!=base);ids=ids[np.argsort(-g[ids],kind='stable')]
    if not len(ids):return {'R':0,'D':0,'W':0}
    end=np.r_[np.flatnonzero(g[ids][1:]!=g[ids][:-1]),len(ids)-1]
    d=np.r_[0,np.cumsum(base[ids]==y[ids])[end]]
    r=np.r_[0,np.cumsum((base[ids]!=y[ids])&(p[ids]==y[ids]))[end]]
    w=np.r_[0,np.cumsum((base[ids]!=y[ids])&(p[ids]!=y[ids]))[end]]
    feasible=np.flatnonzero(d<=budget)
    # Independent lexicographic ordering, including the frozen point.
    order=np.lexsort((w[feasible],d[feasible],-r[feasible]));j=feasible[order[0]]
    return {'R':int(r[j]),'D':int(d[j]),'W':int(w[j])}


def main():
    verify();verify_fits();torch.set_num_threads(C['budget']['threads'])
    records=json.loads((A/'scores.json').read_text());transfers=json.loads((A/'transfers.json').read_text())
    assert len(records)==186 and len(transfers)==180
    nheads=0;noutputs=0;replayed=[]
    for st in states():
        key=st['key'];s=st['seed'];r=st['role'];source,sup,sx,w0,b0=support(st);sy=sup['labels']
        q=read(FA/f'features_{s}_{r}_query.npz');ev=read(FA/f'evaluation_labels_{r}.npz');y=ev['labels']
        assert np.array_equal(q['rows'],ev['rows']) and not np.intersect1d(q['rows'],sup['rows']).size
        x=(q['features']-source['mu'])/source['sd'];original=q['logits'];base=original.argmax(axis=1)
        sim=normalize(x)@normalize(sx).T;all_ids=np.arange(len(sy));full=nearest_class(sim,sy,all_ids)
        sets=read(A/f'splits_{key}.npz');assert np.array_equal(sets['rows'],sup['rows'])
        old=read(PA/f'oof_{key}.npz');assert np.array_equal(old['labels'],sy)
        families={}
        for family,n in [('old',3),('bag',5)]:
            raw=[];gains=[];routed=[]
            for b in range(n):
                if family=='old':
                    ids=np.flatnonzero(old['folds']!=b);path=P/'checkpoints'/f'oof_{key}_fold{b}_head1.npz'
                else:
                    ids=sets[f'bag{b}'];expected=[]
                    for i,c in enumerate(sy):
                        candidates=np.flatnonzero(sy==c)
                        omit=candidates[(b+int(c))%len(candidates)] if len(candidates)>1 else -1
                        if i!=omit:expected.append(i)
                    assert np.array_equal(ids,np.array(expected)) and len(np.unique(sy[ids]))==102
                    path=CK/f'{key}_bag{b}.npz'
                    rebuilt,_=fit_head(sx[ids],sy[ids],w0,b0,C['linear'])
                    saved=read(path)
                    np.testing.assert_allclose(rebuilt['weight'],saved['weight'],atol=1e-7,rtol=1e-7)
                    np.testing.assert_allclose(rebuilt['bias'],saved['bias'],atol=1e-7,rtol=1e-7);nheads+=1
                model=read(path);z=x@model['weight'].T+model['bias'];p=z.argmax(axis=1)
                raw.append(p);gains.append(gain(z,original,p,base));routed.append(route(p,base,nearest_class(sim,sy,ids)))
            families[family]=(raw,gains,routed)
        expected={}
        for family,ks in [('old',[2,3]),('bag',[3,4,5])]:
            ps,gs,rs=families[family]
            for k in ks:
                expected[f'{family}_local_vote{k}']=independent_vote(base,rs,gs,k)
                if k<(3 if family=='old' else 5):
                    p,g=independent_vote(base,ps,gs,k);p=route(p,base,full)
                    expected[f'{family}_heads_vote{k}_full_local']=(p,np.where(p!=base,g,0))
        for name,oldname in [('full_head','raw_head1'),('full_local','local_head1')]:
            cached=read(PA/f'pred_{key}_{oldname}.npz');expected[name]=(cached['predictions'],cached['gain'])
        assert set(expected)==set(C['policies'])
        baseline=expected['old_local_vote2'][0];legacy=read(VA/f'pred_{key}_local_head1_uncalibrated_vote2.npz')
        assert np.array_equal(baseline,legacy['predictions'])
        fullm=independent_metrics(y,expected['full_head'][0],base)
        bm=independent_metrics(y,base,base);wrong=int((base!=y).sum())
        if st['draw']==C['support_seeds'][0]:
            frozen=next(a for a in records if a['key']==key and a['policy']=='frozen')
            for k,v in bm.items():np.testing.assert_allclose(frozen[k],v,atol=1e-12)
            noutputs+=1
        for name,(p,g) in expected.items():
            cached=read(A/f'pred_{key}_{name}.npz')
            assert np.array_equal(p,cached['predictions'])
            np.testing.assert_allclose(g,cached['gain'],atol=2e-6,rtol=1e-5)
            row=next(a for a in records if a['key']==key and a['policy']==name);m=independent_metrics(y,p,base)
            for k,v in m.items():np.testing.assert_allclose(row[k],v,atol=1e-12)
            np.testing.assert_allclose(row['delta_f1_pp'],100*(m['macro_f1']-bm['macro_f1']),atol=1e-10)
            np.testing.assert_allclose(row['Recovery_retention'],m['Recovery']/fullm['Recovery'],atol=1e-12)
            for control in ['full_head','full_local','old_local_vote2']:
                cp,cg=expected[control];match=independent_frontier_at_D(y,cp,base,cg,m['D'])
                np.testing.assert_allclose(row[f'actualD_gain_vs_{control}_pp'],100*(m['R']-match['R'])/wrong,atol=1e-10)
            tr=next(a for a in transfers if a['key']==key and a['policy']==name)
            truth={'new_correct_vs_old':int(((p==y)&(baseline!=y)).sum()),'lost_correct_vs_old':int(((p!=y)&(baseline==y)).sum()),'new_damage_vs_old':int(((base==y)&(p!=y)&(baseline==y)).sum()),'removed_damage_vs_old':int(((base==y)&(p==y)&(baseline!=y)).sum())}
            # New/lost correctness excludes frozen-correct cases by definition in stored transitions.
            truth['new_correct_vs_old']=int(((base!=y)&(p==y)&(baseline!=y)).sum())
            truth['lost_correct_vs_old']=int(((base!=y)&(p!=y)&(baseline==y)).sum())
            for k,v in truth.items():assert tr[k]==v,(key,name,k,tr[k],v)
            replayed.append(row);noutputs+=1
        print(json.dumps({'audited':key,'refitted_heads':nheads,'outputs':noutputs}),flush=True)
    assert nheads==90 and noutputs==186
    summary=json.loads((A/'summary.json').read_text())
    for s in summary:
        rr=[r for r in replayed if r['role']==s['role'] and r['policy']==s['policy']]
        for k in ['macro_f1','accuracy','delta_f1_pp','NAR','Recovery','Recovery_retention','actualD_gain_vs_full_local_pp','actualD_gain_vs_old_local_vote2_pp']:
            np.testing.assert_allclose(s[k],np.mean([a[k] for a in rr]),atol=1e-12)
        assert s['max_NAR']==max(a['NAR'] for a in rr)
    decision=json.loads((A/'decision.json').read_text());pass_roles=[]
    for d in decision['roles']:
        ss=next(s for s in summary if s['role']==d['role'] and s['policy']==C['primary'])
        old=next(s for s in summary if s['role']==d['role'] and s['policy']=='old_local_vote2')
        nonnegative=sum(np.mean([r['actualD_gain_vs_full_local_pp'] for r in replayed if r['role']==d['role'] and r['policy']==C['primary'] and r['seed']==seed])>=0 for seed in C['model_seeds'])
        ok=ss['max_NAR']<=.01 and 100*(ss['Recovery']-old['Recovery'])>=.5 and ss['macro_f1']>=old['macro_f1'] and ss['actualD_gain_vs_full_local_pp']>=0 and nonnegative>=2
        assert d['passed']==bool(ok);pass_roles.append(ok)
    assert decision['passed']==bool(all(pass_roles))
    output_bytes=sum(p.stat().st_size for folder in [A,CK,RUN/'lgs'] for p in folder.rglob('*') if p.is_file())
    assert output_bytes<=C['budget']['max_output_gib']*1024**3
    save('audit.json',{'passed':True,'heads_retrained_and_weight_checked':nheads,'prediction_states_independently_replayed':noutputs,'old_fold_models_replayed':54,'support_splits_verified':90,'independent_metrics_routes_frontiers_transitions':True,'input_freeze_and_query_isolation':True,'summary_and_decision_verified':True,'output_bytes':output_bytes})
    paths=list(CK.glob('*.npz'))+list(A.glob('*.npz'))+[A/n for n in ['audit.json','summary.json','scores.json','decision.json','transfers.json','fit_freeze.json','freeze.json']]
    save('delivery_manifest.json',{str(p.relative_to(RUN)):sha(p) for p in paths})


if __name__=='__main__':main()
