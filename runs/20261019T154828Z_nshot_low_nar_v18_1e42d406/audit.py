"""All support/OOF choices, model reloads and independent terminal metrics."""
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from runner import (RUN, C, ART, CK, P, PA, sha, save, read, load_support, verify, verify_fits, query_outputs)
from methods import (unit, predict, local_scores, policies, select_policy, best_under_damage,
                     least_damage_for_recovery)


def independent_curve(y,p,base,gain):
    ix=np.flatnonzero(p!=base);ix=ix[np.argsort(-gain[ix],kind='stable')]
    if not len(ix):return {k:np.array([0],dtype=int) for k in ['R','D','W']}
    end=np.r_[np.flatnonzero(gain[ix][1:]!=gain[ix][:-1]),len(ix)-1]
    r=np.cumsum((base[ix]!=y[ix])&(p[ix]==y[ix]))
    d=np.cumsum((base[ix]==y[ix])&(p[ix]!=y[ix]))
    w=np.cumsum((base[ix]!=y[ix])&(p[ix]!=y[ix]))
    return {k:np.r_[0,v[end]] for k,v in [('R',r),('D',d),('W',w)]}


def assert_proto(model,x,y,source):
    z=unit(x);counts=np.bincount(y,minlength=102)
    expected=[]
    for cls in range(102):
        sums=z[y==cls].sum(0,dtype=np.float64)
        expected.append((sums+3*source[cls])/(counts[cls]+3))
    np.testing.assert_allclose(model['centers'],unit(np.array(expected)),atol=1e-6,rtol=1e-5)


def main():
    torch.set_num_threads(C['budget']['threads'])
    verify();verify_fits()
    states=json.loads((ART/'states.json').read_text())
    scorelist=json.loads((ART/'scores.json').read_text())
    scores={s['key']:s for s in scorelist}
    matches={m['key']:m for m in json.loads((ART/'matches.json').read_text())}
    assert len(states)==54 and len(scores)==1410 and len(matches)==1404
    checked_models=set();checked_states=[];risk=[]
    for i,state in enumerate(states):
        seed,role,draw,cap=(state[k] for k in ['seed','role','draw','cap']);key=state['key']
        source,support,x,original,w0,b0=load_support(seed,role,draw,cap)
        assert state['labels']==len(support['labels'])
        assert state['source_support_sha256']==sha(PA/f'support_{role}_{draw}_cap{cap}.npz')
        y=support['labels'];oof=read(ART/f'oof_{key}.npz');folds=oof['folds']
        assert np.array_equal(oof['labels'],y) and np.array_equal(oof['rows'],support['rows'])
        np.testing.assert_array_equal(oof['original'],original)
        for cls in np.unique(y):
            idx=np.flatnonzero(y==cls)
            np.testing.assert_array_equal(folds[idx],(cls+np.arange(len(idx)))%3)
        for name in C['base_models']:
            path=CK/f'{key}_{name}.npz';model=read(path);checked_models.add(str(path))
            if name=='shrink':assert_proto(model,x,y,source['means'])
            else:
                assert model['weight'].shape==w0.shape and model['bias'].shape==b0.shape
                assert np.isfinite(model['weight']).all() and np.isfinite(model['bias']).all()
            if name in ['head1','shrink']:
                old='head_l2' if name=='head1' else 'shrink_proto'
                assert sha(path)==sha(P/'checkpoints'/f'{key}_{old}.npz')
        for fold in range(3):
            train=np.flatnonzero(folds!=fold);hold=np.flatnonzero(folds==fold)
            assert set(support['rows'][train]).isdisjoint(support['rows'][hold])
            np.testing.assert_array_equal(oof['local'][hold],local_scores(x[hold],x[train],y[train]))
            for name in C['base_models']:
                path=CK/f'oof_{key}_fold{fold}_{name}.npz';model=read(path);checked_models.add(str(path))
                if name=='shrink':assert_proto(model,x[train],y[train],source['means'])
                else:assert np.isfinite(model['weight']).all() and np.isfinite(model['bias']).all()
                np.testing.assert_array_equal(oof[name][hold],predict(model,x[hold]))
        outputs=list(policies(original,{n:oof[n] for n in C['base_models']},oof['local'],C['policies'],C['prototype_logit_scale']))
        selection=select_policy(y,original,outputs,C['selectors'])
        assert selection==json.loads((ART/f'selection_{key}.json').read_text())
        rows,base,outputs=query_outputs(state)
        eval_y=read(PA/f'evaluation_labels_{role}.npz')
        assert np.array_equal(rows,eval_y['rows']) and set(support['rows']).isdisjoint(rows)
        qy=eval_y['labels'];correct=int((base==qy).sum());wrong=len(qy)-correct
        curves={}
        for name,p,g in outputs:
            state_id=key+'_'+name;s=scores[state_id];saved=read(ART/f'pred_{state_id}.npz')
            np.testing.assert_array_equal(p,saved['predictions']);np.testing.assert_array_equal(g,saved['gain'])
            assert abs(accuracy_score(qy,p)-s['accuracy'])<1e-12
            assert abs(f1_score(qy,p,labels=np.arange(102),average='macro',zero_division=0)-s['macro_f1'])<1e-12
            is_correct=base==qy
            r=int((~is_correct&(p==qy)).sum());d=int((is_correct&(p!=qy)).sum());w=int((~is_correct&(p!=qy)&(p!=base)).sum())
            assert (r,d,w)==(s['R'],s['D'],s['W'])
            assert s['NAR']==d/correct and s['Recovery']==r/wrong
            curve=independent_curve(qy,p,base,g);curves[name]=curve
            at1=best_under_damage(curve,int(correct*.01))
            assert s['diagnostic_Recovery_at_NAR1']==at1['R']/wrong
            checked_states.append(state_id)
            if name in C['selectors']:
                risk.append({'key':state_id,'OOF_correct':selection[name]['OOF_original_correct'],
                             'OOF_D':selection[name]['OOF_D'],'OOF_NAR':selection[name]['OOF_empirical_NAR'],
                             'chosen_policy':selection[name]['policy'],'threshold':selection[name]['threshold'],
                             'query_NAR':s['NAR'],'query_Recovery':s['Recovery'],'OOF_to_query_NAR_gap':s['NAR']-selection[name]['OOF_empirical_NAR']})
        anchor=best_under_damage(curves['raw_head1'],int(correct*.01))
        for name,curve in curves.items():
            m=matches[key+'_'+name]
            assert m['anchor_head_at1']==anchor
            assert m['same_damage']==best_under_damage(curve,anchor['D'])
            assert m['same_recovery']==least_damage_for_recovery(curve,anchor['R'])
        if i%9==0:print(json.dumps({'audited_support_states':i+1,'total':54}),flush=True)
    for seed in C['model_seeds']:
        for role in C['roles']:
            base=read(PA/f'features_{seed}_{role}_query.npz')['logits'].argmax(1)
            y=read(PA/f'evaluation_labels_{role}.npz')['labels'];s=scores[f'{seed}_{role}_frozen']
            assert abs(f1_score(y,base,labels=np.arange(102),average='macro')-s['macro_f1'])<1e-12
    assert len(checked_models)==864 and len(checked_states)==1404
    size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file());assert size<C['budget']['max_output_gib']*1024**3
    save('risk_transfer.json',risk)
    save('audit.json',{'passed':True,'support_states':54,'model_files_reloaded':len(checked_models),'new_supervised_heads':594,
                       'OOF_caches_reloaded':54,'selectors_recomputed':108,'query_states_recomputed':1410,
                       'independent_metrics_and_curves':True,'fold_and_support_query_isolation':True,'bytes':size})
    save('delivery_manifest.json',{str(p.relative_to(RUN)):sha(p) for folder in [ART,CK] for p in sorted(folder.glob('*.npz'))})
    print(json.dumps({'audit_passed':True,'models':len(checked_models),'states':1410,'bytes':size}),flush=True)


if __name__=='__main__':main()
