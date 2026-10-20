"""Independent vote logic, all held-out calibrators and terminal metrics."""
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from runner import (RUN,C,ART,P,PA,FA,read,save,sha,verify,verify_calibration,calibration_from_oof,fold_outputs)
from methods import predict,local_scores,apply_choice,best_under_damage,least_damage_for_recovery


def curve(y,p,b,g):
    ix=np.flatnonzero(p!=b);ix=ix[np.argsort(-g[ix],kind='stable')]
    if not len(ix):return {k:np.array([0],dtype=int) for k in ['R','D','W']}
    ends=np.r_[np.flatnonzero(g[ix][1:]!=g[ix][:-1]),len(ix)-1]
    return {k:np.r_[0,np.cumsum(v)[ends]] for k,v in [('R',(b[ix]!=y[ix])&(p[ix]==y[ix])),('D',(b[ix]==y[ix])&(p[ix]!=y[ix])),('W',(b[ix]!=y[ix])&(p[ix]!=y[ix]))]}


def independent_vote(b,ps,gs,k):
    a,c,d=ps
    if k==2:p=np.where(a==c,a,np.where(a==d,a,np.where(c==d,c,b)))
    else:p=np.where((a==c)&(a==d),a,b)
    mask=np.stack([v==p for v in ps]);gain=(np.stack(gs)*mask).sum(0)/np.maximum(mask.sum(0),1)
    return p,np.where(p!=b,gain,0).astype(np.float32)


def main():
    torch.set_num_threads(2);verify();verify_calibration()
    states=json.loads((ART/'states.json').read_text());scores={r['key']:r for r in json.loads((ART/'scores.json').read_text())}
    matches={r['key']:r for r in json.loads((ART/'matches.json').read_text())};models=set();checked=0
    assert len(states)==36 and len(scores)==726 and len(matches)==720
    for number,state in enumerate(states):
        key=state['key'];seed,role,draw,cap=(state[k] for k in ['seed','role','draw','cap'])
        support=read(FA/f'support_{role}_{draw}_cap{cap}.npz');oof=read(PA/f'oof_{key}.npz')
        source=read(FA/f'source_features_{seed}.npz');reference=read(FA/f'features_{seed}_{role}_reference.npz')
        x=(reference['features'][support['reference_positions']]-source['mu'])/source['sd']
        assert np.array_equal(oof['rows'],support['rows']) and np.array_equal(oof['labels'],support['labels'])
        for fold in range(3):
            train=np.flatnonzero(oof['folds']!=fold);hold=np.flatnonzero(oof['folds']==fold)
            assert set(support['rows'][train]).isdisjoint(support['rows'][hold])
            np.testing.assert_array_equal(oof['local'][hold],local_scores(x[hold],x[train],support['labels'][train]))
            for name in C['models']:
                path=P/'checkpoints'/f'oof_{key}_fold{fold}_{name}.npz';models.add(str(path))
                np.testing.assert_array_equal(oof[name][hold],predict(read(path),x[hold]))
        calibration=json.loads((ART/f'calibration_{key}.json').read_text())
        assert calibration==calibration_from_oof(state)
        rows,base,folds=fold_outputs(state);labels=read(FA/f'evaluation_labels_{role}.npz');y=labels['labels']
        assert np.array_equal(rows,labels['rows']) and set(support['rows']).isdisjoint(rows)
        outputs=[]
        for route in [r['name'] for r in C['routes']]:
            for mode in C['modes']:
                ps,gs=[],[]
                for fold in range(3):
                    output=next(o for o in folds[fold] if o[0]==route)
                    p,g=output[1:] if mode=='uncalibrated' else apply_choice(calibration[str(fold)][route][mode],[output],base)
                    ps.append(p);gs.append(g)
                for k in C['votes']:
                    p,g=independent_vote(base,ps,gs,k);outputs.append((f'{route}_{mode}_vote{k}',p,g))
        for name,old in [('full_head1','raw_head1'),('full_local_head1','local_head1')]:
            f=read(PA/f'pred_{key}_{old}.npz');outputs.append((name,f['predictions'],f['gain']))
        curves={};correct=int((base==y).sum());wrong=len(y)-correct
        basef1=f1_score(y,base,labels=np.arange(102),average='macro',zero_division=0)
        for name,p,g in outputs:
            label=key+'_'+name;s=scores[label];saved=read(ART/f'pred_{label}.npz')
            np.testing.assert_array_equal(p,saved['predictions']);np.testing.assert_array_equal(g,saved['gain'])
            assert abs(accuracy_score(y,p)-s['accuracy'])<1e-12
            f1=f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)
            assert abs(f1-s['macro_f1'])<1e-12 and abs(100*(f1-basef1)-s['delta_f1_pp'])<1e-10
            oldcorrect=base==y;r=int((~oldcorrect&(p==y)).sum());d=int((oldcorrect&(p!=y)).sum());w=int((~oldcorrect&(p!=y)&(p!=base)).sum())
            assert (r,d,w)==(s['R'],s['D'],s['W']) and s['NAR']==d/correct and s['Recovery']==r/wrong
            curves[name]=curve(y,p,base,g)
            assert s['diagnostic_Recovery_at_NAR1']==best_under_damage(curves[name],int(correct*.01))['R']/wrong
            checked+=1
        anchor=best_under_damage(curves['full_head1'],int(correct*.01))
        for name,c in curves.items():
            m=matches[key+'_'+name];assert m['anchor_head_at1']==anchor
            assert m['same_damage']==best_under_damage(c,anchor['D'])
            assert m['same_recovery']==least_damage_for_recovery(c,anchor['R'])
        if number%6==0:print(json.dumps({'audited_support_states':number+1,'total':36}),flush=True)
    for seed in C['model_seeds']:
        for role in C['roles']:
            b=read(FA/f'features_{seed}_{role}_query.npz')['logits'].argmax(1);y=read(FA/f'evaluation_labels_{role}.npz')['labels']
            s=scores[f'{seed}_{role}_frozen'];assert abs(f1_score(y,b,labels=np.arange(102),average='macro')-s['macro_f1'])<1e-12
    size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file());assert size<C['budget']['max_output_gib']*1024**3
    assert len(models)==324 and checked==720
    save('audit.json',{'passed':True,'heldout_model_files':len(models),'OOF_caches':36,'calibrators_recomputed':648,
                       'prediction_states':726,'independent_vote_and_metrics':True,'support_fold_query_isolation':True,'bytes':size})
    save('delivery_manifest.json',{str(p.relative_to(RUN)):sha(p) for p in sorted(ART.glob('*.npz'))})
    print(json.dumps({'audit_passed':True,'models':len(models),'states':726,'bytes':size}),flush=True)


if __name__=='__main__':main()
