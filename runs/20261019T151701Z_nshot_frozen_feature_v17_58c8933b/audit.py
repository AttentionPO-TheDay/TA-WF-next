"""Reload every support/model/prediction; independent sklearn metric checks."""
import json
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from runner import RUN, ROOT, C, ART, CK, sha, save, read_npz, prepared, split, verify, load_model, extract
from maintenance import unit, predict, margin_gain, frontier, best_under_damage, least_damage_for_recovery


def independent_curve(y, p, base, score):
    ix = np.flatnonzero(p != base)
    ix = ix[np.argsort(-score[ix], kind='stable')]
    r = np.cumsum((base[ix] != y[ix]) & (p[ix] == y[ix]))
    d = np.cumsum((base[ix] == y[ix]) & (p[ix] != y[ix]))
    w = np.cumsum((base[ix] != y[ix]) & (p[ix] != y[ix]))
    end = np.r_[np.flatnonzero(score[ix][1:] != score[ix][:-1]), len(ix)-1] if len(ix) else np.array([],dtype=int)
    return {k: np.r_[0, val[end]] for k, val in [('R',r),('D',d),('W',w)]}


def main():
    torch.set_num_threads(C['budget']['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    verify()
    fits = json.loads((ART/'fits.json').read_text())
    scores = json.loads((ART/'scores.json').read_text())
    matches = {r['key']: r for r in json.loads((ART/'matched.json').read_text())}
    assert len(fits) == 162 and len(scores) == 174
    assert json.loads((ART/'fits_complete.json').read_text())['fit_manifest_sha256'] == sha(ART/'fits.json')
    features_checked, states_checked, supports_checked = [], [], []
    for role in C['targets']:
        data = prepared(C['targets'][role]); s = split(role)
        ref_y = data['labels'][s['support_indices']].numpy()
        for draw in C['support_seeds']:
            prior = set()
            for n in C['nominal_shots']:
                support = read_npz(ART/f'support_{role}_{draw}_cap{n}.npz')
                ix = support['reference_positions']
                assert prior <= set(ix); prior = set(ix)
                assert len(set(ix)) == len(ix)
                assert np.array_equal(support['labels'],ref_y[ix])
                assert np.array_equal(support['prepared_indices'],s['support_indices'][ix])
                assert np.array_equal(support['class_counts'], np.minimum(np.bincount(ref_y,minlength=102),n))
                assert not set(support['rows']) & set(s['query'])
                supports_checked.append(f'{role}_{draw}_{n}')
    for seed in C['model_seeds']:
        source = read_npz(ART/f'source_features_{seed}.npz')
        np.testing.assert_allclose(source['mu'],source['features'].mean(0),atol=0,rtol=0)
        np.testing.assert_allclose(source['sd'],np.maximum(source['features'].std(0),1e-3),atol=0,rtol=0)
        sx = unit((source['features']-source['mu'])/source['sd'])
        means = np.stack([sx[source['labels']==c].mean(0,dtype=np.float64) for c in range(102)]).astype(np.float32)
        np.testing.assert_allclose(means,source['means'],atol=1e-6,rtol=1e-5)
        for role in C['targets']:
            query=read_npz(ART/f'features_{seed}_{role}_query.npz')
            ref=read_npz(ART/f'features_{seed}_{role}_reference.npz')
            eval_y=read_npz(ART/f'evaluation_labels_{role}.npz')
            assert np.array_equal(query['rows'],eval_y['rows'])
            y=eval_y['labels']; original=query['logits']; base=original.argmax(1)
            x=(query['features']-source['mu'])/source['sd']
            curves={}
            for row in [r for r in scores if r['seed']==seed and r['role']==role]:
                key=row['key']; arm=row['arm']; cap=row['cap']; draw=row['draw']
                if arm=='frozen':
                    z=original
                else:
                    path=CK/(f'{seed}_source_proto.npz' if arm=='source_proto' else f'{key}.npz')
                    model=read_npz(path)
                    if cap:
                        fit=next(f for f in fits if f['key']==key)
                        assert sha(path)==fit['checkpoint_sha256']
                        sup_path=ART/f'support_{role}_{draw}_cap{cap}.npz'
                        assert sha(sup_path)==fit['support_sha256']
                        sup=read_npz(sup_path); assert row['labels']==len(sup['labels'])
                        if arm.endswith('proto'):
                            sup_x=unit((ref['features'][sup['reference_positions']]-source['mu'])/source['sd'])
                            target=np.stack([sup_x[sup['labels']==c].mean(0,dtype=np.float64) for c in range(102)])
                            counts=sup['class_counts'][:,None]
                            k=C['prototype']['shrink_source_pseudocount'] if arm=='shrink_proto' else 0
                            expected=unit((target*counts+k*means)/(counts+k))
                            np.testing.assert_allclose(model['centers'],expected,atol=1e-6,rtol=1e-5)
                    z=predict(model,x)
                p=z.argmax(1); gain=margin_gain(z,original,p,base)
                saved=read_npz(ART/f'pred_{key}.npz')
                assert np.array_equal(p,saved['predictions'])
                np.testing.assert_array_equal(gain,saved['gain'])
                assert abs(accuracy_score(y,p)-row['accuracy'])<1e-12
                assert abs(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)-row['macro_f1'])<1e-12
                correct=base==y
                r=int(((~correct)&(p==y)).sum()); d=int((correct&(p!=y)).sum()); w=int(((~correct)&(p!=y)&(p!=base)).sum())
                assert (r,d,w)==(row['R'],row['D'],row['W'])
                assert row['NAR']==d/int(correct.sum()) and row['Recovery']==r/int((~correct).sum())
                curve=independent_curve(y,p,base,gain)
                for k,v in frontier(y,p,base,gain).items():
                    np.testing.assert_array_equal(v,curve[k])
                at1=best_under_damage(curve,int(np.floor(correct.sum()*.01)))
                assert at1==row['at1']
                assert row['diagnostic_Recovery_at_NAR1']==at1['R']/int((~correct).sum())
                curves[(draw,cap,arm)]=curve
                states_checked.append(key)
            for row in [r for r in scores if r['seed']==seed and r['role']==role and r['cap']]:
                key=row['key']; draw=row['draw']; cap=row['cap']
                anchor=best_under_damage(curves[(draw,cap,'target_proto')],int(np.floor((base==y).sum()*.01)))
                m=matches[key]
                assert m['anchor']==anchor
                curve=curves[(draw,cap,row['arm'])]
                assert best_under_damage(curve,anchor['D'])==m['same_damage']
                assert least_damage_for_recovery(curve,anchor['R'])==m['same_recovery']
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['feature_gpu'])
    torch.cuda.set_per_process_memory_fraction(C['budget']['cuda_memory_fraction'])
    select=read_npz(ART/'source_selection.npz')
    for seed in C['model_seeds']:
        model=load_model(seed)
        data=prepared(C['source_run']+'/artifacts/prepared.pt')['source']
        n=C['audit']['reextract_rows']
        # Preserve the original full-batch shape, compare its first n rows.
        batch=C['budget']['batch_size']
        sx,sz=extract(model,data,select['indices'][:batch],f'audit_source_{seed}')
        old=read_npz(ART/f'source_features_{seed}.npz')
        np.testing.assert_allclose(sx[:n],old['features'][:n],atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(sz[:n],old['logits'][:n],atol=1e-5,rtol=1e-5)
        for role in C['targets']:
            data=prepared(C['targets'][role]); s=split(role)
            for kind,idx in [('reference',s['support_indices']),('query',s['query_indices'])]:
                x,z=extract(model,data,idx[:batch],f'audit_{seed}_{role}_{kind}')
                old=read_npz(ART/f'features_{seed}_{role}_{kind}.npz')
                np.testing.assert_allclose(x[:n],old['features'][:n],atol=1e-5,rtol=1e-5)
                np.testing.assert_allclose(z[:n],old['logits'][:n],atol=1e-5,rtol=1e-5)
                features_checked.append(f'{seed}_{role}_{kind}')
        del model
        torch.cuda.empty_cache()
    size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    assert size<C['budget']['max_output_gib']*1024**3
    save('audit.json',{'passed':True,'states':len(states_checked),'fits':len(fits),'support_sets':len(supports_checked),
                       'feature_states_reextracted':len(features_checked)+3,'rows_per_feature_reextract':C['audit']['reextract_rows'],
                       'bytes':size,'metrics_independently_recomputed':True,'query_frontiers_not_deployed':True})
    save('delivery_manifest.json',{str(p.relative_to(RUN)):sha(p) for folder in [ART,CK] for p in sorted(folder.glob('*.npz'))})
    print(json.dumps({'audit_passed':True,'states':len(states_checked),'fits':len(fits),'bytes':size}),flush=True)


if __name__=='__main__':
    main()
