"""Post-hoc deliverable verifier; never imported by adaptation or selection.

Checks the full frozen request rather than inferring success from process exit.
This new verifier does not alter the frozen experiment or its report renderer.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from runtime import RUN,ROOT,C,sha,save

def read(name):return json.loads((RUN/'artifacts'/name).read_text())
def main():
    freeze=read('freeze.json')
    for p,h in freeze.items():assert sha(ROOT/p)==h,p
    protocol=RUN/'DIAGNOSTIC_PROTOCOL.md'
    assert protocol.read_bytes()==(ROOT/'MULTI_FACTOR_DIAGNOSTIC_PROTOCOL.md').read_bytes()
    data=read('data_audit.json');gate=read('diagnostic_gate.json')
    assert data['passed'] and gate['unlocked'] and gate['scope']=='observed_diagnostic_only'
    assert not gate['independent_confirmation'] and not gate['official_site_map_verified']
    assert data['unknown_version_unique_excluded']==10017 and data['attributed_version_unique_before_filters']==90145
    manifest=list(csv.DictReader((RUN/'artifacts/diagnostic_rows.csv').open()))
    assert len({r['tam_hash'] for r in manifest})==len(manifest)==124643
    for name in C['slices']:
        subset=[r for r in manifest if r['slice']==name]
        s=[r for r in subset if r['role']=='reference'];q=[r for r in subset if r['role']=='query']
        assert len(s)==1024 and len(q)==data['slices'][name]['query']
        assert {int(r['label']) for r in q}==set(range(102))
        assert {r['group'] for r in s}.isdisjoint(r['group'] for r in q)
        if name=='behavior_subpage':assert len({r['url'] for r in subset})==17739
    scores=read('scores.json');expected={(s,n,c) for s in C['seeds'] for n in C['slices'] for c in C['conditions']}
    assert len(scores)==210 and {(r['seed'],r['slice'],r['condition']) for r in scores}==expected
    post=read('postrun_audit.json');assert post['passed'] and post['prediction_cells_reloaded']==post['metrics_recomputed']==210
    for r in scores:
        key=f"{r['seed']}_{r['slice']}_{r['condition']}"
        assert sha(RUN/'checkpoints'/f'{key}.pt')==r['checkpoint_sha256']
        assert sha(RUN/'artifacts'/f'pred_{key}.npz')==r['prediction_sha256']
        history=read(f'updates_{key}.json')
        assert len(history)==r['steps']==(0 if r['condition'] in ('frozen','token_adain') else 800)
        assert all(np.isfinite(x['loss']) for x in history)
    table1=list(csv.DictReader((RUN/'artifacts/table1_impact.csv').open()))
    table2=list(csv.DictReader((RUN/'artifacts/table2_alignment.csv').open()))
    table3=list(csv.DictReader((RUN/'artifacts/table3_external_v4.csv').open()))
    assert len(table1)==10 and len(table2)==70 and len(table3)==8
    for r in table2:
        values=[x for x in scores if x['slice']==r['slice'] and x['condition']==r['condition']]
        assert len(values)==3
        for key in ('macro_f1','accuracy','delta_f1_pp','nar','recovery'):
            x=np.array([v[key] for v in values]);scale=1 if key=='delta_f1_pp' else 100
            assert np.isclose(float(r[key+'_mean']),x.mean()*scale,rtol=0,atol=1e-10)
            assert np.isclose(float(r[key+'_sd']),x.std(ddof=1)*scale,rtol=0,atol=1e-10)
    reuse=read('external_reuse_audit.json');assert reuse['passed'] and reuse['metrics_reused']==144 and not reuse['new_external_scoring']
    ext=ROOT/C['external_run']
    assert (RUN/'artifacts/external_v4_scores_reused.json').read_bytes()==(ext/'artifacts/scores.json').read_bytes()
    assert (RUN/'table3_external_v4_full.md').read_bytes()==(ext/'RESULTS.md').read_bytes()
    rendered=read('render_audit.json');assert rendered['passed'] and rendered['joint_tsne']
    for p,h in rendered['files_sha256'].items():assert sha(RUN/p)==h,p
    for name in ['figure_impact_radar','figure_manifold_version_048','figure_manifold_behavior_subpage']:
        for ext in ('png','pdf','svg'):
            p=RUN/'artifacts'/f'{name}.{ext}';assert p.is_file() and p.stat().st_size>1000
    manifold=read('manifold_data.json');assert set(manifold)==set(C['visualization']['slices'])
    for name,r in manifold.items():
        z=np.array(r['joint_coordinates']);assert z.shape==(r['source_count']+2*r['target_count']+10,2) and np.isfinite(z).all()
        assert len(r['rows'])==len(r['labels'])==r['target_count'] and set(r['labels'])<=set(range(10))
        for point in ('before','after'):assert all(np.isfinite(v) for v in r['metrics'][point].values())
    # Re-extract every plotted vector from its actual checkpoint. These
    # post-hoc queries never feed a gradient, threshold or checkpoint choice.
    from contextlib import nullcontext
    from runtime import setup,load_model,load_slice,configure,reference
    from mechanisms import get_tokens,restore_aligner
    from token_alignment import aligned_generator
    setup();seed=C['visualization']['seed'];ref=reference(seed)
    mu=ref['moments']['mean'].numpy();sd=ref['moments']['variance'].clamp_min(1e-5).sqrt().numpy()
    prototypes=(ref['class_prototypes'].numpy()-mu)/sd
    for name in C['visualization']['slices']:
        _,q=load_slice(name);y=q['labels'].numpy();selected=[]
        for label in C['visualization']['classes']:
            ids=np.flatnonzero(y==label)
            selected.extend(sorted(ids,key=lambda i:hashlib.sha256(str(int(q['rows'][i])).encode()).hexdigest())[:C['visualization']['max_per_class']])
        d={k:q[k][selected] for k in ('timestamps','tam','rows')}
        for condition,point in [('frozen','before'),('prototype_anchor','after')]:
            saved=torch.load(RUN/'artifacts'/f'visual_{name}_{condition}.pt',map_location='cpu',weights_only=True)
            assert torch.equal(saved['rows'],d['rows']) and torch.equal(saved['labels'],q['labels'][selected])
            m=load_model(seed);configure(m,condition)
            ck=torch.load(RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt',map_location='cpu',weights_only=True)
            m.load_state_dict(ck['model'],strict=True);a=restore_aligner(ck['aligner'],condition)
            with aligned_generator(m,a) if a is not None else nullcontext():features=get_tokens(m,d).mean(1).cpu()
            assert torch.allclose(features,saved['features'],rtol=1e-5,atol=1e-5)
            x=(features.numpy()-mu)/sd;labels=q['labels'][selected].numpy()
            mse=((x-prototypes[labels])**2).mean()
            assert np.isclose(mse,manifold[name]['metrics'][point]['true_prototype_mse'],rtol=1e-5,atol=1e-5)
            del m,a
    # A negative mechanism result still completes the authorized experiment.
    decisions=read('decisions.json');assert 'candidate_pass' in decisions and 'topology_pass' in decisions
    save('delivery_audit.json',{'passed':True,'scope':'full frozen diagnostic request','manifest_rows':len(manifest),'query_rows':114403,'evaluated_cells':len(scores),'table_rows':[10,70,8],'reused_external_cells':144,'figures':3,'figure_formats':['png','pdf','svg'],'plotted_token_sets_reextracted':4,'independent_confirmation_claimed':False,'method_success_required_for_completion':False,'verifier_sha256':sha(Path(__file__))})
    print('All frozen deliverables verified: 210 cells, three tables, three figures, external144 reuse.')

if __name__=='__main__':main()
