"""Check exploratory score-ordering finding across all ten previously observed slices."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];C=json.loads((RUN/'config.json').read_text())
v3=ROOT/C['runs']['v3'];v4=ROOT/C['runs']['v4'];INPUTS={};DEL={}
INHERITED=json.loads((v4/'artifacts/freeze.json').read_text())
def load(r,name):
    p=r/'artifacts'/name
    if str(r) not in DEL:DEL[str(r)]=json.loads((r/'artifacts/delivery_audit.json').read_text())['outputs']
    h=hashlib.sha256(p.read_bytes()).hexdigest();expected=DEL[str(r)].get('artifacts/'+name,INHERITED.get(str(p.relative_to(ROOT))));assert expected is not None and h==expected,str(p)
    INPUTS[str(p.relative_to(ROOT))]=h
    with np.load(p) as a:return {k:a[k] for k in a.files}
def factor(sl):return sl.split('_')[0]
def joint_frontier(cells,key,cap,eligibility):
    fcounts={f:sum(factor(v['slice'])==f for v in cells) for f in set(factor(v['slice']) for v in cells)}
    arrays=[]
    for v in cells:
        y,b,p=v['y'],v['b'],v['p'];correct=b==y;wrong=~correct
        eligible=np.ones(len(y),bool) if eligibility=='all_changes' else v['signals']['combined']>=0
        use=(p!=b)&eligible;weight=1/(len(fcounts)*fcounts[factor(v['slice'])])
        arrays.append((v['signals'][key][use],weight*correct[use]/correct.sum(),weight*(p[use]==y[use])/wrong.sum()))
    s=np.concatenate([a[0] for a in arrays]);d=np.concatenate([a[1] for a in arrays]);r=np.concatenate([a[2] for a in arrays]);ix=np.argsort(-s,kind='stable');sv=s[ix]
    ends=np.r_[np.flatnonzero(sv[:-1]!=sv[1:]),len(sv)-1];nar=np.r_[0.,np.cumsum(d[ix])[ends]];rec=np.r_[0.,np.cumsum(r[ix])[ends]]
    k=np.flatnonzero(nar<=cap+1e-12)[-1];return 100*float(nar[k]),100*float(rec[k])
def main():
    cfg=json.loads((v4/'config.json').read_text());rows=[];percell=[]
    for method in ['v4_soft_geometry','v3_ungated_lora']:
        for seed in C['seeds']:
            cells=[]
            for sl in cfg['slices']:
                b=load(v3,f'base_{seed}_{sl}.npz');y=b['labels'];bp=b['predictions']
                raw=load(v4 if method.startswith('v4') else v3,f'raw_{seed}_{sl}_'+('soft_geometry' if method.startswith('v4') else 'soft_neighbor_lora')+'.npz')
                assert np.array_equal(b['rows'],raw['rows']);cp=raw['predictions']
                sig=load(v4,f'signal_{seed}_{sl}_'+('soft_geometry' if method.startswith('v4') else 'ungated_lora')+'.npz')
                cells.append(dict(slice=sl,y=y,b=bp,p=cp,signals=sig))
                rec=(bp!=y)&(cp==y);damage=(bp==y)&(cp!=y);mask=rec|damage
                for key in sig:percell.append(dict(method=method,seed=seed,slice=sl,signal=key,auc=float(roc_auc_score(rec[mask],sig[key][mask])) if rec.any() and damage.any() else None))
            for eligibility in ['all_changes','same_frozen_distance_eligible']:
                for key in ['combined','normalized_margin_gain']:
                    row=dict(method=method,seed=seed,signal=key,eligibility=eligibility)
                    for cap in [.01,.02]:
                        n,r=joint_frontier(cells,key,cap,eligibility);row['nar'+str(int(cap*100))]=n;row['recovery'+str(int(cap*100))]=r
                    rows.append(row)
    summary=[]
    for method in ['v4_soft_geometry','v3_ungated_lora']:
        for eligibility in ['all_changes','same_frozen_distance_eligible']:
            for key in ['combined','normalized_margin_gain']:
                rr=[r for r in rows if r['method']==method and r['eligibility']==eligibility and r['signal']==key]
                summary.append(dict(method=method,signal=key,eligibility=eligibility,**{k+'_mean':float(np.mean([r[k] for r in rr])) for k in ['recovery1','recovery2']}))
    (RUN/'artifacts/wider_score_diagnostics.json').write_text(json.dumps({'role':'exploratory observed ten-slice posthoc; shared threshold across factors per seed; no operational selection','cells':percell,'joint':rows,'summary':summary},indent=2)+'\n')
    (RUN/'artifacts/wider_inputs_sha256.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
    (RUN/'artifacts/wider_score_audit.json').write_text(json.dumps({'passed':True,'new_training':0,'new_model_inference':0,'input_arrays_checked_against_delivery':len(INPUTS),'all_ten_slices':cfg['slices']},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
