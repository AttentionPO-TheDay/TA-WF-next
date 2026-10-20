"""Exploratory score dissection after primary retrospective; labels only diagnose."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text())

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def frontier(y,b,p,s,eligible,cap):
    change=(p!=b)&eligible;correct=b==y;wrong=~correct
    scores=s[change];d=correct[change].astype(float)/correct.sum();r=(p[change]==y[change]).astype(float)/wrong.sum()
    if not len(scores):return 0.,0.
    ix=np.argsort(-scores,kind='stable');sv=scores[ix];ends=np.r_[np.flatnonzero(sv[:-1]!=sv[1:]),len(sv)-1]
    nar=np.r_[0.,np.cumsum(d[ix])[ends]];rec=np.r_[0.,np.cumsum(r[ix])[ends]]
    k=np.flatnonzero(nar<=cap+1e-12)[-1]
    return float(nar[k]),float(rec[k])

def main():
    frozen=json.loads((RUN/'artifacts/inputs_sha256.json').read_text())
    version='v6';r=ROOT/C['runs'][version];rows=[]
    for seed in C['seeds']:
        for sl in C['slices']:
            paths=[r/'artifacts'/f'{prefix}_{seed}_{sl}{suffix}.npz' for prefix,suffix in [('base',''),('raw','_sharp_target_only'),('signal','_sharp_target_only')]]
            for p in paths:assert sha(p)==frozen[str(p.relative_to(ROOT))]
            with np.load(paths[0]) as a:y=a['labels'];bp=a['predictions']
            with np.load(paths[1]) as a:cp=a['predictions']
            rec=(bp!=y)&(cp==y);dam=(bp==y)&(cp!=y);mask=rec|dam
            with np.load(paths[2]) as a:signals={k:a[k] for k in a.files}
            for eligibility in ['all_changes','same_frozen_distance_eligible']:
                eligible=np.ones(len(y),bool) if eligibility=='all_changes' else signals['combined']>=0
                for signal,s in signals.items():
                    used=mask&eligible;v={'seed':seed,'slice':sl,'method':'v6_sharp_target_only','eligibility':eligibility,'signal':signal,
                        'auc_recovery_vs_damage':float(roc_auc_score(rec[used],s[used])),
                        'recoveries_available':int((rec&eligible).sum()),'damages_available':int((dam&eligible).sum())}
                    for cap in [.01,.02]:
                        n,rr=frontier(y,bp,cp,s,eligible,cap);v['nar_at_cap'+str(int(cap*100))]=100*n;v['recovery_at_cap'+str(int(cap*100))]=100*rr
                    rows.append(v)
    summary=[]
    for sl in C['slices']:
        for eligibility in ['all_changes','same_frozen_distance_eligible']:
            for signal in signals:
                rr=[r for r in rows if r['slice']==sl and r['eligibility']==eligibility and r['signal']==signal]
                d=dict(slice=sl,eligibility=eligibility,signal=signal)
                for key in ['auc_recovery_vs_damage','recoveries_available','damages_available','recovery_at_cap1','recovery_at_cap2']:
                    d[key+'_mean']=float(np.mean([r[key] for r in rr]));d[key+'_sd']=float(np.std([r[key] for r in rr],ddof=1))
                summary.append(d)
    (RUN/'artifacts/score_dissection.json').write_text(json.dumps({'role':'posthoc exploratory; thresholds use terminal labels, never deployed','cells':rows,'summary':summary},indent=2)+'\n')
    (RUN/'artifacts/score_dissection_audit.json').write_text(json.dumps({'passed':True,'original_input_hashes_checked':True,'new_training':False,'new_inference':False,'new_operational_thresholds':False,'script_sha256':sha(RUN/'score_diagnostics.py')},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
