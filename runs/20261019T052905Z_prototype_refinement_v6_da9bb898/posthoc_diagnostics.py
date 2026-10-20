"""After completion: describe frozen-output error groups; never fit/change inference."""
import json
from collections import defaultdict
import numpy as np
from runtime import RUN,ROOT,C,save,verify,sha

def main():
    verify()
    assert json.loads((RUN/'artifacts/postrun_audit.json').read_text())['passed']
    parent=ROOT/C['v5_run'];records=[]
    oldrows=json.loads((parent/'artifacts/scores.json').read_text())
    newrows=json.loads((RUN/'artifacts/scores.json').read_text());paired=[]
    for row in newrows:
        if row['condition']!='sharp_target_only':continue
        b=next(b for b in oldrows if b['seed']==row['seed'] and b['slice']==row['slice'] and b['condition']=='momentum_proto')
        paired.append(dict(seed=row['seed'],slice=row['slice'],delta_f1_over_v5_pp=row['delta_f1_pp']-b['delta_f1_pp'],
            delta_nar_pp=100*(row['nar']-b['nar']),delta_recovery_pp=100*(row['recovery']-b['recovery']),
            source_valid_f1_change_pp=100*(row['source_valid_macro_f1']-b['source_valid_macro_f1'])))
    save('paired_primary.json',paired)
    for seed in C['seeds']:
        for sl in C['slices']:
            with np.load(RUN/'artifacts'/f'base_{seed}_{sl}.npz') as a:y=a['labels'];bp=a['predictions'];z=a['logits'].astype(np.float64)
            with np.load(RUN/'artifacts'/f'geometry_{seed}_{sl}.npz') as a:near=a['nearest']
            with np.load(parent/'artifacts'/f'raw_{seed}_{sl}_momentum_proto.npz') as a:pp=a['predictions']
            with np.load(RUN/'artifacts'/f'raw_{seed}_{sl}_sharp_target_only.npz') as a:cp=a['predictions']
            z-=z.max(-1,keepdims=True);p=np.exp(z);p/=p.sum(-1,keepdims=True)
            wrong=bp!=y;correct=~wrong
            masks={'all_wrong':wrong,'wrong_high_confidence_ge90':wrong&(p.max(-1)>=.9),
                   'wrong_lower_confidence':wrong&(p.max(-1)<.9),
                   'wrong_nearest_prototype_correct':wrong&(near==y),
                   'wrong_classifier_prototype_agree':wrong&(near==bp),
                   'wrong_neither_correct_disagree':wrong&(near!=bp)&(near!=y)}
            for group,mask in masks.items():
                n=int(mask.sum())
                records.append(dict(seed=seed,slice=sl,group=group,rows=n,
                    share_of_frozen_wrong_pct=100*n/int(wrong.sum()),
                    v5_recovery_pct=100*float((pp[mask]==y[mask]).mean()) if n else None,
                    v6_recovery_pct=100*float((cp[mask]==y[mask]).mean()) if n else None,
                    new_correct=int((mask&(pp!=y)&(cp==y)).sum()),
                    lost_correct=int((mask&(pp==y)&(cp!=y)).sum())))
    save('posthoc_error_groups.json',{'role':'after-score diagnostic, no method or threshold changes','records':records})
    summary=[]
    for sl in C['slices']:
        for group in sorted({r['group'] for r in records}):
            rr=[r for r in records if r['slice']==sl and r['group']==group]
            summary.append(dict(slice=sl,group=group,**{k:float(np.mean([r[k] for r in rr])) for k in ['share_of_frozen_wrong_pct','v5_recovery_pct','v6_recovery_pct','new_correct','lost_correct']}))
    save('posthoc_error_group_summary.json',summary)
    save('posthoc_analysis_audit.json',{'passed':True,'source':'saved audited predictions and frozen per-query geometry only','training_or_inference_changed':False,'query_labels':'posthoc diagnostics only','script_sha256':sha(RUN/'posthoc_diagnostics.py'),'groups':len(records)})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
