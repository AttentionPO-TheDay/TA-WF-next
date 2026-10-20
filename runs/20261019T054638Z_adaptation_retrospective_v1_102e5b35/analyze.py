"""CPU-only reconstruction of saved evidence; no model import, forward or fitting."""
import hashlib,json,time,os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text());START=time.monotonic()
torch.set_num_threads(2)
INPUTS={};MANIFEST={};COUNTS={'files_verified_against_delivery':0}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def use(p):
    p=Path(p);rel=str(p.relative_to(ROOT))
    if rel not in INPUTS:
        hh=sha(p);INPUTS[rel]=hh
        for version,location in C['runs'].items():
            parent=ROOT/location
            if p.is_relative_to(parent):
                if version not in MANIFEST:
                    dp=parent/'artifacts/delivery_audit.json'
                    MANIFEST[version]=json.loads(dp.read_text())['outputs'];INPUTS[str(dp.relative_to(ROOT))]=sha(dp)
                key=str(p.relative_to(parent))
                if key in MANIFEST[version]:assert MANIFEST[version][key]==hh,str(p);COUNTS['files_verified_against_delivery']+=1
    if time.monotonic()-START>C['budget']['max_wall_seconds']:raise TimeoutError('analysis budget')
    return p

def js(p):return json.loads(use(p).read_text())
def arr(p):
    with np.load(use(p)) as a:return {k:a[k] for k in a.files}
def save(name,x):(RUN/'artifacts'/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def softmax(x):
    v=x-x.max(-1,keepdims=True);p=np.exp(v);return p/p.sum(-1,keepdims=True)
def ent(p):return -(p*np.log(np.maximum(p,1e-30))).sum(-1)
def pct(x,mask=None):return float(x[mask].mean()*100) if mask is not None and mask.any() else (float(x.mean()*100) if mask is None else None)
def metric(y,p,b):
    cm=np.zeros((102,102),dtype=int);np.add.at(cm,(y,p),1);den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0).mean()
    return dict(macro_f1=float(f),accuracy=float((y==p).mean()),nar=float((p[b==y]!=y[b==y]).mean()),recovery=float((p[b!=y]==y[b!=y]).mean()))

def main():
    # Freeze intended scope and report-code hash before reading diagnostic values.
    save('analysis_spec_freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [RUN/'PLAN.md',RUN/'config.json',RUN/'analyze.py']})
    for version,location in C['runs'].items():
        r=ROOT/location
        for name in ['PLAN.md','RESULTS.md','config.json']:
            use(r/name)
        assert js(r/'artifacts/postrun_audit.json')['passed']
        for file in ['methods.py','candidate.py','routing.py','momentum.py','refinement.py','REPAIR_HISTORY.md']:
            if (r/file).exists():use(r/file)
    # Historical all-slice means are kept separate from the two-slice mechanism comparison.
    summaries={}
    for version in ['v3','v4','v5','v6']:
        rr=ROOT/C['runs'][version]
        for filename in ['summary.json','scores.json']:
            if (rr/'artifacts'/filename).exists():summaries[version+'_'+filename]=js(rr/'artifacts'/filename)
    cells=[];evidence=[];sets=[];training=[];memory=[];paired=[];bootstrap=[]
    method_specs=[('v3','soft_neighbor_lora'),('v3','tent_bn_reference'),('v4','soft_geometry')]
    method_specs += [('v5',a) for a in js(ROOT/C['runs']['v5']/'config.json')['conditions']]
    method_specs += [('v6',a) for a in js(ROOT/C['runs']['v6']/'config.json')['conditions']]
    for seed in C['seeds']:
        for sl in C['slices']:
            base=arr(ROOT/C['runs']['v6']/'artifacts'/f'base_{seed}_{sl}.npz');y=base['labels'];bp=base['predictions'];bz=base['logits'].astype(np.float64);prob=softmax(bz)
            bmet=metric(y,bp,bp);wrong=bp!=y;correct=~wrong
            ev=arr(ROOT/C['runs']['v3']/'artifacts'/f'evidence_{seed}_{sl}.npz');votes=ev['votes'].astype(np.float64);bank=votes.argmax(-1)
            geo=arr(ROOT/C['runs']['v6']/'artifacts'/f'geometry_{seed}_{sl}.npz');near=geo['nearest'];v=votes.mean(0)
            q=softmax(bz/2+np.log(np.maximum(v,1e-12)));qp=q.argmax(-1);sort=np.sort(q,axis=-1)
            lock=(sort[:,-1]-sort[:,-2])>1/3
            top5=np.argpartition(-bz,4,axis=-1)[:,:5];hit5=(top5==y[:,None]).any(-1)
            bankagree=bank[0]==bank[1];samewrong=(bank[0]==bank[1])&(bank[0]!=y)
            row=dict(seed=seed,slice=sl,query_rows=len(y),baseline_f1=100*bmet['macro_f1'],frozen_wrong=int(wrong.sum()),
                frozen_wrong_top5_contains_true_pct=pct(hit5,wrong),frozen_wrong_geo_correct_pct=pct(near==y,wrong),
                frozen_wrong_head_geo_agree_pct=pct(near==bp,wrong),
                frozen_wrong_both_banks_same_wrong_pct=pct(samewrong,wrong),
                frozen_wrong_both_banks_correct_pct=pct((bank[0]==y)&(bank[1]==y),wrong),
                frozen_wrong_either_bank_correct_pct=pct((bank[0]==y)|(bank[1]==y),wrong),
                bank_agreement_all_pct=pct(bankagree),bank_agreement_wrong_pct=pct(bankagree,wrong),
                qfixed_query_accuracy_pct=pct(qp==y),qfixed_rescue_of_frozen_wrong_pct=pct(qp==y,wrong),
                qfixed_damages_frozen_correct_pct=pct(qp!=y,correct),
                qfixed_query_frozen_wrong_top1_stays_frozen_pct=pct(qp==bp,wrong),
                qfixed_wrong_query_mixture_cannot_switch_class_pct=pct(lock,qp!=y),
                frozen_wrong_qfixed_wrong_and_locked_pct=pct((qp!=y)&lock,wrong),
                frozen_wrong_confidence_ge90_pct=pct(prob.max(-1)>=.9,wrong))
            evidence.append(row);preds={};risks={}
            for version,arm in method_specs:
                r=ROOT/C['runs'][version];name=version+'_'+arm
                raw=arr(r/'artifacts'/f'raw_{seed}_{sl}_{arm}.npz');assert np.array_equal(raw['rows'],base['rows']) and np.array_equal(raw['labels'],y)
                cp=raw['predictions'];z=raw['logits'].astype(np.float64);preds[name]=cp
                recovered=wrong&(cp==y);damaged=correct&(cp!=y);changed=cp!=bp
                met=metric(y,cp,bp)
                if version in ['v3','v5','v6']:
                    rr=next(a for a in summaries[version+'_scores.json'] if a['seed']==seed and a['slice']==sl and a['condition']==arm and a.get('policy','raw')=='raw')
                    for key in met:assert abs(rr[key]-met[key])<1e-12
                norm=np.linalg.norm(z,axis=-1);bnorm=np.linalg.norm(bz,axis=-1)
                item=dict(seed=seed,slice=sl,method=name,**met,delta_f1_pp=100*(met['macro_f1']-bmet['macro_f1']),
                    changed=int(changed.sum()),recovered=int(recovered.sum()),damaged=int(damaged.sum()),
                    wrong_to_other_wrong=int((wrong&changed&(cp!=y)).sum()),
                    mean_logit_norm=float(norm.mean()),frozen_mean_logit_norm=float(bnorm.mean()),
                    logit_norm_ratio=float(norm.mean()/bnorm.mean()),entropy_mean=float(ent(softmax(z)).mean()))
                if version in ['v4','v5','v6']:
                    sig=arr(r/'artifacts'/f'signal_{seed}_{sl}_{arm}.npz');sc=sig['combined'];keep=sc>=0
                    mask=recovered|damaged
                    item.update(score_rec_vs_damage_auc=float(roc_auc_score(recovered[mask],sc[mask])) if recovered.any() and damaged.any() else None,
                        recovered_rejected_by_distance_pct=pct(~keep,recovered),
                        damaged_rejected_by_distance_pct=pct(~keep,damaged))
                    risks[name]=sc
                cells.append(item)
                if version in ['v5','v6']:
                    hist=js(r/'artifacts'/f'updates_{seed}_{sl}_{arm}.json')
                    tr=dict(seed=seed,slice=sl,method=name,steps=len(hist),reference_rows=1024,reference_passes=len(hist)*64/1024,
                        clipped_fraction=float(np.mean([h['gradient_norm']>1 for h in hist])))
                    for key in ['loss','kl','pull','relation','identity','contrast','gradient_norm','weight_mean','effective_samples','student_entropy','target_entropy','target_teacher_disagreement','prototype_drift_mean','classes']:
                        if key in hist[0]:
                            tr[key+'_first20']=float(np.mean([h[key] for h in hist[:20]]));tr[key+'_last20']=float(np.mean([h[key] for h in hist[-20:]]))
                    training.append(tr)
                    if arm in ['momentum_proto','momentum_joint','sharp_target_only']:
                        state=torch.load(use(r/'checkpoints'/f'{seed}_{sl}_{arm}.pt'),map_location='cpu',weights_only=True)['reference_state']
                        pr=state['probabilities'];cnt=torch.bincount(pr.argmax(-1),minlength=102)
                        src=state['source'];proto=.5*(src+state['prototype'])
                        qgeo=(torch.nn.functional.normalize(state['features'],dim=-1)@torch.nn.functional.normalize(proto,dim=-1).T/(.05 if arm=='sharp_target_only' else .1)).softmax(-1)
                        memory.append(dict(seed=seed,slice=sl,method=name,reference_memory_rows=len(pr),
                            memory_predicted_classes_present=int((cnt>0).sum()),memory_classes_below5_rows=int((cnt<5).sum()),
                            memory_classes_below2_rows=int((cnt<2).sum()),memory_max_class_rows=int(cnt.max()),
                            mean_class_pair_distance_ratio=float(torch.pdist(proto).mean()/torch.pdist(src).mean()),
                            geometric_target_entropy=float(-(qgeo*qgeo.clamp_min(1e-12).log()).sum(-1).mean())))
            a=preds['v5_momentum_proto'];b=preds['v6_sharp_target_only'];t=preds['v3_tent_bn_reference']
            rec_a=wrong&(a==y);rec_b=wrong&(b==y);rec_t=wrong&(t==y)
            da=correct&(a!=y);db=correct&(b!=y)
            sets.append(dict(seed=seed,slice=sl,v5_recovered=int(rec_a.sum()),v6_recovered=int(rec_b.sum()),
                v6_extra_recoveries=int((rec_b&~rec_a).sum()),v6_lost_recoveries=int((rec_a&~rec_b).sum()),
                v5_v6_recovery_jaccard=float((rec_a&rec_b).sum()/(rec_a|rec_b).sum()),
                tent_extra_vs_v6_recoveries=int((rec_t&~rec_b).sum()),v6_extra_vs_tent_recoveries=int((rec_b&~rec_t).sum()),
                oracle_union_tent_v6_recovery_pct=pct(rec_t|rec_b,wrong),v6_raw_recovery_pct=pct(rec_b,wrong),
                v6_new_damages=int((db&~da).sum()),v6_avoided_damages=int((da&~db).sum())))
    save('cells.json',cells);save('evidence.json',evidence);save('training_trajectories.json',training);save('reference_memory.json',memory);save('correction_overlap.json',sets)
    def summarize(records,groups):
        out=[];keys=list(dict.fromkeys(k for r in records for k,v in r.items() if isinstance(v,(int,float)) and k!='seed'))
        for vals in sorted({tuple(r[k] for k in groups) for r in records}):
            rr=[r for r in records if all(r[k]==v for k,v in zip(groups,vals))];d=dict(zip(groups,vals))
            for k in keys:
                vv=[r[k] for r in rr if r.get(k) is not None]
                if vv:d[k+'_mean']=float(np.mean(vv));d[k+'_sd']=float(np.std(vv,ddof=1)) if len(vv)>1 else None
            out.append(d)
        return out
    for name,data,group in [('cell_summary',cells,['slice','method']),('evidence_summary',evidence,['slice']),('trajectory_summary',training,['slice','method']),('memory_summary',memory,['slice','method']),('overlap_summary',sets,['slice'])]:save(name+'.json',summarize(data,group))
    for rel,h in INPUTS.items():assert sha(ROOT/rel)==h,rel
    save('inputs_sha256.json',INPUTS)
    elapsed=time.monotonic()-START
    save('analysis_audit.json',dict(passed=True,seconds=elapsed,input_files=len(INPUTS),**COUNTS,
        metric_cells_independently_recomputed=len(cells),new_training=0,new_adaptation=0,new_model_forward=0,physical_gpu_use=False,
        reference_true_labels_read=False,source_runs_unchanged=True))
    print(json.dumps(dict(passed=True,seconds=elapsed,input_files=len(INPUTS),metric_cells=len(cells))))

if __name__=='__main__':main()
