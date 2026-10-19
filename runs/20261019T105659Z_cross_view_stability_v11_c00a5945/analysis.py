import hashlib,json,csv
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,f1_score
from runtime import RUN,ROOT,C,save,read,old_logits

def metrics(y,p,b):
    bc=b==y;bad=~bc
    return dict(macro_f1=float(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)),accuracy=float((p==y).mean()),nar=float((p[bc]!=y[bc]).mean()) if bc.any() else 0.,recovery=float((p[bad]==y[bad]).mean()) if bad.any() else 0.)
def confusion_check(y,p,value):
    a=np.zeros((102,102),dtype=np.int64);np.add.at(a,(y,p),1);tp=np.diag(a);den=a.sum(0)+a.sum(1);v=np.divide(2*tp,den,out=np.zeros(102),where=den>0);assert abs(v.mean()-value['macro_f1'])<1e-12 and abs(tp.sum()/len(y)-value['accuracy'])<1e-12
def frontier(y,b,c,s):
    ix=np.flatnonzero(b!=c);ix=ix[np.argsort(-s[ix],kind='stable')];bc=b==y;bad=~bc
    ends=np.flatnonzero(np.r_[s[ix][1:]!=s[ix][:-1],True]) if len(ix) else np.array([],dtype=int)
    ns=np.r_[0.,(np.cumsum(bc[ix])/max(bc.sum(),1))[ends]];rs=np.r_[0.,(np.cumsum(bad[ix]&(c[ix]==y[ix]))/max(bad.sum(),1))[ends]]
    allowed=np.flatnonzero(ns<=.01);j=int(allowed[np.argmax(rs[allowed])]);threshold=float(s[ix][ends[j-1]]) if j else None;mask=(b!=c)&(s>=threshold) if threshold is not None else np.zeros(len(y),bool);pred=np.where(mask,c,b);met=metrics(y,pred,b);confusion_check(y,pred,met);assert abs(met['nar']-ns[j])<1e-12 and abs(met['recovery']-rs[j])<1e-12
    points=np.unique(np.r_[0,np.linspace(0,len(ns)-1,min(200,len(ns)),dtype=int),j]);return dict(rec_at_nar1=float(rs[j]),attained_nar1=float(ns[j]),diagnostic_threshold=threshold,diag_f1=met['macro_f1'],diag_accuracy=met['accuracy'],accepted_changes=int(mask.sum())),np.stack((ns[points],rs[points]),1),pred
def discrimination(y,b,c,s):
    change=b!=c;corr=(b!=y)&(c==y);damage=(b==y)&change;other=change&(b!=y)&(c!=y);ix=corr|damage
    a=float(roc_auc_score(corr[ix],s[ix])) if corr.any() and damage.any() else None
    # Full utility includes different-wrong changes among negatives.
    full=float(roc_auc_score(corr[change],s[change])) if corr.any() and (damage|other).any() else None
    return dict(correction_damage_auc=a,full_change_auc=full,corrections=int(corr.sum()),damages=int(damage.sum()),other_wrong_changes=int(other.sum()),changed_count=int(change.sum()),all_change_precision=float(corr.sum()/max(change.sum(),1)))
def probabilities(arm,role):
    out=[];rows=None
    for seed in C['seeds']:
        summary=json.loads((ROOT/C['v10_run']/'artifacts'/f'train_summary_{seed}.json').read_text());temp=next(r['temperature'] for r in summary if r['arm']==arm);v=[]
        for view in C['views']:
            a=old_logits(seed,arm,role) if view=='original' else read(RUN/'artifacts'/f'logits_{seed}_{arm}_{role}_{view}.npz')
            if rows is None:rows=a['rows']
            assert np.array_equal(rows,a['rows']);v.append(torch.from_numpy(a['logits']/temp).softmax(-1).numpy())
        out.append(np.stack(v))
    return np.stack(out),rows
def shuffled(s,rows,changed,key):
    ix=np.flatnonzero(changed);h=[hashlib.sha256((key+'|'+str(int(rows[i]))).encode()).digest() for i in ix];perm=np.array(sorted(range(len(ix)),key=lambda j:h[j]),dtype=int);out=s.copy();out[ix]=s[ix[perm]];assert np.array_equal(np.sort(out[ix]),np.sort(s[ix]));return out
def scores(probs,si,b,c,baseline,rows,key):
    n=len(b);ii=np.arange(n);contrast=probs[:,:,ii,c]-probs[:,:,ii,b];peer=[j for j in range(3) if j!=si]
    out=dict(baseline=baseline,perturb_support=contrast[si,1:].min(0),seed_support=contrast[peer,0].min(0),joint_support=contrast.min((0,1)))
    out['shuffled_joint']=shuffled(out['joint_support'],rows,b!=c,key);assert list(out)==C['scores'] and all(np.isfinite(s).all() for s in out.values());return out

def main():
    v9=ROOT/C['v9_run'];old9=json.loads((v9/'artifacts/frontier_cells.json').read_text());old10=json.loads((ROOT/C['v10_run']/'artifacts/frontier_cells.json').read_text());cells=[];invariance=[];curves={};references=[]
    for arm in C['arms']:
        for role in ['valid']+[f'{name}_{r}' for name in C['slices'] for r in ['reference','query']]:
            probs,rows=probabilities(arm,role)
            if role=='valid':
                for si,seed in enumerate(C['seeds']):
                    a=old_logits(seed,arm,role);y=a['labels'];orig=metrics(y,probs[si,0].argmax(-1),probs[si,0].argmax(-1))
                    for vi,view in enumerate(C['views']):
                        p=probs[si,vi].argmax(-1);met=metrics(y,p,probs[si,0].argmax(-1));confusion_check(y,p,met);invariance.append(dict(seed=seed,arm=arm,view=view,macro_f1=met['macro_f1'],drop_f1_pp=100*(orig['macro_f1']-met['macro_f1']),prediction_agreement=float((p==probs[si,0].argmax(-1)).mean())))
            if role.endswith('_reference'):
                references.append(dict(arm=arm,role=role,rows=len(rows),all_seed_view_top1_agreement=float((probs.argmax(-1)==probs[0,0].argmax(-1)[None,None]).mean()),fit=False));continue
            for name in C['slices'] if role=='valid' else [role.removesuffix('_query')]:
                partition='valid' if role=='valid' else 'query'
                for si,seed in enumerate(C['seeds']):
                    a=old_logits(seed,arm,role);y=a['labels'];base=read(v9/'artifacts'/f'{"valid" if partition=="valid" else "query"}_{seed}_{name}_frozen.npz');assert np.array_equal(base['labels'],y) and np.array_equal(base['rows'],rows);b=base['logits'].argmax(-1)
                    for pool in C['candidate_pools']:
                        if pool=='v10_view':
                            c=probs[si,0].argmax(-1);q=np.sort(probs[si,0],axis=-1);baseline=q[:,-1]-q[:,-2]
                        else:
                            prior=read(v9/'artifacts'/f'{partition}_{seed}_{name}_smooth_balance.npz');assert np.array_equal(prior['rows'],rows) and np.array_equal(prior['labels'],y);c=prior['logits'].argmax(-1);baseline=prior['margin_gain']
                        ss=scores(probs,si,b,c,baseline,rows,f'v11|{seed}|{name}|{partition}|{arm}|{pool}');common=None
                        if partition=='query':common=read(v9/'artifacts'/f'base_{seed}_{name}.npz')['common_wrong']
                        raw=metrics(y,c,b);confusion_check(y,c,raw);key=f'{partition}_{seed}_{name}_{arm}_{pool}';np.savez_compressed(RUN/'artifacts'/f'scores_{key}.npz',**ss,candidates=c,base=b,rows=rows,labels=y)
                        for score_name,s in ss.items():
                            met,curve,p=frontier(y,b,c,s);disc=discrimination(y,b,c,s);row=dict(seed=seed,slice=name,partition=partition,arm=arm,pool=pool,score=score_name,**met,**disc,raw_f1=raw['macro_f1'],raw_nar=raw['nar'],raw_recovery=raw['recovery'])
                            if common is not None:row['common_corrected_at_nar1']=float((p[common]==y[common]).mean())
                            if partition=='query' and score_name=='baseline':
                                previous=next(r for r in (old10 if pool=='v10_view' else old9) if r['seed']==seed and r['slice']==name and (r.get('arm')==arm and r.get('score')=='view_probability_margin' if pool=='v10_view' else r['condition']=='smooth_balance'))
                                assert abs(previous['rec_at_nar1']-met['rec_at_nar1'])<1e-12
                            cells.append(row);curves[f'{key}_{score_name}']=curve
            del probs
    summaries=[]
    for partition in ['valid','query']:
        for name in C['slices']:
            for arm in C['arms']:
                for pool in C['candidate_pools']:
                    for score in C['scores']:
                        rr=[r for r in cells if r['partition']==partition and r['slice']==name and r['arm']==arm and r['pool']==pool and r['score']==score];assert len(rr)==3;out=dict(partition=partition,slice=name,arm=arm,pool=pool,score=score)
                        for metric in ['rec_at_nar1','attained_nar1','diag_f1','correction_damage_auc','full_change_auc','all_change_precision','raw_f1','raw_nar','raw_recovery']+(['common_corrected_at_nar1'] if partition=='query' else []):
                            vals=[r[metric] for r in rr];scale=1 if metric.endswith('auc') else 100;out[metric+'_mean']=float(np.mean(vals)*scale) if all(x is not None for x in vals) else None;out[metric+'_sd']=float(np.std(vals,ddof=1)*scale) if all(x is not None for x in vals) else None
                        summaries.append(out)
    inv_summary=[]
    for arm in C['arms']:
        for view in C['views']:
            rr=[r for r in invariance if r['arm']==arm and r['view']==view];inv_summary.append(dict(arm=arm,view=view,drop_f1_pp=float(np.mean([r['drop_f1_pp'] for r in rr])),prediction_agreement=float(np.mean([r['prediction_agreement'] for r in rr]))))
    invpass=all(r['drop_f1_pp']<=2 for r in inv_summary if r['arm']==C['primary_arm']);decisions=[]
    for pool in C['candidate_pools']:
        per=[]
        for name in C['slices']:
            def row(s):return next(r for r in summaries if r['partition']=='query' and r['slice']==name and r['arm']==C['primary_arm'] and r['pool']==pool and r['score']==s)
            b,j,sh=row('baseline'),row('joint_support'),row('shuffled_joint');gain=j['rec_at_nar1_mean']-b['rec_at_nar1_mean'];au=j['correction_damage_auc_mean'];delta=au-b['correction_damage_auc_mean'] if au is not None and b['correction_damage_auc_mean'] is not None else None;shuffle=j['rec_at_nar1_mean']-sh['rec_at_nar1_mean'];old=float(np.mean([x['rec_at_nar1'] for x in old9 if x['condition']=='smooth_balance' and x['slice']==name])*100)
            per.append(dict(slice=name,rec1_gain_pp=gain,auc=au,auc_gain=delta,rec1_vs_shuffled_pp=shuffle,rec1_vs_v9_pp=j['rec_at_nar1_mean']-old,passed=bool(gain>=.5 and au is not None and au>=.65 and delta is not None and delta>=.03 and shuffle>=.5)))
        decisions.append(dict(pool=pool,passed=bool(invpass and all(r['passed'] for r in per)),slices=per))
    decision=dict(primary_arm=C['primary_arm'],primary_pool=C['primary_pool'],source_invariance_passed=invpass,primary_passed=decisions[0]['passed'],utility_passed=bool(decisions[0]['passed'] and all(r['rec1_vs_v9_pp']>=.2 for r in decisions[0]['slices'])),pools=decisions,tta_integrated=False,new_training_steps=0)
    for n,v in [('cells.json',cells),('summary.json',summaries),('invariance_cells.json',invariance),('invariance_summary.json',inv_summary),('registered_decisions.json',decision),('reference_diagnostics.json',references)]:save(n,v)
    for name,data in [('cells',cells),('summary',summaries),('invariance',invariance)]:
        keys=list(dict.fromkeys(k for r in data for k in r))
        with (RUN/'artifacts'/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
    np.savez_compressed(RUN/'artifacts/frontier_curves.npz',**curves);save('metric_audit.json',dict(passed=True,metric_states=len(cells),source_perturbation_states=len(invariance),baseline_frontiers_exact=36,confusion_and_threshold_ties_checked=True,changed_score_multiset_preserved=True,query_fit=False))
    lines=['# 跨视图稳定性 v11','',f'主候选门槛：{decision["primary_passed"]}；源不变性：{invpass}；相对v9可利用性：{decision["utility_passed"]}。零新训练、零TTA整合。','', '所有结果为两个已观察诊断切片，三seed均值；同NAR1阈值由query标签定位，不能部署。主视图direction_iat由v10反馈选择，明确为新一轮开发；不能当作独立确认。候选类别始终不变，单独隔离排序信号作用。','', '|候选池|切片|评分|Rec@NAR1 %|较同池baseline pp|纠错vs损伤AUROC|完整改写AUROC|','|---|---|---|---:|---:|---:|---:|']
    for pool in C['candidate_pools']:
        for name in C['slices']:
            rr=[r for r in summaries if r['partition']=='query' and r['slice']==name and r['arm']==C['primary_arm'] and r['pool']==pool];base=next(r for r in rr if r['score']=='baseline')
            for r in rr:
                f=lambda v:f'{v:.3f}' if v is not None else '—';lines.append('|'+ '|'.join([pool,name,r['score'],f(r['rec_at_nar1_mean']),f(r['rec_at_nar1_mean']-base['rec_at_nar1_mean']),f(r['correction_damage_auc_mean']),f(r['full_change_auc_mean'])])+'|')
    (RUN/'table_main.md').write_text('\n'.join(lines[6:])+'\n')
    lines+=['','各arm源valid不变性（drop正数=下降；不按目标表现删除扰动）：','', '|视图模型|扰动|F1下降 pp|原预测保持 %|','|---|---|---:|---:|']
    for r in inv_summary:lines.append(f'|{r["arm"]}|{r["view"]}|{r["drop_f1_pp"]:.3f}|{100*r["prediction_agreement"]:.3f}|')
    lines+=['','预注册判决：','', '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','', '完整逐seed、控制arm及源valid排序结果见artifacts/cells.csv与summary.csv，未只挑最好切片。条件AUROC只含真正纠错与损伤；完整改写AUROC另外包含旧错→其他错。对候选池同分阈值不拆开，零改写始终可选。未更新校准温度或风险阈值。跨seed/扰动稳定只提供一致性证据，仍可能稳定地错误；源不变性不能证明目标不变性。sourcevalid各切片行重复用于诊断，不能当作独立重复。','', '复用9个v10模型的额外源监督预算依然属于方法成本；本轮新增推理5倍/跨seed3倍，不能只报无参数而隐去推理成本。另一候选池v9smooth是历史checkpoint固定预测，raw输出性能未因本实验提高，任何Rec@NAR1收益只是评分前沿诊断。没有访问WTT/AWF，没有额外目标标签训练或改动既有封存。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(5)
    for ax,name in zip(axes,C['slices']):
        for pi,pool in enumerate(C['candidate_pools']):
            ys=[next(r['rec_at_nar1_mean'] for r in summaries if r['partition']=='query' and r['slice']==name and r['arm']==C['primary_arm'] and r['pool']==pool and r['score']==s) for s in C['scores']];ax.bar(x+(pi-.5)*.36,ys,width=.36,label=pool)
        ax.set_xticks(x,C['scores'],rotation=20);ax.set_title(name);ax.set_ylabel('Recovery at NAR <= 1% (diagnostic, %)');ax.legend()
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_ranking.{ext}',dpi=170)
    plt.close(fig)
if __name__=='__main__':main()
