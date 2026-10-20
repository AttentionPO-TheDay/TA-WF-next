"""Terminal-label diagnostics. No thresholds from this file enter inference/training."""
import csv,json
from collections import defaultdict
import numpy as np
from runtime import RUN,ROOT,C,save,metrics,sha

SIGNALS=['combined']
def factor(s):return 'Version' if s.startswith('version') else 'Network' if s.startswith('network') else 'Behavior'
def aggregate(rows,key):
    factors=defaultdict(list)
    for r in rows:factors[factor(r['slice'])].append(r[key])
    return float(np.mean([np.mean(v) for v in factors.values()]))

def writecsv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (RUN/'artifacts'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def curve(cells,signal):
    # A single threshold per method/seed, shared across all included slices.
    nfactor=len({factor(v['slice']) for v in cells});counts={f:sum(factor(v['slice'])==f for v in cells) for f in {factor(v['slice']) for v in cells}}
    arrays=[];oracle=0.;raw_nar=0.;raw_rec=0.;eligible_rec=0.;eligible_nar=0.
    for v in cells:
        y,bp,cp,sc=v['y'],v['bp'],v['cp'],v['scores'][signal];correct=bp==y;wrong=~correct;change=cp!=bp
        wr=1/(nfactor*counts[factor(v['slice'])]);cn=max(correct.sum(),1);wn=max(wrong.sum(),1)
        damage=change&correct;recovered=change&wrong&(cp==y)
        raw_nar+=wr*damage.sum()/cn;raw_rec+=wr*recovered.sum()/wn;oracle+=wr*recovered.sum()/wn
        eligible=change & (sc>=0) if signal=='combined' else change
        eligible_rec+=wr*(recovered&eligible).sum()/wn;eligible_nar+=wr*(damage&eligible).sum()/cn
        arrays.append((sc[eligible],wr*damage[eligible]/cn,wr*recovered[eligible]/wn))
    s=np.concatenate([v[0] for v in arrays]);d=np.concatenate([v[1] for v in arrays]);r=np.concatenate([v[2] for v in arrays])
    if len(s):
        order=np.argsort(-s,kind='stable');s=s[order];d=d[order];r=r[order];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
        nar=np.r_[0.,np.cumsum(d)[ends]];rec=np.r_[0.,np.cumsum(r)[ends]];threshold=np.r_[np.inf,s[ends]]
    else:nar=np.array([0.]);rec=np.array([0.]);threshold=np.array([np.inf])
    assert np.all(np.diff(nar)>=-1e-12) and np.all(np.diff(rec)>=-1e-12)
    result={'raw_nar':raw_nar,'raw_recovery':raw_rec,'candidate_oracle_recovery':oracle,'eligible_recovery':eligible_rec,'eligible_nar':eligible_nar,'eligible_changes':len(s)}
    for cap in [.01,.02]:
        ix=np.flatnonzero(nar<=cap+1e-12)[-1];k=str(int(cap*100));result['rec_at_nar'+k]=float(rec[ix]);result['attained_nar'+k]=float(nar[ix]);result['diagnostic_threshold'+k]=float(threshold[ix]) if np.isfinite(threshold[ix]) else None
        result['random_rec_at_nar'+k]=eligible_rec*min(1,cap/eligible_nar) if eligible_nar>0 else eligible_rec
    for target in [.02,.04]:
        ix=np.flatnonzero(rec>=target-1e-12);result['nar_at_rec'+str(int(target*100))]=float(nar[ix[0]]) if len(ix) else None
    return result,(nar,rec,threshold)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rs=[]
    for seed in C['seeds']:rs+=json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text())
    assert len(rs)==30;save('scores.json',rs);controls=[];v3=ROOT/C['baseline_run'];v4=ROOT/C['v4_run'];oldrows=json.loads((v3/'artifacts/scores.json').read_text());q=[];summ=[];fr=[];curves={}
    methods=['frozen','v3_ungated_lora','v3_tent_reference','v4_soft_geometry']+C['conditions']
    for name in C['slices']:
        for method in methods:
            metricrows=[];frontierrows=[]
            for seed in C['seeds']:
                with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as a:base=a['predictions'];y=a['labels'];ids=a['rows']
                if method=='frozen':pred=base;score=np.zeros(len(pred));row={'seed':seed,'slice':name,'condition':method,'policy':'raw',**metrics(y,pred,base),'delta_f1_pp':0.}
                elif method in ['v3_ungated_lora','v3_tent_reference']:
                    cond='soft_neighbor_lora' if method=='v3_ungated_lora' else 'tent_bn_reference'
                    with np.load(v3/'artifacts'/f'raw_{seed}_{name}_{cond}.npz') as a:pred=a['predictions'];assert np.array_equal(ids,a['rows'])
                    sigcond='ungated_lora' if method=='v3_ungated_lora' else 'tent_reference'
                    with np.load(v4/'artifacts'/f'signal_{seed}_{name}_{sigcond}.npz') as a:score=a['combined']
                    row=next(r.copy() for r in oldrows if r['seed']==seed and r['slice']==name and r['condition']==cond and r['policy']=='raw');row['condition']=method;row['reused_control']=True
                elif method=='v4_soft_geometry':
                    oldr=json.loads((v4/'artifacts'/f'adapted_scores_{seed}.json').read_text());row=next(r.copy() for r in oldr if r['slice']==name and r['condition']=='soft_geometry' and r['policy']=='raw');row['condition']=method;row['reused_control']=True
                    with np.load(v4/'artifacts'/f'raw_{seed}_{name}_soft_geometry.npz') as a:pred=a['predictions'];assert np.array_equal(ids,a['rows'])
                    with np.load(v4/'artifacts'/f'signal_{seed}_{name}_soft_geometry.npz') as a:score=a['combined']
                else:
                    row=next(r for r in rs if r['seed']==seed and r['slice']==name and r['condition']==method)
                    with np.load(RUN/'artifacts'/f'raw_{seed}_{name}_{method}.npz') as a:pred=a['predictions'];assert np.array_equal(ids,a['rows'])
                    with np.load(RUN/'artifacts'/f'signal_{seed}_{name}_{method}.npz') as a:score=a['combined']
                if method not in C['conditions']:controls.append(row)
                mm=metrics(y,pred,base)
                for key in mm:assert abs(row[key]-mm[key])<1e-12
                metricrows.append(row)
                f,cv=curve([{'slice':name,'y':y,'bp':base,'cp':pred,'scores':{'combined':score}}],'combined');fr.append({'seed':seed,'slice':name,'condition':method,**f});frontierrows.append(f);curves[(name,method,seed)]=cv
                np.savez_compressed(RUN/'artifacts'/f'curve_{seed}_{name}_{method}.npz',nar=cv[0],recovery=cv[1],threshold=cv[2])
            summary={'slice':name,'condition':method}
            for k in ['macro_f1','accuracy','delta_f1_pp','nar','recovery']:
                a=np.array([r[k] for r in metricrows]);scale=1 if k=='delta_f1_pp' else 100;summary[k+'_mean']=float(a.mean())*scale;summary[k+'_sd']=float(a.std(ddof=1))*scale
            for k in ['rec_at_nar1','rec_at_nar2','candidate_oracle_recovery','random_rec_at_nar1','nar_at_rec2','nar_at_rec4']:
                a=[r[k] for r in frontierrows];summary[k+'_mean']=float(np.mean(a))*100 if all(v is not None for v in a) else None;summary[k+'_sd']=float(np.std(a,ddof=1))*100 if all(v is not None for v in a) else None
            summ.append(summary)
    save('controls.json',controls);save('summary.json',summ);save('frontier_cells.json',fr);writecsv('summary.csv',summ);writecsv('cells.csv',rs+controls);writecsv('frontier_cells.csv',fr)
    decisions=[]
    for name in C['slices']:
        joint=next(r for r in summ if r['slice']==name and r['condition']=='momentum_joint');static=next(r for r in summ if r['slice']==name and r['condition']=='full_static');tent=next(r for r in summ if r['slice']==name and r['condition']=='v3_tent_reference')
        seedvals=[r['delta_f1_pp'] for r in rs if r['slice']==name and r['condition']=='momentum_joint'];passed=joint['delta_f1_pp_mean']>=3 and min(seedvals)>0 and joint['rec_at_nar1_mean']>=static['rec_at_nar1_mean']
        decisions.append({'slice':name,'joint_delta_f1_pp':joint['delta_f1_pp_mean'],'joint_each_seed_delta_f1_pp':seedvals,'joint_minus_full_static_f1_pp':joint['macro_f1_mean']-static['macro_f1_mean'],'joint_minus_reference_tent_f1_pp':joint['macro_f1_mean']-tent['macro_f1_mean'],'joint_minus_static_recovery_at_nar1_pp':joint['rec_at_nar1_mean']-static['rec_at_nar1_mean'],'f1_target_passed':joint['delta_f1_pp_mean']>=3,'all_seed_positive':min(seedvals)>0,'matched_risk_not_worse':joint['rec_at_nar1_mean']>=static['rec_at_nar1_mean'],'slice_gate_passed':bool(passed)})
    decision={'expansion_eligible':all(r['slice_gate_passed'] for r in decisions),'slices':decisions,'posthoc_frontier_not_deployment_guarantee':True,'full_matrix_launched':False};save('registered_decisions.json',decision)
    lines=['# 两切片攻坚主表','', '全部为最终400步原始分类头输出，无硬旁路；均值±SD，3seed。历史对照明确标记v3/v4。','', '| Slice | Method | ΔF1 pp | NAR % | Recovery % | Rec@NAR≤1% % | Rec@NAR≤2% % |','|---|---|---:|---:|---:|---:|---:|']
    for r in summ:lines.append('| '+r['slice']+' | '+r['condition']+' | '+' | '.join(f"{r[k+'_mean']:.3f}±{r[k+'_sd']:.3f}" for k in ['delta_f1_pp','nar','recovery','rec_at_nar1','rec_at_nar2'])+' |')
    (RUN/'table_main.md').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));grid=np.linspace(0,.1,301)
    for ax,name in zip(axs,C['slices']):
        for method in methods[1:]:
            vals=[]
            for seed in C['seeds']:
                n,r,_=curves[(name,method,seed)];u=np.unique(n);ends=np.r_[np.flatnonzero(n[:-1]!=n[1:]),len(n)-1];rr=r[ends];ix=np.searchsorted(u,grid,side='right')-1;vals.append(rr[np.maximum(ix,0)])
            vals=np.array(vals)*100;ax.plot(grid*100,vals.mean(0),label=method);ax.fill_between(grid*100,vals.min(0),vals.max(0),alpha=.06)
        ax.set(title=name,xlabel='NAR (%)',ylabel='Recovery (%)',xlim=(0,10));ax.grid(alpha=.2);ax.axvline(1,color='gray',ls=':');ax.axvline(2,color='gray',ls=':')
    axs[1].legend(fontsize=6);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_frontier.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,name in zip(axs,C['slices']):
        for method in C['conditions']:
            histories=[json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{method}.json').read_text()) for seed in C['seeds']];arr=np.array([[h['prototype_drift_mean'] for h in hist] for hist in histories]);ax.plot(np.arange(1,401),arr.mean(0),label=method)
        ax.set(title=name,xlabel='Reference adaptation step',ylabel='Mean source-anchored prototype drift');ax.grid(alpha=.2)
    axs[1].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_prototype.{ext}',dpi=180)
    plt.close(fig)
    text=['# 动量原型与批内软对比 v5 结果','',f'30/30适应完成。预定联合arm扩展门槛：{"PASS" if decision["expansion_eligible"] else "FAIL"}；不按事后最优arm改写主条件。','', '[两切片完整主表](table_main.md)；[安全—纠错曲线](figure_frontier.png)；[原型漂移轨迹](figure_prototype.png)。','']
    for d in decisions:text.append(f"{d['slice']}：joint ΔF1={d['joint_delta_f1_pp']:+.3f}pp，较匹配full_static {d['joint_minus_full_static_f1_pp']:+.3f}pp，较ref-TENT {d['joint_minus_reference_tent_f1_pp']:+.3f}pp；同NAR1%纠错差={d['joint_minus_static_recovery_at_nar1_pp']:+.3f}pp；切片门槛={d['slice_gate_passed']}。")
    text+=['','全部五组使用完整102类目标，以full_static分离旧Top5支持变化的影响。动量权重始终>=.05；只更新LoRA，原始分类头直接输出，无frozen回退。原型与soft pair targets可能强化错误聚类，不能预设有益；source锚定与无query拟合不等于零负适应保证。','', '曲线使用目标标签事后定位阈值，只用于比较候选安全—纠错前沿，不是本轮部署策略，也不回流训练。query不参与EMA/prototype/记忆库，源valid仅终端评分。源和目标均为已观察诊断开发，不是未见确认；无WTT/AWF新评分。','', '重复单位为训练seed，不能将同一query的三个seed结果当独立样本；三seed仅报告描述性稳定性。源valid保留与所有step损失/权重/熵/类支持和原型漂移均在artifacts。历史基线不是新独立重复。', '', '预算固定30任务，未达到扩展门槛即停止，不用query返调参数继续冲分。']
    (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n');print((RUN/'table_main.md').read_text(),flush=True)

if __name__=='__main__':main()
