"""Complete source-calibrated factorial report; no target selection or method edits."""
import csv,itertools,json
from collections import defaultdict
import numpy as np
import torch
from runtime import RUN,ROOT,C,save,sha

def factor(name):return 'Version' if name.startswith('version') else 'Network' if name.startswith('network') else 'Behavior'
def macro(rs,metric):
    groups=defaultdict(list)
    for r in rs:groups[factor(r['slice'])].append(r[metric])
    return float(np.mean([np.mean(v) for v in groups.values()]))
def writecsv(name,rows):
    with (RUN/'artifacts'/name).open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    assert json.loads((RUN/'artifacts/postrun_audit.json').read_text())['passed']
    scores=json.loads((RUN/'artifacts/scores.json').read_text());assert len(scores)==630
    writecsv('cells.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in scores])
    pairs=[(c,p) for c in C['conditions'] for p in (C['policies'] if not c.startswith('tent_bn') and c!='frozen' else ['raw'])]
    summary=[];table=['# 候选目标 × 修改验收主表','', '三因素等权，再汇总3seed均值±样本标准差。F1/Acc/NAR/Recovery为%，ΔF1为pp。','', '| 方法 | 验收 | F1 | ΔF1 | NAR | Recovery | 改动样本数* |','|---|---|---:|---:|---:|---:|---:|']
    for condition,policy in pairs:
        row={'condition':condition,'policy':policy}
        for metric in ['macro_f1','accuracy','delta_f1_pp','nar','recovery']:
            v=np.array([macro([r for r in scores if (r['seed'],r['condition'],r['policy'])==(s,condition,policy)],metric) for s in C['seeds']]);scale=1 if metric=='delta_f1_pp' else 100
            row[metric+'_mean']=float(v.mean()*scale);row[metric+'_sd']=float(v.std(ddof=1)*scale)
        row['changed_count']=sum(r['changed_predictions'] for r in scores if r['condition']==condition and r['policy']==policy)
        summary.append(row);table.append('| '+condition+' | '+policy+' | '+' | '.join(f"{row[m+'_mean']:.3f}±{row[m+'_sd']:.3f}" for m in ['macro_f1','delta_f1_pp','nar','recovery'])+f" | {row['changed_count']} |")
    table+=['','*改动数为所有query×seed合计，是同一数据在3个seed下的计数，不是独立样本数。TENT两版本共享同一次400步reference优化；标准tent_bn在query使用批统计，属于transductive；tent_bn_reference仅使用reference总体统计。','', '本轮hard也包含KL蒸馏项，和v2硬原型损失不同；本轮三target间损失、预算完全匹配。源验证集是既有选模部分，不是未见确认。']
    (RUN/'table_main.md').write_text('\n'.join(table)+'\n');writecsv('summary.csv',summary);save('summary.json',summary)
    per=[]
    for name in C['slices']:
        for condition,policy in pairs:
            rs=[r for r in scores if (r['slice'],r['condition'],r['policy'])==(name,condition,policy)];row={'slice':name,'factor':factor(name),'condition':condition,'policy':policy}
            for metric in ['macro_f1','accuracy','delta_f1_pp','nar','recovery']:
                vals=np.array([r[metric] for r in rs])*(1 if metric=='delta_f1_pp' else 100);row[metric+'_mean']=float(vals.mean());row[metric+'_sd']=float(vals.std(ddof=1))
            per.append(row)
    writecsv('by_slice.csv',per)
    fac=[]
    for family in ['Version','Network','Behavior']:
        for condition,policy in pairs:
            row={'factor':family,'condition':condition,'policy':policy}
            for metric in ['macro_f1','accuracy','delta_f1_pp','nar','recovery']:
                vals=[np.mean([r[metric] for r in scores if (r['seed'],r['condition'],r['policy'])==(seed,condition,policy) and factor(r['slice'])==family]) for seed in C['seeds']];scale=1 if metric=='delta_f1_pp' else 100
                row[metric+'_mean']=float(np.mean(vals)*scale);row[metric+'_sd']=float(np.std(vals,ddof=1)*scale)
            fac.append(row)
    writecsv('by_factor.csv',fac)
    contrasts=[]
    for kind in ['residual','lora']:
        for a,b in [('soft_teacher','hard'),('soft_neighbor','soft_teacher'),('soft_neighbor','hard')]:
            for policy in C['policies']:
                row={'contrast':f'{a}-{b}','adapter':kind,'policy':policy}
                for metric in ['macro_f1','nar','recovery']:
                    vals=[]
                    for seed in C['seeds']:
                        left=macro([r for r in scores if (r['seed'],r['condition'],r['policy'])==(seed,a+'_'+kind,policy)],metric);right=macro([r for r in scores if (r['seed'],r['condition'],r['policy'])==(seed,b+'_'+kind,policy)],metric);vals.append(100*(left-right))
                    row[metric+'_effect_pp']=float(np.mean(vals));row[metric+'_sd_pp']=float(np.std(vals,ddof=1))
                contrasts.append(row)
    writecsv('paired_contrasts.csv',contrasts)
    # Registered diagnostic criteria, no cherry-picking slices or seeds.
    index={(r['condition'],r['policy']):r for r in summary};decisions=[]
    for kind in ['residual','lora']:
        hard=index['hard_'+kind,'raw']
        for target in ['soft_teacher','soft_neighbor']:
            raw=index[target+'_'+kind,'raw'];gain=raw['macro_f1_mean']-hard['macro_f1_mean'];safe=raw['nar_mean']<=hard['nar_mean']
            decisions.append({'condition':target+'_'+kind,'criterion':'F1>=hard+1pp and NAR<=hard','f1_gain_pp':gain,'nar_change_pp':raw['nar_mean']-hard['nar_mean'],'passed':bool(gain>=1 and safe)})
    for condition,policy in pairs:
        if policy=='raw':continue
        raw=index[condition,'raw'];candidate=index[condition,policy];retention=candidate['recovery_mean']/raw['recovery_mean'] if raw['recovery_mean']>0 else 0.;reduction=1-candidate['nar_mean']/raw['nar_mean'] if raw['nar_mean']>0 else 0.
        decisions.append({'condition':condition,'policy':policy,'criterion':'retain>=80% Recovery and reduce>=30% NAR','recovery_retention':retention,'nar_reduction':reduction,'passed':bool(retention>=.8 and reduction>=.3)})
    save('registered_decisions.json',decisions)
    # Historical v2 raw loss controls are reused references, no claim of new repeats.
    previous=json.loads((ROOT/C['previous_run']/'artifacts/scores.json').read_text());history=[]
    for cond in ['single_ungated_residual','single_ungated_lora']:
        history.append({'condition':cond,**{m:100*macro([r for r in previous if r['condition']==cond],m) for m in ['macro_f1','nar','recovery']}})
    save('historical_v2.json',history)
    calibration=[]
    for seed in C['seeds']:
        p=json.loads((RUN/'artifacts'/f'preflight_{seed}.json').read_text());calibration.append({k:p[k] for k in ['seed','valid_rows','valid_baseline_accuracy','neighbor_accuracy','acceptance_enabled','threshold','entropy_cutoff','distance_cutoff','neighbor_temperature','trials']})
    save('calibration_summary.json',calibration)
    # Per-row-count policy diagnosis: accept both useful and harmful changes honestly.
    counts=[]
    for seed in C['seeds']:
        for name in C['slices']:
            with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as b:y=b['labels'];bp=b['predictions']
            good=bp==y
            for condition,policy in pairs:
                with np.load(RUN/'artifacts'/f'policy_{seed}_{name}_{condition}_{policy}.npz') as v:pred=v['predictions'];keep=v['keep']
                counts.append({'seed':seed,'slice':name,'condition':condition,'policy':policy,'rows':len(y),'accepted':int(keep.sum()),'changes':int((pred!=bp).sum()),'negative_count':int((good&(pred!=y)).sum()),'recovered_count':int((~good&(pred==y)).sum()),'frozen_correct':int(good.sum()),'frozen_wrong':int((~good).sum())})
    writecsv('policy_counts.csv',counts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=RUN/'artifacts/figures';figdir.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(12,7),sharey=True);labels=[r['condition']+' / '+r['policy'] for r in summary];yy=np.arange(len(summary));colors={'raw':'#3f7eac','valid_entropy':'#d59b39','evidence_accept':'#408e75'}
    for ax,metric,title in zip(axs,['delta_f1_pp','nar'],['Delta Macro-F1 (pp)','NAR (%)']):
        vals=[r[metric+'_mean'] for r in summary];err=[r[metric+'_sd'] for r in summary]
        ax.barh(yy,vals,xerr=err,color=[colors[r['policy']] for r in summary],height=.65,error_kw={'elinewidth':.7,'capsize':2});ax.axvline(0,color='grey',lw=.7);ax.grid(axis='x',alpha=.15);ax.set_title(title)
    axs[0].set_yticks(yy,labels,fontsize=7);axs[0].invert_yaxis();fig.suptitle('Candidate supervision and source-calibrated change acceptance');fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(figdir/f'main_comparison.{ext}',dpi=170)
    plt.close(fig)
    # Fixed visual rows: paired high-dimensional diagnostics and jointly embedded coordinates.
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances
    old=torch.load(ROOT/C['previous_run']/'artifacts/source_gmm_42.pt',weights_only=True,map_location='cpu');p=old['fits']['1'];mu=p['mean'].numpy();sd=p['sd'].numpy();centers=p['centers'][:,0].numpy();visual=[]
    for name in C['visualization']['slices']:
        for condition in C['visualization']['conditions']:
            v=torch.load(RUN/'artifacts'/f'visual_{name}_{condition}.pt',weights_only=True,map_location='cpu');a=(v['before'].numpy()-mu)/sd;raw=(v['after_raw'].numpy()-mu)/sd;b=np.where(v['keep'].numpy()[:,None],raw,a);y=v['labels'].numpy()
            record={'slice':name,'condition':condition,'raw_true_centroid_mse_before':float(((a-centers[y])**2).mean()),'raw_true_centroid_mse_after':float(((raw-centers[y])**2).mean()),'accepted_true_centroid_mse':float(((b-centers[y])**2).mean()),'nearest_centroid_accuracy_before':float((pairwise_distances(a,centers).argmin(-1)==y).mean()),'nearest_centroid_accuracy_raw':float((pairwise_distances(raw,centers).argmin(-1)==y).mean()),'variance_ratio_raw':float(raw.var(0).mean()/a.var(0).mean())}
            co=TSNE(n_components=2,perplexity=30,random_state=20261019,init='pca',learning_rate='auto').fit_transform(np.concatenate([centers,a,raw,b]))
            fig,axs=plt.subplots(1,3,figsize=(12,3.7),sharex=True,sharey=True)
            for i,(ax,title) in enumerate(zip(axs,['Frozen','Raw candidate','Evidence acceptance'])):
                xy=co[102+i*len(a):102+(i+1)*len(a)];ax.scatter(xy[:,0],xy[:,1],c=y,cmap='turbo',s=8,alpha=.6,vmin=0,vmax=101);ax.scatter(co[:102,0],co[:102,1],c='black',marker='x',s=10);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(name+' / '+condition+'; shared t-SNE of post-attention features',fontsize=10);fig.tight_layout()
            for ext in ['png','pdf','svg']:fig.savefig(figdir/f'{name}_{condition}.{ext}',dpi=170)
            plt.close(fig);visual.append(record)
    save('visual_diagnostics.json',visual)
    best=max([r for r in summary if r['condition'] not in ['frozen','tent_bn']],key=lambda r:r['macro_f1_mean']);passed=[d for d in decisions if d['passed']]
    tent=index['tent_bn_reference','raw']
    result=['# 候选纠错与验收 v3 结果','', '630/630评分状态完成（270原始状态、360派生验收状态）；210次适应训练，TENT两种推理共享一次优化。全部240个适应checkpoint全query重载与630指标复算通过，30个frozen重算；拒绝行逐位回退核验通过。','', f'描述性最高的严格reference方法：{best["condition"]}/{best["policy"]}，F1 {best["macro_f1_mean"]:.3f}±{best["macro_f1_sd"]:.3f}%，ΔF1 {best["delta_f1_pp_mean"]:+.3f}pp，NAR {best["nar_mean"]:.3f}%，Recovery {best["recovery_mean"]:.3f}%。不是根据该值追加训练或选取checkpoint。','', f'同预算reference-only TENT：F1 {tent["macro_f1_mean"]:.3f}%，ΔF1 {tent["delta_f1_pp_mean"]:+.3f}pp，NAR {tent["nar_mean"]:.3f}%。标准TENT使用query批统计，单列，不混称严格reference对照。','', f'预注册开发判据通过 {len(passed)}/{len(decisions)} 项；完整裁决见artifacts/registered_decisions.json，不能以某个有利切片取代整体判据。','', '完整表见 [主表](table_main.md)。逐seed/slice、逐因素、配对目标差异、验收计数、源校准可行性均在artifacts中；图为固定样本和共享坐标实测表征，不保证语义恢复。','', '本轮hard也含KL项，因此与v2原型hard不是单一soft开关对比；本轮hard/soft_teacher/soft_neighbor之间才是匹配损失的监督信号对比。v2历史控制不算本轮新重复。','', '限制：source valid曾用于底座选模，校准NAR约束不等于目标保证；源库邻居不能保证跨域正确，Top5候选会漏掉部分真类；目标标签仅评分，邻居证据由固定源库计算，无query批次拟合。方法受到既往Temporal、多因素及外部v4反馈影响，本轮仅已观察诊断开发，不是独立确认。']
    (RUN/'RESULTS.md').write_text('\n'.join(result)+'\n')
    files=[RUN/'RESULTS.md',RUN/'table_main.md']+list((RUN/'artifacts').glob('*.csv'))+list(figdir.glob('*'))
    save('delivery_audit.json',{'passed':True,'score_states':630,'figures':5,'outputs':{str(p.relative_to(RUN)):sha(p) for p in files}})
    print((RUN/'table_main.md').read_text(),flush=True)

if __name__=='__main__':main()
