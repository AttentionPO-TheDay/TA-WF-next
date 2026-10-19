"""Registered primary/secondary decisions, full tables and error-transfer diagnostics."""
import json
import numpy as np
from runtime import RUN,ROOT,C,save
from frontier_base import curve,writecsv
from scoring import read
from audit import policy_audit

RAWKEYS=['macro_f1','accuracy','delta_f1_pp','nar','recovery','source_valid_macro_f1','cluster_purity','cluster_ari','cluster_nmi','source_prototype_accuracy','common_same_wrong','common_corrected','common_other_wrong']

def aggregate(rows,keys):
    out={}
    for k in keys:
        a=np.array([r[k] for r in rows]);scale=1 if k=='delta_f1_pp' else 100;out[k+'_mean']=float(a.mean())*scale;out[k+'_sd']=float(a.std(ddof=1))*scale
    return out


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policy_audit();rows=sum([json.loads((RUN/'artifacts'/f'scores_{s}.json').read_text()) for s in C['seeds']],[]);assert len(rows)==30;save('scores.json',rows);writecsv('cells.csv',rows)
    methods=['frozen','v6_lora']+C['conditions'];summ=[];front=[];curves={}
    for name in C['slices']:
        for condition in methods:
            rr=[v for v in rows if v['slice']==name and v['condition']==condition];ff=[]
            for seed in C['seeds']:
                d=read(RUN/'artifacts'/f'query_{seed}_{name}_{condition}.npz');y=d['labels'];bp=d['base_logits'].argmax(-1);cp=d['logits'].argmax(-1);f,cv=curve([dict(slice=name,y=y,bp=bp,cp=cp,scores=dict(margin_gain=d['margin_gain']))],'margin_gain');ff.append(f);front.append(dict(seed=seed,slice=name,condition=condition,**f));curves[(name,condition,seed)]=cv
            summ.append(dict(slice=name,condition=condition,**aggregate(rr,RAWKEYS),**aggregate(ff,['rec_at_nar1','rec_at_nar2'])))
    save('summary.json',summ);save('frontier_cells.json',front);writecsv('summary.csv',summ)
    pr=json.loads((RUN/'artifacts/policy_scores.json').read_text());assert len(pr)==120;ps=[]
    for name in C['slices']:
        for condition in methods:
            for policy in C['v8']['score_policies']:
                rr=[r for r in pr if r['slice']==name and r['condition']==condition and r['policy']==policy];ps.append(dict(slice=name,condition=condition,policy=policy,**aggregate(rr,['macro_f1','accuracy','delta_f1_pp','nar','recovery','source_valid_nar','source_valid_recovery']),query_accepted_mean=float(np.mean([v['query_accepted'] for v in rr])),reference_accepted_mean=float(np.mean([v['reference_accepted'] for v in rr]))))
    save('policy_summary.json',ps);writecsv('policy_summary.csv',ps)
    comparisons=[]
    for name in C['slices']:
        b=next(r for r in summ if r['slice']==name and r['condition']=='v6_lora')
        for condition in C['conditions']:
            a=next(r for r in summ if r['slice']==name and r['condition']==condition);r=dict(slice=name,condition=condition,f1_gain_over_v6_pp=a['macro_f1_mean']-b['macro_f1_mean'],nar_change_pp=a['nar_mean']-b['nar_mean'],recovery_change_pp=a['recovery_mean']-b['recovery_mean'],rec_at_nar1_gain_pp=a['rec_at_nar1_mean']-b['rec_at_nar1_mean'],cluster_purity_gain_pp=a['cluster_purity_mean']-b['cluster_purity_mean'],common_correction_gain_pp=a['common_corrected_mean']-b['common_corrected_mean'],common_other_wrong_change_pp=a['common_other_wrong_mean']-b['common_other_wrong_mean'])
            r['passed']=bool(r['f1_gain_over_v6_pp']>=0 and r['nar_change_pp']<=.2 and r['rec_at_nar1_gain_pp']>=.5);comparisons.append(r)
    candidates={arm:all(v['passed'] for v in comparisons if v['condition']==arm) for arm in C['conditions']};scoretests=[]
    for name in C['slices']:
        a=next(r for r in ps if r['slice']==name and r['condition']=='v6_lora' and r['policy']=='agreement_q95');b=next(r for r in ps if r['slice']==name and r['condition']=='v6_lora' and r['policy']=='margin_ref_rank_source1');ns=[r['nar']*100 for r in pr if r['slice']==name and r['condition']=='v6_lora' and r['policy']=='agreement_q95'];delta=a['recovery_mean']-b['recovery_mean'];scoretests.append(dict(slice=name,nar_each_source_seed=ns,recovery_change_pp=delta,passed=bool(max(ns)<=1 and delta>=-.5)))
    decision=dict(primary='no_teacher_kl',primary_passed=candidates['no_teacher_kl'],secondary_candidates=candidates,comparisons=comparisons,score_primary='fixed_v6_agreement_q95',scoring_passed=all(v['passed'] for v in scoretests),scoretests=scoretests,full_matrix_launched=False);save('registered_decisions.json',decision)
    logs=[]
    for seed in C['seeds']:
        for name in C['slices']:
            for arm in C['conditions']:
                hh=json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{arm}.json').read_text());logs.append(dict(seed=seed,slice=name,condition=arm,clip_steps=sum(v['gradient_norm']>1 for v in hh),geometry_weight_mean=float(np.mean([v['geometry_weight'] for v in hh])),geometry_weight_max=max(v['geometry_weight'] for v in hh),active_kl_mean=float(np.mean([v['active_kl'] for v in hh])),final_identity=hh[-1]['identity'],gate_fraction_mean=float(np.mean([v['kl_gate_fraction'] for v in hh]))))
    save('training_summary.json',logs)
    lines=['# Teacher解耦完整主表','', '均值±样本SD，3个源训练seed。ΔF1是相对frozen；Rec@NAR1为同一raw margin信号的query标签诊断，非部署阈值。所有新arm固定final400。','', '| Slice | Condition | ΔF1 pp | NAR % | Recovery % | Rec@NAR1 % | Cluster purity % | Common-wrong corrected % |','|---|---|---:|---:|---:|---:|---:|---:|']
    for a in summ:lines.append('| '+a['slice']+' | '+a['condition']+' | '+' | '.join(f"{a[k+'_mean']:.3f}±{a[k+'_sd']:.3f}" for k in ['delta_f1_pp','nar','recovery','rec_at_nar1','cluster_purity','common_corrected'])+' |')
    (RUN/'table_main.md').write_text('\n'.join(lines)+'\n');lines=['# 部署评分及风险转移','', '阈值只由reference和获准sourcevalid生成，无query拟合。NAR不能因采用极端分位而得到保证。','', '| Slice | Candidate | Policy | ΔF1 pp | NAR % | Recovery % | Source NAR % | Ref accepted | Query accepted |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for a in ps:lines.append('| '+a['slice']+' | '+a['condition']+' | '+a['policy']+' | '+' | '.join(f"{a[k+'_mean']:.3f}±{a[k+'_sd']:.3f}" for k in ['delta_f1_pp','nar','recovery','source_valid_nar'])+f" | {a['reference_accepted_mean']:.1f} | {a['query_accepted_mean']:.1f} |")
    (RUN/'table_scoring.md').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));grid=np.linspace(0,.06,241)
    for ax,name in zip(axs,C['slices']):
        for condition in methods[1:]:
            arr=[]
            for seed in C['seeds']:
                n,r,_=curves[(name,condition,seed)];ix=np.maximum(np.searchsorted(n,grid,side='right')-1,0);arr.append(r[ix])
            ax.plot(grid*100,np.mean(arr,0)*100,label=condition)
        ax.set(title=name,xlabel='Diagnostic NAR (%)',ylabel='Recovery (%)');ax.grid(alpha=.2);ax.axvline(1,color='gray',ls=':')
    axs[1].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_frontier.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,name in zip(axs,C['slices']):
        hs=[json.loads((RUN/'artifacts'/f'updates_{s}_{name}_geometry_grad_balance.json').read_text()) for s in C['seeds']];a=np.array([[v['geometry_weight'] for v in h] for h in hs]);ax.plot(np.arange(1,401),a.mean(0));ax.fill_between(np.arange(1,401),a.min(0),a.max(0),alpha=.2);ax.set(title=name,xlabel='Reference update',ylabel='Geometry multiplier');ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_gradient_weights.{ext}',dpi=170)
    plt.close(fig)
    text=['# Teacher解耦与reference风险路由 v8','',f'18/18新适应完成。预注册去KL主判据={decision["primary_passed"]}；固定v6一致性包络评分判据={decision["scoring_passed"]}。', '', '[完整主表](table_main.md)；[全部120部署评分状态](table_scoring.md)；[同风险诊断曲线](figure_frontier.png)；[几何梯度倍率轨迹](figure_gradient_weights.png)。']
    for r in comparisons:text.append(f"\n{r['slice']} / {r['condition']}：相对v6 F1 {r['f1_gain_over_v6_pp']:+.3f}pp，NAR {r['nar_change_pp']:+.3f}pp，Rec@NAR1 {r['rec_at_nar1_gain_pp']:+.3f}pp；聚类纯度 {r['cluster_purity_gain_pp']:+.3f}pp，共错集合纠错 {r['common_correction_gain_pp']:+.3f}pp。判据={r['passed']}。")
    for s in scoretests:text.append(f"\n{s['slice']} / 固定v6 agreement_q95：三个source-seed NAR={s['nar_each_source_seed']}%，纠错较rank-source1变化{s['recovery_change_pp']:+.3f}pp。判据={s['passed']}。")
    text+=['', '解释边界：KL去除只移除了直接输出分布约束，源邻居混合q、pull/relation几何锚点仍继承原编码器证据，不能称完全teacher-free。几何倍增不是创造独立证据；梯度范数不包含方向信息，需同时查看cosine与触发裁剪情况。', '', '所有BN affine/运行统计和纯source底座保持原值，未运行TENT拼接、BN协同更新或梯度正交投影。固定v6控制历史适应不算本轮新重复；所有源/query/reference/validation控制预测已重新核对。', '', '聚类只拟合reference标准化embedding102类KMeans；query majority purity/ARI/NMI只作终端诊断，不能用query类标签给聚类命名或部署。共错集合固定为原head/source-prototype/双source-bank同错，报告still-same-wrong/other-wrong/corrected，变化不等于全部可救错误。', '', 'ECDF与分位拟合只用独立reference；sourcevalid阈值规则不从query返调。无GMM及sigmoid(gain)。q99/一致性包络不是Damage Probability，NAR须看逐seed实测与纠错代价，不把拒绝所有改写包装为成功。', '', '本轮两片已参与既往开发，仅诊断开发证据；三个seed为纯source训练重复，适应随机流和reference固定。使用既有Version类别内容对应假设，USA仅地理节点不解释为定量丢包。未开放WTT/AWF，无新源模型训练，无全矩阵扩展。主次候选及所有阴性保留。']
    (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n');print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
