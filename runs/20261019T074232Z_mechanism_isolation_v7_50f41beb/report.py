"""All-arm terminal report. Predeclared comparisons; no feedback-driven additions."""
import json
import numpy as np
from runtime import RUN,ROOT,C,save,metrics
from frontier_base import curve,writecsv
from scoring import read,policies,KEYS
from audit import mm


def aggregate(rows,keys):
    o={}
    for k in keys:
        a=np.array([r[k] for r in rows]);scale=1 if k=='delta_f1_pp' else 100
        o[k+'_mean']=float(a.mean())*scale;o[k+'_sd']=float(a.std(ddof=1))*scale
    return o


def audit_policies():
    import torch
    from routing import scores
    records=[];rows=json.loads((RUN/'artifacts/policy_scores.json').read_text())
    for seed in C['seeds']:
        src=torch.load(ROOT/C['v4_run']/'artifacts'/f'source_{seed}.pt',map_location='cpu',weights_only=True)
        for name in C['slices']:
            old=ROOT/C['v6_run']/'artifacts';b=read(RUN/'artifacts'/f'base_{seed}_{name}.npz');raw=read(old/f'raw_{seed}_{name}_sharp_target_only.npz');vv=read(old/f'valid_{seed}_{name}_sharp_target_only.npz');r=read(RUN/'artifacts'/f'reference_signal_{seed}_{name}.npz')
            q=dict(base_logits=b['logits'],candidate_logits=raw['logits'],labels=b['labels']);v=dict(base_logits=src['valid_logits'].numpy(),candidate_logits=vv['logits'],labels=vv['labels'])
            vs={k:t.numpy() for k,t in scores(src['valid_logits'],torch.from_numpy(vv['logits']),src['valid_geometry'],src['calibration']).items()};qs=read(old/f'signal_{seed}_{name}_sharp_target_only.npz');rs={k:r[k] for k in KEYS}
            pp,notes=policies(v,q,r,vs,qs,rs);saved=json.loads((RUN/'artifacts'/f'policy_fit_{seed}_{name}.json').read_text());assert notes==saved
            vb=v['base_logits'].argmax(-1);vc=v['candidate_logits'].argmax(-1)
            # Independent source-only count checks for every registered threshold trial.
            from scoring import scaled
            vl=scaled(vs,vb!=vc,vs);mapping={'combined_source1':vs['combined'],'margin_source1':vs['normalized_margin_gain'],'margin_ref_z_source1':vl['z'],'margin_ref_rank_source1':vl['rank'],'rank_fusion_source1':vl['fusion']}
            for k,s in mapping.items():
                for trial in notes[k]['calibration']['trials']:
                    out=np.where((vc!=vb)&(s>=trial['threshold']),vc,vb);met=mm(v['labels'],out,vb);correct=vb==v['labels'];damage=int(((out!=v['labels'])&correct).sum());recover=int(((out==v['labels'])&~correct).sum())
                    assert damage==trial['damaged'] and recover==trial['recovered'] and abs(met['nar']-trial['nar'])<1e-12
            bp=q['base_logits'].argmax(-1)
            for policy,pred in pp.items():
                file=read(RUN/'artifacts'/f'policy_{seed}_{name}_{policy}.npz');assert np.array_equal(file['predictions'],pred) and np.array_equal(file['rows'],b['rows'])
                row=next(z for z in rows if z['seed']==seed and z['slice']==name and z['policy']==policy);met=mm(q['labels'],pred,bp)
                for k in met:assert abs(met[k]-row[k])<1e-12
                records.append(dict(seed=seed,slice=name,policy=policy,independent_metrics=True,query_free_fit_reproduced=True))
    save('policy_audit.json',dict(passed=True,records=records))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit_policies();new=sum([json.loads((RUN/'artifacts'/f'scores_{s}.json').read_text()) for s in C['seeds']],[]);assert len(new)==54;save('scores.json',new)
    methods=['frozen','v6_lora']+C['v7']['bn_states']+C['conditions'];cells=[];front=[];summ=[];curves={}
    for name in C['slices']:
        for method in methods:
            rr=[];ff=[]
            for seed in C['seeds']:
                b=read(RUN/'artifacts'/f'base_{seed}_{name}.npz');bp=b['predictions'];y=b['labels']
                if method=='frozen':cp=bp;score=np.zeros(len(y));row=dict(seed=seed,slice=name,condition=method,**metrics(y,cp,bp),delta_f1_pp=0.)
                else:
                    old=method=='v6_lora';location=ROOT/C['v6_run'] if old else RUN;cond='sharp_target_only' if old else method;raw=read(location/'artifacts'/f'raw_{seed}_{name}_{cond}.npz');cp=raw['predictions'];assert np.array_equal(raw['rows'],b['rows']);sig=read(location/'artifacts'/f'signal_{seed}_{name}_{cond}.npz');score=sig['combined']
                    met=mm(y,cp,bp);row=dict(seed=seed,slice=name,condition=method,**met,delta_f1_pp=100*(met['macro_f1']-mm(y,bp,bp)['macro_f1']),reused_control=old)
                    if not old:
                        original=next(r for r in new if r['seed']==seed and r['slice']==name and r['condition']==method)
                        for k in met:assert abs(row[k]-original[k])<1e-12
                f,cv=curve([dict(slice=name,y=y,bp=bp,cp=cp,scores=dict(combined=score))],'combined');rr.append(row);ff.append(f);cells.append(row);front.append(dict(seed=seed,slice=name,condition=method,**f));curves[(name,method,seed)]=cv
            out=dict(slice=name,condition=method,**aggregate(rr,['macro_f1','accuracy','delta_f1_pp','nar','recovery']),**aggregate(ff,['rec_at_nar1','rec_at_nar2']));summ.append(out)
    save('all_cells.json',cells);save('summary.json',summ);save('frontier_cells.json',front);writecsv('summary.csv',summ);writecsv('cells.csv',cells)
    policyrows=json.loads((RUN/'artifacts/policy_scores.json').read_text());ps=[]
    for name in C['slices']:
        for k in C['v7']['score_policies']:ps.append(dict(slice=name,policy=k,**aggregate([r for r in policyrows if r['slice']==name and r['policy']==k],['macro_f1','accuracy','delta_f1_pp','nar','recovery'])))
    save('policy_summary.json',ps);writecsv('policy_summary.csv',ps)
    def get(n,m):return next(r for r in summ if r['slice']==n and r['condition']==m)
    comparisons=[];scoretests=[]
    for name in C['slices']:
        route=get(name,'memory_route');mass=get(name,'memory_mass_control');stats=get(name,'stats_lora');lora=get(name,'v6_lora');joint=get(name,'stats_memory_route')
        c=dict(slice=name,memory_f1_gain_pp=route['macro_f1_mean']-mass['macro_f1_mean'],memory_rec_at_nar1_change_pp=route['rec_at_nar1_mean']-mass['rec_at_nar1_mean'],stats_f1_gain_pp=stats['macro_f1_mean']-lora['macro_f1_mean'],stats_nar_change_pp=stats['nar_mean']-lora['nar_mean'],joint_minus_stats_f1_pp=joint['macro_f1_mean']-stats['macro_f1_mean'],joint_minus_route_f1_pp=joint['macro_f1_mean']-route['macro_f1_mean'],interaction_f1_pp=joint['macro_f1_mean']-stats['macro_f1_mean']-route['macro_f1_mean']+lora['macro_f1_mean'])
        c['memory_passed']=bool(c['memory_f1_gain_pp']>=.2 and c['memory_rec_at_nar1_change_pp']>=-1e-12);c['stats_passed']=bool(c['stats_f1_gain_pp']>=.2 and c['stats_nar_change_pp']<=.2);comparisons.append(c)
        rank=next(r for r in ps if r['slice']==name and r['policy']=='margin_ref_rank_source1');combined=next(r for r in ps if r['slice']==name and r['policy']=='combined_source1');sc=dict(slice=name,nar_mean=rank['nar_mean'],recovery_gain_pp=rank['recovery_mean']-combined['recovery_mean']);sc['passed']=bool(sc['nar_mean']<=1 and sc['recovery_gain_pp']>=.5);scoretests.append(sc)
    decision=dict(memory_passed=all(r['memory_passed'] for r in comparisons),stats_passed=all(r['stats_passed'] for r in comparisons),scoring_passed=all(r['passed'] for r in scoretests),comparisons=comparisons,scoretests=scoretests,full_matrix_launched=False);save('registered_decisions.json',decision)
    lines=['# v7 全部状态','', '单位%或pp，三源训练seed均值±样本SD。Rec@NAR是使用query标签定位的诊断上限，不是部署阈值。','', '| Slice | Method | ΔF1 pp | NAR % | Recovery % | Rec@NAR1 % |','|---|---|---:|---:|---:|---:|']
    for r in summ:lines.append('| '+r['slice']+' | '+r['condition']+' | '+' | '.join(f"{r[k+'_mean']:.3f}±{r[k+'_sd']:.3f}" for k in ['delta_f1_pp','nar','recovery','rec_at_nar1'])+' |')
    (RUN/'table_main.md').write_text('\n'.join(lines)+'\n');lines=['# 源域校准与reference策略','', '| Slice | Policy | ΔF1 pp | NAR % | Recovery % |','|---|---|---:|---:|---:|']
    for r in ps:lines.append('| '+r['slice']+' | '+r['policy']+' | '+' | '.join(f"{r[k+'_mean']:.3f}±{r[k+'_sd']:.3f}" for k in ['delta_f1_pp','nar','recovery'])+' |')
    (RUN/'table_scoring.md').write_text('\n'.join(lines)+'\n')
    # Frozen/candidate paired error sets: oracle only for localization, never an output policy.
    pair=[]
    for seed in C['seeds']:
        for name in C['slices']:
            b=read(RUN/'artifacts'/f'base_{seed}_{name}.npz');y=b['labels'];wrong=b['predictions']!=y
            names=['affine_only','stats_only','tent_both','lora_stats_composed','lora_tent_composed','stats_lora','stats_memory_route']
            a=read(ROOT/C['v6_run']/'artifacts'/f'raw_{seed}_{name}_sharp_target_only.npz')['predictions']==y
            for k in names:
                z=read(RUN/'artifacts'/f'raw_{seed}_{name}_{k}.npz')['predictions']==y
                pair.append(dict(seed=seed,slice=name,condition=k,oracle_union_recovery=float(((a|z)&wrong).sum()/wrong.sum()),intersection_recovery=float(((a&z)&wrong).sum()/wrong.sum()),only_intervention_recovery=float(((~a&z)&wrong).sum()/wrong.sum()),only_v6_recovery=float(((a&~z)&wrong).sum()/wrong.sum())))
    save('error_pairs.json',pair)
    # Writer/gradient diagnostics are collected on reference only.
    diagnostics=[]
    for seed in C['seeds']:
        for name in C['slices']:
            for arm in C['conditions']:
                h=json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{arm}.json').read_text());gp=json.loads((RUN/'artifacts'/f'gradients_{seed}_{name}_{arm}.json').read_text())
                diagnostics.append(dict(seed=seed,slice=name,condition=arm,clip_steps=sum(r['gradient_norm']>1 for r in h),mean_writer_mass=float(np.mean([r['writer_weighted_mass'] for r in h])),mean_effective_samples=float(np.mean([r['effective_samples'] for r in h])),final_prototype_drift=h[-1]['prototype_drift_mean'],gradients=gp))
    save('training_diagnostics.json',diagnostics)
    # Unpaired channel moments and same-row paired CKA, all predeclared layers.
    layers=[]
    for seed in C['seeds']:
        for name in C['slices']:
            d=json.loads((RUN/'artifacts'/f'layers_{seed}_{name}.json').read_text())
            for arm,ls in d.items():
                for layer,values in ls.items():layers.append(dict(seed=seed,slice=name,condition=arm,layer=layer,**values))
    save('layer_cells.json',layers);writecsv('layer_cells.csv',layers)
    fig,axs=plt.subplots(1,2,figsize=(11,4));grid=np.linspace(0,.06,241)
    for ax,name in zip(axs,C['slices']):
        for method in ['v6_lora','tent_both']+C['conditions']:
            vals=[]
            for seed in C['seeds']:
                n,r,_=curves[(name,method,seed)];idx=np.maximum(np.searchsorted(n,grid,side='right')-1,0);vals.append(r[idx])
            ax.plot(grid*100,np.mean(vals,0)*100,label=method)
        ax.set(title=name,xlabel='Diagnostic NAR (%)',ylabel='Recovery (%)');ax.grid(alpha=.2)
    axs[1].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_frontier.{ext}',dpi=160)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7));ln=C['v7']['probe_layers']
    for j,name in enumerate(C['slices']):
        for arm in ['frozen','stats_only','affine_only','tent_both','v6_lora','stats_lora']:
            for i,(key,ylabel) in enumerate([('channel_gaussian_w2','Unpaired channel moment W2'),('paired_reference_cka','Paired reference CKA')]):
                val=[np.mean([r[key] for r in layers if r['slice']==name and r['condition']==arm and r['layer']==l]) for l in ln];axs[i,j].plot(np.arange(len(ln)),val,marker='o',label=arm);axs[i,j].set(title=name,ylabel=ylabel);axs[i,j].set_xticks(np.arange(len(ln)),['Gen1','Gen2','Gen3','TimeProj','Block1','Block2']);axs[i,j].grid(alpha=.2)
    axs[0,1].legend(fontsize=6);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_layers.{ext}',dpi=160)
    plt.close(fig)
    text=['# 三痛点机制定位 v7','',f'24新适应+30推理状态完成。预注册：可靠写入={decision["memory_passed"]}；早期统计={decision["stats_passed"]}；reference-rank源阈值={decision["scoring_passed"]}。','', '[全部状态](table_main.md)；[评分策略](table_scoring.md)；[风险纠错曲线](figure_frontier.png)；[层级探针](figure_layers.png)。','']
    for c in comparisons:text.append(f"{c['slice']}：可靠写入对写入总量匹配对照F1 {c['memory_f1_gain_pp']:+.3f}pp、Rec@NAR1 {c['memory_rec_at_nar1_change_pp']:+.3f}pp；stats_lora对v6 F1 {c['stats_f1_gain_pp']:+.3f}pp、NAR {c['stats_nar_change_pp']:+.3f}pp；联合交互F1 {c['interaction_f1_pp']:+.3f}pp。")
    text+=['', '解释边界：写入可靠性仍来自同一编码器，成功也只证明该固定筛选有用，不能证明独立语义证据；所有梯度保留正权重。mass-control匹配自身批次的总加权写入量，不能保证跨臂状态演化后逐步总量完全一致。', '', 'BN六层仅位于generator；stats_only保留纯source affine；affine_only保留纯source统计；tent_both精确复现既有strict-reference TENT。无新优化组合lora_tent_composed带历史400+400步成本，不能当同预算联合学习。stats_lora与v6 LoRA新优化预算相同但新增reference BN校准前向。', '', '分布距离不证明不可逆信息损坏；CKA只对同一reference样本前后配对。源/目标均只256行层级探针，无目标标签。各干预W2按对应source通道SD标准化，需结合绝对范围与预测指标判断，不能直接解释为因果修复幅度。', '', '部署评分阈值只由源valid标签选择；目标局部统计/GMM只拟合独立1024reference变化池。source规范化和reference分布可能仍不匹配；NAR约束不会自动跨域成立。GMM posterior并非安全概率，rank95只控制候选比例。', '', '所有arm/seed/负结果保留。三seed为源训练重复，适应rng和reference固定；SD不是独立切片抽样置信区间。Version/USA已观察诊断开发，不构成独立确认；Network仅地理节点，未测量丢包/时延。没有外部新评分、没有全矩阵扩展。', '', '数值、correlation、梯度及每行层探针见artifacts。独立query输入未用于BN、memory、prototype、CDF或GMM拟合。']
    (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n');print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
