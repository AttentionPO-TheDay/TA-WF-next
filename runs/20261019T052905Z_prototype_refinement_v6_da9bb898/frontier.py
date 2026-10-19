"""Registered terminal evaluation; query-derived thresholds remain diagnostic only."""
import json
import numpy as np
from runtime import RUN, ROOT, C, save, metrics
from frontier_base import curve, writecsv


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rs=[]
    for seed in C['seeds']:rs+=json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text())
    assert len(rs)==36;save('scores.json',rs)
    prior=ROOT/C['v5_run'];v3=ROOT/C['baseline_run'];v4=ROOT/C['v4_run']
    oldrows=json.loads((prior/'artifacts/scores.json').read_text());old3=json.loads((v3/'artifacts/scores.json').read_text())
    methods=['frozen','v5_full_static','v5_batch_momentum','v5_momentum_proto','v3_tent_reference']+C['conditions']
    cells=[];fr=[];summ=[];curves={}
    for name in C['slices']:
        for method in methods:
            mr=[];ff=[]
            for seed in C['seeds']:
                with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as a:base=a['predictions'];y=a['labels'];ids=a['rows']
                if method=='frozen':
                    pred=base;score=np.zeros(len(pred));row={'seed':seed,'slice':name,'condition':method,'policy':'raw',**metrics(y,pred,base),'delta_f1_pp':0.}
                else:
                    if method.startswith('v5_'):
                        cond=method.removeprefix('v5_');location=prior
                        row=next(r.copy() for r in oldrows if r['seed']==seed and r['slice']==name and r['condition']==cond)
                        sig=prior/'artifacts'/f'signal_{seed}_{name}_{cond}.npz'
                    elif method=='v3_tent_reference':
                        cond='tent_bn_reference';location=v3
                        row=next(r.copy() for r in old3 if r['seed']==seed and r['slice']==name and r['condition']==cond and r['policy']=='raw')
                        sig=v4/'artifacts'/f'signal_{seed}_{name}_tent_reference.npz'
                    else:
                        cond=method;location=RUN
                        row=next(r.copy() for r in rs if r['seed']==seed and r['slice']==name and r['condition']==cond)
                        sig=RUN/'artifacts'/f'signal_{seed}_{name}_{cond}.npz'
                    with np.load(location/'artifacts'/f'raw_{seed}_{name}_{cond}.npz') as a:
                        pred=a['predictions'];assert np.array_equal(ids,a['rows']) and np.array_equal(y,a['labels'])
                    with np.load(sig) as a:score=a['combined']
                    row['condition']=method;row['reused_control']=location!=RUN
                met=metrics(y,pred,base)
                for k,v in met.items():assert abs(row[k]-v)<1e-12
                assert abs(row['delta_f1_pp']-100*(met['macro_f1']-metrics(y,base,base)['macro_f1']))<1e-10
                f,cv=curve([{'slice':name,'y':y,'bp':base,'cp':pred,'scores':{'combined':score}}],'combined')
                mr.append(row);ff.append(f);cells.append(row);fr.append({'seed':seed,'slice':name,'condition':method,**f});curves[(name,method,seed)]=cv
                np.savez_compressed(RUN/'artifacts'/f'curve_{seed}_{name}_{method}.npz',nar=cv[0],recovery=cv[1],threshold=cv[2])
            result={'slice':name,'condition':method}
            for k in ['macro_f1','accuracy','delta_f1_pp','nar','recovery']:
                a=np.array([r[k] for r in mr]);scale=1 if k=='delta_f1_pp' else 100
                result[k+'_mean']=float(a.mean())*scale;result[k+'_sd']=float(a.std(ddof=1))*scale
            for k in ['rec_at_nar1','rec_at_nar2','nar_at_rec2','nar_at_rec4']:
                a=[r[k] for r in ff];result[k+'_mean']=float(np.mean(a))*100 if all(v is not None for v in a) else None
                result[k+'_sd']=float(np.std(a,ddof=1))*100 if all(v is not None for v in a) else None
            summ.append(result)
    save('all_cells.json',cells);save('summary.json',summ);save('frontier_cells.json',fr)
    writecsv('summary.csv',summ);writecsv('cells.csv',cells);writecsv('frontier_cells.csv',fr)
    contrasts=[]
    for name in C['slices']:
        base=next(r for r in summ if r['slice']==name and r['condition']=='v5_momentum_proto')
        for arm in C['conditions']:
            x=next(r for r in summ if r['slice']==name and r['condition']==arm)
            vals=[r['delta_f1_pp'] for r in cells if r['slice']==name and r['condition']==arm]
            c={'slice':name,'condition':arm,'f1_gain_over_v5_pp':x['macro_f1_mean']-base['macro_f1_mean'],
               'nar_change_over_v5_pp':x['nar_mean']-base['nar_mean'],
               'recovery_change_over_v5_pp':x['recovery_mean']-base['recovery_mean'],
               'rec_at_nar1_change_pp':x['rec_at_nar1_mean']-base['rec_at_nar1_mean'],
               'rec_at_nar2_change_pp':x['rec_at_nar2_mean']-base['rec_at_nar2_mean'],
               'delta_f1_pp':x['delta_f1_pp_mean'],'each_source_seed_positive_vs_frozen':min(vals)>0}
            c['registered_slice_gate']=bool(c['f1_gain_over_v5_pp']>=.2 and c['nar_change_over_v5_pp']<=.2 and c['rec_at_nar1_change_pp']>=-1e-12 and min(vals)>0)
            contrasts.append(c)
    save('contrasts.json',contrasts);writecsv('contrasts.csv',contrasts)
    decision={'primary_candidate':C['refinement_v6']['primary_candidate'],'candidates':{},'diagnostic_frontiers_not_deployment':True,'full_matrix_launched':False}
    for arm in [C['refinement_v6']['primary_candidate']]+C['refinement_v6']['secondary_candidates']:
        cc=[c for c in contrasts if c['condition']==arm]
        decision['candidates'][arm]={'passed':all(c['registered_slice_gate'] for c in cc),'slices':cc}
    decision['primary_passed']=decision['candidates'][decision['primary_candidate']]['passed']
    decision['expansion_eligible']=decision['primary_passed'] and all(c['delta_f1_pp']>=3 for c in decision['candidates'][decision['primary_candidate']]['slices'])
    save('registered_decisions.json',decision)
    lines=['# 原型细化 v6 完整主表','','单位为百分比或pp，均值±样本SD；三源模型训练seed。全部新组固定最终400步、原始分类头直接输出。v5/v3为历史复用。','',
           '| Slice | Method | ΔF1 pp | NAR % | Recovery % | Rec@NAR≤1% % | Rec@NAR≤2% % |',
           '|---|---|---:|---:|---:|---:|---:|']
    for x in summ:
        lines.append('| '+x['slice']+' | '+x['condition']+' | '+' | '.join(f"{x[k+'_mean']:.3f}±{x[k+'_sd']:.3f}" for k in ['delta_f1_pp','nar','recovery','rec_at_nar1','rec_at_nar2'])+' |')
    (RUN/'table_main.md').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));grid=np.linspace(0,.06,301)
    plotted=['v5_momentum_proto','v3_tent_reference']+C['conditions']
    for ax,name in zip(axs,C['slices']):
        for method in plotted:
            vals=[]
            for seed in C['seeds']:
                n,r,_=curves[(name,method,seed)];ends=np.r_[np.flatnonzero(n[:-1]!=n[1:]),len(n)-1];u=n[ends];rr=r[ends]
                ix=np.searchsorted(u,grid,side='right')-1;vals.append(rr[np.maximum(ix,0)])
            vals=np.array(vals)*100;ax.plot(grid*100,vals.mean(0),label=method);ax.fill_between(grid*100,vals.min(0),vals.max(0),alpha=.06)
        ax.set(title=name,xlabel='NAR (%)',ylabel='Recovery (%)',xlim=(0,6));ax.grid(alpha=.2);ax.axvline(1,color='gray',ls=':');ax.axvline(2,color='gray',ls=':')
    axs[1].legend(fontsize=6);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_frontier.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,name in zip(axs,C['slices']):
        for arm in C['conditions']:
            hs=[json.loads((RUN/'artifacts'/f'updates_{s}_{name}_{arm}.json').read_text()) for s in C['seeds']]
            ar=np.array([[v['prototype_drift_mean'] for v in h] for h in hs]);ax.plot(np.arange(1,401),ar.mean(0),label=arm)
        ax.set(title=name,xlabel='Reference step',ylabel='Mean prototype displacement');ax.grid(alpha=.2)
    axs[1].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_prototype.{ext}',dpi=180)
    plt.close(fig)
    text=['# 原型细化 v6 结果','',f'36/36新适应完成；预定主候选sharp_target_only配对改进门槛：{decision["primary_passed"]}；扩展资格：{decision["expansion_eligible"]}。',
          '', '[全部方法与逐切片主表](table_main.md)；[风险—纠错前沿](figure_frontier.png)；[原型移动轨迹](figure_prototype.png)。','',
          '对照为已审计v5动量原型的历史复用。所有新组采用相同纯净底座、分区、400步及源identity预算。没有新增源模型训练。']
    for name in C['slices']:
        text+=['',f'{name}：']
        for c in [a for a in contrasts if a['slice']==name]:
            text.append(f"- {c['condition']}：ΔF1 {c['delta_f1_pp']:+.3f}pp，较v5 {c['f1_gain_over_v5_pp']:+.3f}pp；NAR变化{c['nar_change_over_v5_pp']:+.3f}pp；同NAR1%纠错变化{c['rec_at_nar1_change_pp']:+.3f}pp。")
    text+=['','所有臂、三个源训练seed和两切片完整保留。适应初始化/批流均固定20261019，源训练seed不等于独立适应随机流；均值SD仅描述稳定性。',
           '', '曲线用query标签事后定位阈值，只是诊断前沿，不能把对应点当作可部署策略或风险保证。实际报告raw预测无冻结回退，source-valid仅终端评分；query不参与原型/记忆拟合。',
           '', '本轮受v5公开切片结果反馈影响，仅为两切片诊断开发证据；Version身份与跨域类别采用已有内容支持假设，Network仅地理切片，不归因丢包或定量时延。未进行WTT/AWF新评分，未启动全矩阵。',
           '', '预算36新任务固定结束；任何最佳探索性组不能替换预定主候选。若未改善或未达+3pp目标，保留阴性，不继续追加搜索。']
    (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print((RUN/'table_main.md').read_text(),flush=True)

if __name__=='__main__':main()
