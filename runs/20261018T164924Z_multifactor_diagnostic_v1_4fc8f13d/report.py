"""Fixed complete tables and shared-coordinate topology visualizations; no favorable selection."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from runtime import RUN,ROOT,C,sha,save,reference

def csv_file(name,rows):
    p=RUN/'artifacts'/name
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def external():
    ext=ROOT/C['external_run'];audit=json.loads((ext/'artifacts/postrun_audit.json').read_text());assert audit['passed'] and audit['metrics_recomputed']==144
    summary=json.loads((ext/'artifacts/dataset_summary.json').read_text())
    assert summary['source_scores_sha256']==sha(ext/'artifacts/scores.json')
    rows=summary['summary'];csv_file('table3_external_v4.csv',rows)
    (RUN/'table3_external_v4_full.md').write_text((ext/'RESULTS.md').read_text())
    (RUN/'artifacts/external_v4_scores_reused.json').write_text((ext/'artifacts/scores.json').read_text())
    text=['# Table 3: External method transfer (reused v4)', '',
          '各外部历史数据独立训练方向模型，非Temporal闭集分类器跨类别零样本迁移；新prototype_anchor未在本表评价。切片×seed等权。', '',
          '| Dataset | Method | F1 (%) | Accuracy (%) | ΔF1 (pp) | NAR (%) | Positive / total |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['dataset']} | {r['condition']} | {r['macro_f1_pct']:.3f} | {r['accuracy_pct']:.3f} | {r['delta_f1_pp']:+.3f} | {r['nar_pct']:.3f} | {r['positive_cells']}/{r['cells']} |")
    text+=['','WTT 2022-02-16 frozen F1=18.545%，严重退化原因未证实，全部日期与负收益均保留。AWF按source定义200类评分，未知标签排除口径沿用v4。',
           '本轮选择原型方法受到上述外部反馈影响；登记为post-external-feedback版本，不再把该测试部分视作新方法未反馈确认集。',
           f'原始144单元与完整逐日期表：`{C["external_run"]}/RESULTS.md`。']
    (RUN/'table3_external_v4.md').write_text('\n'.join(text)+'\n')
    save('external_reuse_audit.json',{'passed':True,'source_run':C['external_run'],'metrics_reused':144,'new_external_scoring':False,'source_scores_sha256':sha(ext/'artifacts/scores.json'),'table_sha256':sha(RUN/'artifacts/table3_external_v4.csv')})
    return '\n'.join(text)

def grouped(rows):
    output=[]
    for name in C['slices']:
        for condition in C['conditions']:
            selected=[r for r in rows if r['slice']==name and r['condition']==condition];assert len(selected)==3
            r={'slice':name,'condition':condition,'query_rows':selected[0]['query_rows']}
            for key in ('macro_f1','accuracy','delta_f1_pp','nar','recovery'):
                v=np.array([x[key] for x in selected]);scale=1 if key=='delta_f1_pp' else 100
                r[key+'_mean']=float(v.mean()*scale);r[key+'_sd']=float(v.std(ddof=1)*scale)
            if name=='behavior_subpage':
                r['hostname_nmi']=float(np.mean([x['hostname_nmi'] for x in selected]));r['hostname_ari']=float(np.mean([x['hostname_ari'] for x in selected]))
            else:r['hostname_nmi']=r['hostname_ari']=''
            output.append(r)
    return output

def table(rows,keys):
    lines=['| Slice | Condition | '+' | '.join(keys)+' |','|---|---|'+ '|'.join(['---:']*len(keys))+'|']
    for r in rows:
        lines.append('| '+r['slice']+' | '+r['condition']+' | '+' | '.join(f'{r[k+"_mean"]:.3f}±{r[k+"_sd"]:.3f}' for k in keys)+' |')
    return '\n'.join(lines)

def exportfig(fig,name):
    for ext in ('png','svg','pdf'):fig.savefig(RUN/'artifacts'/f'{name}.{ext}',dpi=180,bbox_inches='tight')

def visuals():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    ref=reference(C['visualization']['seed']);mu=ref['moments']['mean'].numpy();sd=ref['moments']['variance'].clamp_min(1e-5).sqrt().numpy()
    prototypes=(ref['class_prototypes'].numpy()-mu)/sd;source_x=ref['source_embeddings'].numpy();source_y=ref['source_labels'].numpy()
    source_indices=[]
    for label in C['visualization']['classes']:
        ix=np.flatnonzero(source_y==label);source_indices.extend(ix[:C['visualization']['max_per_class']])
    sx=(source_x[source_indices]-mu)/sd;sy=source_y[source_indices];data={}
    for name in C['visualization']['slices']:
        before=torch.load(RUN/'artifacts'/f'visual_{name}_frozen.pt',weights_only=True)
        after=torch.load(RUN/'artifacts'/f'visual_{name}_prototype_anchor.pt',weights_only=True)
        assert torch.equal(before['rows'],after['rows']) and torch.equal(before['labels'],after['labels'])
        bx=(before['features'].numpy()-mu)/sd;ax=(after['features'].numpy()-mu)/sd;y=before['labels'].numpy()
        arrays=np.concatenate([sx,bx,ax,prototypes[C['visualization']['classes']]])
        z=TSNE(n_components=2,perplexity=C['visualization']['tsne_perplexity'],init='pca',learning_rate='auto',random_state=C['visualization']['tsne_seed'],n_jobs=2).fit_transform(arrays)
        a=len(sx);b=a+len(bx);c=b+len(ax);sz,bz,az,cz=z[:a],z[a:b],z[b:c],z[c:]
        fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True,sharey=True)
        for axis,tz,title in zip(axes,[bz,az],['Original tokens','Prototype-anchored tokens']):
            axis.scatter(sz[:,0],sz[:,1],c=sy,cmap='tab10',vmin=0,vmax=9,s=10,alpha=.2,marker='.')
            axis.scatter(tz[:,0],tz[:,1],c=y,cmap='tab10',vmin=0,vmax=9,s=15,alpha=.65)
            axis.scatter(cz[:,0],cz[:,1],c=C['visualization']['classes'],cmap='tab10',vmin=0,vmax=9,s=110,marker='X',edgecolors='black')
            axis.set_title(title);axis.set_xlabel('Joint t-SNE 1')
        axes[0].set_ylabel('Joint t-SNE 2');fig.suptitle(name+'; shared fit, fixed seed; source faint / prototypes X')
        exportfig(fig,'figure_manifold_'+name);plt.close(fig)
        details={}
        for label,x in [('before',bx),('after',ax)]:
            distance=((x[:,None]-prototypes[None])**2).mean(-1)
            n=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8);p=prototypes/(np.linalg.norm(prototypes,axis=1,keepdims=True)+1e-8)
            details[label]={'true_prototype_mse':float(distance[np.arange(len(y)),y].mean()),'nearest_prototype_accuracy':float(((n@p.T).argmax(1)==y).mean()),'mean_feature_variance':float(x.var(0).mean()),'centroid_pair_distance':float(np.linalg.norm(np.array([x[y==k].mean(0) for k in sorted(set(y))])[:,None]-np.array([x[y==k].mean(0) for k in sorted(set(y))])[None],axis=-1).mean())}
        data[name]={'metrics':details,'rows':before['rows'].tolist(),'labels':y.tolist(),'joint_coordinates':z.tolist(),'space':'source-standardized110D pooled time tokens','source_count':a,'target_count':len(y)}
    save('manifold_data.json',data)
    return data

def main():
    t3=external()
    if '--external-only' in sys.argv:return
    assert json.loads((RUN/'artifacts/postrun_audit.json').read_text())['passed']
    rows=json.loads((RUN/'artifacts/scores.json').read_text());summary=grouped(rows);csv_file('table2_alignment.csv',summary)
    base=[dict(r) for r in summary if r['condition']=='frozen']
    reports=json.loads((ROOT/C['source_run']/'artifacts/reports.json').read_text())
    for r in base:
        raw=[x for x in rows if x['slice']==r['slice'] and x['condition']=='frozen']
        for metric in ('macro_f1','accuracy'):
            ratios=[100*x[metric]/next(z for z in reports if z['seed']==x['seed'])['valid'][metric] for x in raw]
            r[metric+'_source_valid_retention_pct']=float(np.mean(ratios))
    csv_file('table1_impact.csv',base)
    t1=table(base,['macro_f1','accuracy']);t2=table(summary,['macro_f1','accuracy','delta_f1_pp','nar','recovery'])
    (RUN/'table1_impact.md').write_text('# Table 1: Observed diagnostic impact\n\n'+t1+'\n')
    (RUN/'table2_alignment.md').write_text('# Table 2: Alignment ablation\n\n'+t2+'\n')
    aggregate=[]
    for family in ('version','network','behavior'):
        for condition in C['conditions']:
            ss=[r for r in summary if r['slice'].startswith(family) and r['condition']==condition]
            aggregate.append({'family':family,'condition':condition,**{k:float(np.mean([r[k+'_mean'] for r in ss])) for k in ('macro_f1','accuracy','delta_f1_pp','nar','recovery')}})
    csv_file('factor_equal_summary.csv',aggregate)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    theta=np.linspace(0,2*np.pi,len(base),endpoint=False);values=np.array([r['macro_f1_mean'] for r in base]);fig=plt.figure(figsize=(9,7));axis=fig.add_subplot(111,projection='polar')
    axis.plot(np.r_[theta,theta[0]],np.r_[values,values[0]],marker='o');axis.set_xticks(theta,[r['slice'] for r in base],fontsize=8);axis.set_ylim(0,100);axis.set_title('Frozen Macro-F1; observed diagnostic slices (%)');exportfig(fig,'figure_impact_radar');plt.close(fig)
    manifold=visuals();decisions={}
    avg=lambda method,key:np.mean([r[key] for r in aggregate if r['condition']==method])
    decisions['anchor_vs_frozen_delta_f1_pp']=float(avg('prototype_anchor','delta_f1_pp'))
    decisions['anchor_vs_gate_delta_f1_pp']=float(avg('prototype_anchor','macro_f1')-avg('tent_entropy_gate','macro_f1'))
    decisions['anchor_nar']=float(avg('prototype_anchor','nar'));decisions['gate_nar']=float(avg('tent_entropy_gate','nar'))
    decisions['candidate_pass']=bool(decisions['anchor_vs_frozen_delta_f1_pp']>0 and decisions['anchor_vs_gate_delta_f1_pp']>0 and decisions['anchor_nar']<=decisions['gate_nar'])
    decisions['topology_delta_f1_pp']=float(avg('prototype_anchor','macro_f1')-avg('prototype_no_topology','macro_f1'))
    decisions['topology_pass']=bool(decisions['topology_delta_f1_pp']>0 and avg('prototype_anchor','nar')<=avg('prototype_no_topology','nar'))
    decisions['version_048_minus_045_f1_pp']=float(next(r for r in base if r['slice']=='version_048')['macro_f1_mean']-next(r for r in base if r['slice']=='version_045')['macro_f1_mean'])
    net=[r['macro_f1_mean'] for r in base if r['slice'].startswith('network')];decisions['network_f1_range_pp']=float(np.ptp(net));decisions['network_f1_sd_pp']=float(np.std(net,ddof=1))
    save('decisions.json',decisions)
    lines=['# 多因素诊断 v1 结果','','210单元完成；所有checkpoint重载、预测及指标复算通过。三seed均值±样本SD，102类内容支持标签对应假设；不是官方身份映射或未见确认。','','## Table 1', '',t1,'','## Table 2','',t2,'',t3,'','## 固定判据','',json.dumps(decisions,ensure_ascii=False,indent=2),'','## 流形图','',json.dumps({k:v['metrics'] for k,v in manifold.items()},ensure_ascii=False,indent=2),'',
           '限制：Version为已知版本锚点重用的诊断子集；Network只有地区信息；Behavior缺官方homepage映射，另报hostname NMI/ARI。TENT BN query前向使用批统计量，其余仅reference适应。类间关系只是候选约束，t-SNE辅助图不证明因果或物理流形恢复。所有阴性结果保留；不按本轮结果继续扫描阈值/步数。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    files=list((RUN/'artifacts').glob('figure_*.*'))+[RUN/'artifacts/manifold_data.json',RUN/'artifacts/table1_impact.csv',RUN/'artifacts/table2_alignment.csv',RUN/'artifacts/table3_external_v4.csv']
    save('render_audit.json',{'passed':True,'files_sha256':{str(p.relative_to(RUN)):sha(p) for p in files},'joint_tsne':True,'fixed_selection':True})

if __name__=='__main__':main()
