"""Full factorial reporting. No adaptive selection, truthful high-dimensional visualization."""
import csv,itertools,json
from collections import defaultdict
import numpy as np
import torch
from runtime import RUN,ROOT,C,save

def factor(name):return 'Version' if name.startswith('version') else 'Network' if name.startswith('network') else 'Behavior'

def macro(rs,metric):
    f=defaultdict(list)
    for r in rs:f[factor(r['slice'])].append(r[metric])
    return float(np.mean([np.mean(v) for v in f.values()]))

def main():
    assert json.loads((RUN/'artifacts/postrun_audit.json').read_text())['passed']
    scores=json.loads((RUN/'artifacts/scores.json').read_text())
    fields=['seed','slice','condition','macro_f1','accuracy','delta_f1_pp','nar','recovery','query_gate_coverage','query_bypassed','adapter_parameters','steps','updates','seconds','peak_cuda_bytes']
    with (RUN/'artifacts/cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(scores)
    summary=[];lines=['# 完整2×2×2组合结果','', '3因素等权，再汇总3个seed；均值±样本标准差。F1/Acc/NAR/Recovery为%，ΔF1为pp。','', '| 方法 | F1 | Acc | ΔF1 | NAR | Recovery | 门控覆盖 |','|---|---:|---:|---:|---:|---:|---:|']
    for cond in C['conditions']:
        d={'condition':cond}
        for metric in ['macro_f1','accuracy','delta_f1_pp','nar','recovery','query_gate_coverage']:
            vs=np.array([macro([r for r in scores if r['seed']==seed and r['condition']==cond],metric) for seed in C['seeds']])
            scale=1 if metric=='delta_f1_pp' else 100
            d[metric+'_mean']=float(vs.mean()*scale);d[metric+'_sd']=float(vs.std(ddof=1)*scale);d[metric+'_seeds']=(vs*scale).tolist()
        summary.append(d)
        lines.append('| '+cond+' | '+' | '.join(f"{d[m+'_mean']:.3f}±{d[m+'_sd']:.3f}" for m in ['macro_f1','accuracy','delta_f1_pp','nar','recovery','query_gate_coverage'])+' |')
    lines+=['','门控覆盖对ungated行也报告同一源域门控的潜在覆盖；ungated实际不旁路。只显示gate覆盖不等于方法实际改变了该比例预测。','', '旧v1基线使用800步、本轮400步；历史结果不作为匹配消融。']
    (RUN/'table_combinations.md').write_text('\n'.join(lines)+'\n')
    with (RUN/'artifacts/summary.csv').open('w',newline='') as f:
        fields=[k for k in summary[0] if not k.endswith('_seeds')];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summary)
    # Explicit slice and factor tables, all methods and seeds.
    per=[]
    for name in C['slices']:
        for cond in C['conditions']:
            rs=[r for r in scores if r['slice']==name and r['condition']==cond]
            row={'slice':name,'factor':factor(name),'condition':cond}
            for metric in ['macro_f1','accuracy','delta_f1_pp','nar','recovery','query_gate_coverage']:
                scale=1 if metric=='delta_f1_pp' else 100
                v=np.array([r[metric] for r in rs])*scale
                row[metric+'_mean']=float(v.mean());row[metric+'_sd']=float(v.std(ddof=1))
            per.append(row)
    with (RUN/'artifacts/by_slice.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    fac=[]
    for family in ['Version','Network','Behavior']:
        for cond in C['conditions']:
            row={'factor':family,'condition':cond}
            for metric in ['macro_f1','accuracy','delta_f1_pp','nar','recovery']:
                v=[np.mean([r[metric] for r in scores if r['seed']==seed and r['condition']==cond and factor(r['slice'])==family]) for seed in C['seeds']]
                scale=1 if metric=='delta_f1_pp' else 100;row[metric+'_mean']=float(np.mean(v)*scale);row[metric+'_sd']=float(np.std(v,ddof=1)*scale)
            fac.append(row)
    with (RUN/'artifacts/by_factor.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fac[0]));w.writeheader();w.writerows(fac)
    # Orthogonal full-factorial effects: mean high-minus-low across all matched combinations.
    effects=[];axes=[['single','gmm4'],['ungated','routed'],['residual','lora']]
    for size in [1,2,3]:
        for subset in itertools.combinations(range(3),size):
            row={'effect':' x '.join(['GMM4-Single','Routed-Ungated','LoRA-Residual'][i] for i in subset)}
            for metric in ['macro_f1','nar','recovery']:
                vals=[]
                for seed in C['seeds']:
                    v=0.
                    for bits in itertools.product([0,1],repeat=3):
                        cond='_'.join(axes[i][bits[i]] for i in range(3));sign=np.prod([1 if bits[i] else -1 for i in subset])
                        v+=sign*macro([r for r in scores if r['seed']==seed and r['condition']==cond],metric)
                    # Interaction is difference of simple effects averaged over unused axes.
                    vals.append(v/(2**(3-size))*100)
                row[metric+'_effect_pp']=float(np.mean(vals));row[metric+'_sd_pp']=float(np.std(vals,ddof=1));row[metric+'_seed_values']=vals
            effects.append(row)
    save('factorial_effects.json',effects);save('summary.json',summary)
    el=['# 配对因子效应','', '主效应为其余因子水平平均后的高−低；二阶交互为差的差，三阶为差的差的差；3因素等权、3seed均值±样本标准差。仅描述性诊断，不作显著性宣称。','', '| 效应 | F1 pp | NAR pp | Recovery pp |','|---|---:|---:|---:|']
    for e in effects:el.append('| '+e['effect']+' | '+' | '.join(f"{e[m+'_effect_pp']:+.3f}±{e[m+'_sd_pp']:.3f}" for m in ['macro_f1','nar','recovery'])+' |')
    (RUN/'table_factorial_effects.md').write_text('\n'.join(el)+'\n')
    old=json.loads((ROOT/C['diagnostic_run']/'artifacts/scores.json').read_text())
    hl=['# 历史参考（未重跑）','', '旧v1同一纯源模型、同一query；适应800步，本轮400步，不是预算匹配比较。','', '| 方法 | F1 % | NAR % |','|---|---:|---:|']
    for cond in ['frozen','tent_bn','tent_entropy_gate','prototype_anchor']:
        hl.append(f"| {cond} | {100*macro([r for r in old if r['condition']==cond],'macro_f1'):.3f} | {100*macro([r for r in old if r['condition']==cond],'nar'):.3f} |")
    (RUN/'table_historical_controls.md').write_text('\n'.join(hl)+'\n')
    # Source/post-attention manifold diagnostics. Joint fit guarantees shared coordinates.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances
    figdir=RUN/'artifacts/figures';figdir.mkdir(exist_ok=True);visuals=[]
    art=torch.load(RUN/'artifacts/source_gmm_42.pt',map_location='cpu',weights_only=True);p=art['fits']['4'];mu=p['mean'].numpy();sd=p['sd'].numpy()
    centers=p['centers'].numpy().mean(1);source=art['source_embeddings'].numpy();sy=art['source_labels'].numpy()
    source_centers=np.stack([((source[sy==k]-mu)/sd).mean(0) for k in range(102)])
    for name in C['visualization']['slices']:
        for cond in C['visualization']['conditions']:
            v=torch.load(RUN/'artifacts'/f'visual_{name}_{cond}.pt',map_location='cpu',weights_only=True)
            a=(v['before'].numpy()-mu)/sd;b=(v['after'].numpy()-mu)/sd;y=v['labels'].numpy()
            record={'slice':name,'condition':cond,'rows':len(y),'true_prototype_mse_before':float(np.mean((a-source_centers[y])**2)),'true_prototype_mse_after':float(np.mean((b-source_centers[y])**2)),'nearest_prototype_accuracy_before':float((pairwise_distances(a,source_centers).argmin(1)==y).mean()),'nearest_prototype_accuracy_after':float((pairwise_distances(b,source_centers).argmin(1)==y).mean()),'variance_ratio':float(b.var(0).mean()/a.var(0).mean()),'routing_coverage':float(v['keep'].float().mean())}
            coords=TSNE(n_components=2,perplexity=30,random_state=20261019,init='pca',learning_rate='auto').fit_transform(np.concatenate([source_centers,a,b]))
            fig,axs=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
            for ax,values,title in zip(axs,[coords[102:102+len(a)],coords[102+len(a):]],['Frozen','Adapted + exact bypass']):
                ax.scatter(values[:,0],values[:,1],c=y,cmap='turbo',s=10,alpha=.65,vmin=0,vmax=101);ax.scatter(coords[:102,0],coords[:102,1],marker='x',c='black',s=18);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f'{name}: {cond}\nJoint t-SNE; post-attention features; black crosses: source centroids',fontsize=10);fig.tight_layout()
            for ext in ['png','pdf','svg']:fig.savefig(figdir/f'{name}_{cond}.{ext}',dpi=180)
            plt.close(fig);visuals.append(record)
    save('visual_diagnostics.json',visuals)
    # F1/NAR comparison with all eight methods.
    fig,ax=plt.subplots(figsize=(8,5))
    for i,r in enumerate(summary):
        ax.scatter(r['nar_mean'],r['delta_f1_pp_mean']);ax.annotate(str(i+1),(r['nar_mean'],r['delta_f1_pp_mean']),xytext=(4,4),textcoords='offset points')
    ax.axhline(0,c='grey',lw=.7);ax.set_xlabel('NAR (%)');ax.set_ylabel('Delta Macro-F1 (pp)');ax.set_title('Full factorial: recovery vs negative adaptation')
    fig.text(.99,.01,'; '.join(f'{i+1}: {r["condition"]}' for i,r in enumerate(summary)),ha='right',fontsize=5,wrap=True);fig.tight_layout(rect=[0,.06,1,1])
    for ext in ['png','pdf','svg']:fig.savefig(figdir/f'combination_f1_nar.{ext}',dpi=180)
    plt.close(fig)
    best=max(summary,key=lambda r:r['macro_f1_mean'])
    result=['# 组合实验结果','', '240/240 新单元完成；全部checkpoint全query重载、基座锁定、指标复算、拒绝行逐位旁路审计通过。','', '完整结果见 [组合总表](table_combinations.md)、[因子效应](table_factorial_effects.md)、[历史参考](table_historical_controls.md)；逐切片、逐因素与240单元CSV见artifacts。','', f'本批描述性最高F1：{best["condition"]}，F1 {best["macro_f1_mean"]:.3f}±{best["macro_f1_sd"]:.3f}%，ΔF1 {best["delta_f1_pp_mean"]:+.3f}pp，NAR {best["nar_mean"]:.3f}%。这是已观察诊断结果中的比较，未据此返调或新增外部评分。','', '主效应：']
    for e in effects[:3]:result.append(f'- {e["effect"]}: F1 {e["macro_f1_effect_pp"]:+.3f}pp，NAR {e["nar_effect_pp"]:+.3f}pp。')
    result+=['','图为实际提取的Transformer后128维特征；LoRA前Token不变。t-SNE是描述性图，必须结合visual_diagnostics.json的高维类心距离、邻近准确率和方差判断，不能据低维聚集宣称语义恢复。','', '样本高熵或距原型远并不证明物理上不可恢复。门控只保证被拒绝行原预测不变，接受行仍可能误杀；GMM源域分量不保证对应真实页面模式。W2采用点高斯到分量高斯，协方差影响软分配，不能宣称估计了测试样本协方差。','', '输入、代码、完整配置、源拟合产物的SHA256见artifacts/freeze.json；既往外部v4与诊断v1均已影响本轮方案，只作开发证据。']
    (RUN/'RESULTS.md').write_text('\n'.join(result)+'\n')
    print((RUN/'table_combinations.md').read_text(),flush=True)

if __name__=='__main__':main()
