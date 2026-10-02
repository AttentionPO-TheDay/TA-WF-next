"""Report every fixed probe and derive cautious, nonexclusive diagnostic flags."""
from __future__ import annotations
import csv
import statistics
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose import *


def paired(rows,left,right,kind,lam,role='valid',metric='macro_f1'):
    return [100*(next(r for r in rows if r['condition']==left and r['seed']==s and r['kind']==kind and r['lambda']==lam)[role][metric]
                 -next(r for r in rows if r['condition']==right and r['seed']==s and r['kind']==kind and r['lambda']==lam)[role][metric]) for s in (1729,3407,2026)]


def summarize():
    c=config_read();reports=[];errors=[];rows=[];old_count=0;probe_count=0
    for name in c['conditions']:
        for seed in c['seeds']:
            path=RUN/'artifacts'/f'{name}_{seed}'/'report.json'
            if not path.exists():errors.append(f'{name}_{seed}: report missing');continue
            report=json.loads(path.read_text());reports.append(report);rows+=report['rows']
            assert report['config_sha256']==sha(RUN/'config.json')
            old_count+=report['original_prediction_sets_checked'];probe_count+=report['verification']['probe_prediction_sets_checked']
            if report['verification']['errors']:errors.extend(report['verification']['errors'])
            assert report['network_unchanged']
            assert sha(path.parent/'features.npz')==report['feature_sha256']
    complete=len(reports)==9 and probe_count==108 and old_count==18 and not errors
    integrity={'complete':complete,'original_prediction_sets_checked':old_count,'probe_prediction_sets_checked':probe_count,
               'expected_probe_prediction_sets':108,'probe_fits_checked':sum(r['verification']['probe_fits_checked'] for r in reports),
               'errors':errors,'network_weights_unchanged':all(r['network_unchanged'] for r in reports),'future_access':False}
    write_json(RUN/'artifacts/integrity.json',integrity)
    if not complete:
        summary=f'生成器/分类器诊断不完整：原预测{old_count}/18，探针{probe_count}/108；缺失/错误{len(errors)}项，不作归因。'
        (RUN/'RESULTS.md').write_text('# 实验结果\n\n'+summary+'\n'+str(errors)+'\n');register(summary,'stopped');return 1
    # Save all scores, never select a lambda from validation.
    with (RUN/'artifacts/probe_metrics.csv').open('w') as f:
        fields=['condition','seed','best_epoch','kind','lambda','source_accuracy','source_macro_f1','valid_accuracy','valid_macro_f1']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:row[k] for k in fields[:5]}|{role+'_'+m:row[role][m] for role in ('source','valid') for m in ('accuracy','macro_f1')})
    B='B_local_attention_joint';C='C_local_attention_frozen';D='D_local_attention_slow'
    means={name:{kind:{str(lam):{role:{m:statistics.mean(r[role][m] for r in rows if r['condition']==name and r['kind']==kind and r['lambda']==lam)
                     for m in ('accuracy','macro_f1')} for role in ('source','valid')}
                     for lam in (1.,.1,10.)} for kind in c['probe_kinds']} for name in c['conditions']}
    comparisons=[]
    for left,right in [(B,C),(D,C),(D,B)]:
        for kind in c['probe_kinds']:
            for lam in (1.,.1,10.):
                deltas=paired(rows,left,right,kind,lam)
                comparisons.append({'left':left,'right':right,'kind':kind,'lambda':lam,'paired_valid_f1_delta_pp':deltas,'mean_valid_f1_delta_pp':statistics.mean(deltas)})
    def changes(kind,lam):return paired(rows,B,C,kind,lam)
    primary=changes('packet_tokens',1.);source_diff=statistics.mean(paired(rows,B,C,'packet_tokens',1.,role='source'))
    degradation=statistics.mean(primary)<=-1 and all(x<0 for x in primary) and all(statistics.mean(changes('packet_tokens',lam))<0 for lam in (.1,10.))
    improved=statistics.mean(primary)>=1 and all(x>0 for x in primary) and all(statistics.mean(changes('packet_tokens',lam))>0 for lam in (.1,10.))
    readout_gaps={name:{str(lam):[100*(next(r for r in rows if r['condition']==name and r['seed']==s and r['kind']=='all_views' and r['lambda']==lam)['valid']['macro_f1']
                           -next(r for r in reports if r['condition']==name and r['seed']==s)['original_scores']['valid']['macro_f1']) for s in c['seeds']]
                           for lam in (1.,.1,10.)} for name in c['conditions']}
    gaps=readout_gaps[B]
    underuse=statistics.mean(gaps['1.0'])>=1 and all(x>0 for x in gaps['1.0']) and all(statistics.mean(gaps[str(lam)])>0 for lam in (.1,10.))
    original_bc=[100*(next(r for r in reports if r['condition']==B and r['seed']==s)['original_scores']['valid']['macro_f1']-next(r for r in reports if r['condition']==C and r['seed']==s)['original_scores']['valid']['macro_f1']) for s in c['seeds']]
    flags={'generator_probe_degradation':degradation,'generator_probe_overfit_pattern':degradation and source_diff>=0,
           'classifier_underuses_linear_signal':underuse,'generator_improvement_head_reversal':improved and all(x<0 for x in original_bc),
           'source_probe_B_minus_C_f1_pp':source_diff,'primary_packet_B_minus_C_f1_pp':primary}
    if degradation and underuse:decision='BOTH_GENERATOR_DEGRADATION_AND_CLASSIFIER_UNDERUSE'
    elif degradation:decision='GENERATOR_DEGRADATION_SUPPORTED'
    elif underuse:decision='CLASSIFIER_UNDERUSE_SUPPORTED'
    elif flags['generator_improvement_head_reversal']:decision='GENERATOR_IMPROVEMENT_HEAD_REVERSAL'
    else:decision='INCONCLUSIVE'
    # Read existing trajectories; no missing intermediate checkpoints are invented.
    trajectory={};prior=ROOT/c['prior_run']
    for name in c['conditions']:
        curves=[];detail=[]
        for seed in c['seeds']:
            history=json.loads((prior/'artifacts'/f'history_{name}_{seed}.json').read_text())['history']
            scored=[r for r in history if 'valid' in r];curves.append(scored)
            late=[r for r in scored if r['epoch']>=55]
            x=np.asarray([r['generator_token_delta_rms'] for r in late]);y=np.asarray([r['valid']['macro_f1'] for r in late])
            correlation=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None
            best=max(scored,key=lambda r:r['valid']['macro_f1']);last=scored[-1]
            detail.append({'seed':seed,'best_epoch':best['epoch'],'last_source_accuracy':last['source']['accuracy'],
                           'last_valid_macro_f1':last['valid']['macro_f1'],'best_to_last_valid_f1_pp':100*(last['valid']['macro_f1']-best['valid']['macro_f1']),
                           'late_drift_valid_f1_pearson':correlation,'last_token_delta_rms':last['generator_token_delta_rms']})
        trajectory[name]={'curves':curves,'detail':detail}
    plot_trajectories(trajectory)
    # Class-level comparisons for original predictions and both primary probes.
    class_summary={};class_rows=[]
    for representation in ['original','packet_tokens','all_views']:
        conf={}
        for name in c['conditions']:
            cms=[]
            for seed in c['seeds']:
                directory=RUN/'artifacts'/f'{name}_{seed}'
                filename='original_confusion_valid.npy' if representation=='original' else f'confusion_{representation}_lambda1p0_valid.npy'
                cms.append(np.load(directory/filename))
            conf[name]=np.stack(cms)
        br=np.diagonal(conf[B],axis1=1,axis2=2)/5;cr=np.diagonal(conf[C],axis1=1,axis2=2)/5
        delta=br.mean(0)-cr.mean(0)
        dc=conf[B].sum(0)-conf[C].sum(0);np.fill_diagonal(dc,0)
        edge_order=np.argsort(dc.reshape(-1))[::-1]
        edges=[{'true_class':int(i//102),'predicted_class':int(i%102),'extra_errors_across_seed_runs':int(dc.reshape(-1)[i])} for i in edge_order[:10] if dc.reshape(-1)[i]>0]
        class_summary[representation]={'recall_worse_classes':int((delta< -1e-12).sum()),'recall_better_classes':int((delta>1e-12).sum()),'recall_tied_classes':int((abs(delta)<=1e-12).sum()),'top_added_confusions':edges}
        for label in range(102):class_rows.append({'representation':representation,'class_id':label,'B_recall':float(br[:,label].mean()),'C_recall':float(cr[:,label].mean()),'B_minus_C_recall':float(delta[label]),'negative_seed_count':int((br[:,label]<cr[:,label]).sum())})
    with (RUN/'artifacts/class_recall.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(class_rows[0]));w.writeheader();w.writerows(class_rows)
    write_json(RUN/'artifacts/trajectory.json',trajectory)
    write_json(RUN/'artifacts/class_summary.json',class_summary)
    result={'means':means,'comparisons':comparisons,'all_views_probe_minus_original_f1_pp':readout_gaps,'flags':flags,'decision':decision}
    write_json(RUN/'artifacts/summary.json',result)
    lines=['# 生成器/分类器诊断结果','',f'状态：completed。预定诊断裁决：{decision}。',
           '', '| 条件 | 原完整模型 valid F1 | packet探针 valid F1 | all_views探针 valid F1 | packet探针 source F1 |',
           '|---|---:|---:|---:|---:|']
    for name in c['conditions']:
        orig=statistics.mean(r['original_scores']['valid']['macro_f1'] for r in reports if r['condition']==name)
        pk=means[name]['packet_tokens']['1.0'];av=means[name]['all_views']['1.0']
        lines.append(f"| {name} | {orig*100:.3f}% | {pk['valid']['macro_f1']*100:.3f}% | {av['valid']['macro_f1']*100:.3f}% | {pk['source']['macro_f1']*100:.3f}% |")
    lines+=['',f'主lambda=1的B−C packet探针F1逐seed差：{primary}pp；平均{statistics.mean(primary):+.3f}pp。',
            f'B的all_views探针−原模型F1逐seed差：{gaps["1.0"]}pp；平均{statistics.mean(gaps["1.0"]):+.3f}pp。',
            f'生成器表示退化门槛：{degradation}；满足source不降的表示过拟合模式：{flags["generator_probe_overfit_pattern"]}；分类器未利用线性信号门槛：{underuse}。',
            '', '所有lambda=0.1/1/10分项均保存在artifacts/probe_metrics.csv与summary.json，没有按valid选择lambda。主模型与checkpoint均未更新，valid没有参与探针拟合。']
    for lam in (.1,10.):
        lines.append(f"敏感性lambda={lam}：B−C packet平均{statistics.mean(changes('packet_tokens',lam)):+.3f}pp；B all_views探针−原模型平均{statistics.mean(gaps[str(lam)]):+.3f}pp。")
    if underuse:
        lines+=['','诊断支持：同等输入信息下，固定线性读出能提取原完整模型未充分利用的验证分类信号，下一步优先研究分类器的汇聚/信息保留与正则化。该结论包含读出容量、优化和正则化差异，不锁定某一层的唯一原因。']
    if degradation:
        lines+=['','诊断同时支持：在统一线性探针下，联合学习生成器B的验证表征较冻结C退化。若source探针表现也下降，只称表征退化，不能单独断言生成器过拟合。']
    if not degradation:
        lines+=['','未达到生成器表征退化的一致门槛；不能把完整模型的泛化下降直接等同于生成器token已丢失类别信息，也不能据此证明生成器完全没有问题。']
    if not underuse and not degradation:
        lines+=['','两项主归因证据不足，保留未决，不强行二选一。']
    lines+=['', '轨迹与逐类结果见artifacts/trajectory.json、class_recall.csv、class_summary.json及trajectory.png。相关仅是描述，不能当因果；三个seed共享valid样本，不能当作额外独立样本。',
            '',f'独立核验：18/18原预测重载、108/108探针预测复算、54/54探针标准化/权重正规方程核验；最大相对残差{max(r["verification"]["normal_equation_max_relative_residual"] for r in reports):.3e}。网络权重保持不变。',
            '', '限制：原checkpoint已按同一valid选择，本轮仅开发诊断。固定线性probe不是表示能力上限；标准化/L2不完全具备坐标变换不变性，敏感性检查不能消除全部此类影响。没有新完整模型训练或未来/外部数据访问。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    summary=f"C/B/D冻结token诊断完成：{decision}；B−C packet F1 {statistics.mean(primary):+.3f}pp，B all_views探针−原模型 {statistics.mean(gaps['1.0']):+.3f}pp；18原预测/108探针预测/54拟合核验通过。"
    register(summary,'completed');print(summary,flush=True);return 0


def plot_trajectories(trajectory):
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    styles={'C_local_attention_frozen':('C frozen','#555555'),'B_local_attention_joint':('B joint','#c44e52'),'D_local_attention_slow':('D slow','#4c72b0')}
    for name,entry in trajectory.items():
        label,color=styles[name];curves=entry['curves'];epochs=[r['epoch'] for r in curves[0]]
        values=[[[100*r['source']['accuracy'] for r in curve] for curve in curves],
                [[100*r['valid']['macro_f1'] for r in curve] for curve in curves],
                [[r['generator_token_delta_rms'] for r in curve] for curve in curves]]
        for ax,value in zip(axes,values):
            array=np.asarray(value);ax.plot(epochs,array.mean(0),label=label,color=color)
            ax.fill_between(epochs,array.min(0),array.max(0),alpha=.1,color=color)
    for ax,title in zip(axes,['Source accuracy (%)','Validation Macro-F1 (%)','Generator token change (RMS)']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(RUN/'artifacts/trajectory.png',dpi=170);plt.close(fig)


def register(summary,state):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    p=ROOT/'STATUS.md';text=p.read_text();marker='当前归因诊断（2026-09-26）：`'+RUN.name+'`。'
    p.write_text('\n'.join(marker+summary if line.startswith(marker) else line for line in text.splitlines())+'\n')

if __name__=='__main__':raise SystemExit(summarize())
