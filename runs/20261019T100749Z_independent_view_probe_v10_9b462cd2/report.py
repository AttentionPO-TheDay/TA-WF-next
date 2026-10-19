import json,csv,math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from runtime import RUN,ROOT,C,save,metrics

def read(p):
    with np.load(p) as a:return {k:a[k] for k in a.files}
def probability(z,temp):return torch.from_numpy(z/temp).softmax(-1).numpy()
def independent_metric(y,p):
    a=np.zeros((102,102),dtype=np.int64);np.add.at(a,(y,p),1);tp=a.diagonal();den=a.sum(0)+a.sum(1);f=np.divide(2*tp,den,out=np.zeros(102,dtype=float),where=den>0)
    return float(f.mean()),float(tp.sum()/len(y))
def frontier(y,b,p,score):
    ix=np.flatnonzero(p!=b);ix=ix[np.argsort(-score[ix],kind='stable')];correct=b==y;bad=~correct
    # Only implementable score-threshold prefixes; never split ties using labels.
    ends=np.flatnonzero(np.r_[score[ix][1:]!=score[ix][:-1],True]) if len(ix) else np.array([],dtype=int)
    d=np.cumsum(correct[ix])/max(correct.sum(),1);r=np.cumsum((p[ix]==y[ix])&bad[ix])/max(bad.sum(),1)
    ns=np.r_[0.,d[ends]];rs=np.r_[0.,r[ends]];allowed=np.flatnonzero(ns<=.01);j=int(allowed[np.argmax(rs[allowed])]);points=np.unique(np.r_[0,np.linspace(0,len(ns)-1,min(200,len(ns)),dtype=int),j])
    return dict(rec_at_nar1=float(rs[j]),attained_nar1=float(ns[j]),candidate_oracle_recovery=float(((p==y)&bad).sum()/max(bad.sum(),1)),changes=len(ix),diagnostic_threshold=float(score[ix][ends[j-1]]) if j else None),np.stack((ns[points],rs[points]),1)
def conditional(y,b,p,common):
    bc=b==y;pc=p==y;change=p!=b;corr=(~bc)&pc;damage=bc&(~pc);bw=(~bc)&(~pc);same=bw&(p==b)
    a=int((bc&pc).sum());bb=int(damage.sum());cc=int(corr.sum());dd=int(bw.sum());den=math.sqrt(max((a+bb)*(cc+dd)*(a+cc)*(bb+dd),1));qden=a*dd+bb*cc
    return dict(common_wrong_count=int(common.sum()),common_wrong_corrected=float(pc[common].mean()),common_wrong_same=float((p[common]==b[common]).mean()),common_wrong_other=float(((~pc)&(p!=b))[common].mean()),frozen_error_count=int((~bc).sum()),frozen_error_corrected=float(pc[~bc].mean()),common_same_wrong_count=int((common&(p==b)).sum()),corrections=int(corr.sum()),damages=int(damage.sum()),other_wrong_changes=int((change&bw).sum()),disagreement_count=int(change.sum()),disagreement_precision=float(corr.sum()/max(change.sum(),1)),correctness_phi=(a*dd-bb*cc)/den,Q_statistic=float((a*dd-bb*cc)/qden) if qden else 0.,sourceview_error_overlap=float(bw.sum()/max((~bc).sum(),1)),same_wrong_on_frozen_errors=float(same.sum()/max((~bc).sum(),1)))
def auc(y,b,p,score):
    corr=(b!=y)&(p==y);damage=(b==y)&(p!=y);ix=corr|damage
    return float(roc_auc_score(corr[ix],score[ix])) if corr.any() and damage.any() else None

def main():
    source_rows=[];cells=[];fronts=[];curves={};v9=ROOT/C['v9_run'];historical=json.loads((v9/'artifacts/frontier_cells.json').read_text())
    for seed in C['seeds']:
        sr=json.loads((RUN/'artifacts'/f'train_summary_{seed}.json').read_text());source_rows+=sr
        # Verify matched initialization, source index streams and fixed checkpoint selection.
        assert len({r['initial_sha256'] for r in sr})==1
        histories=[json.loads((RUN/'artifacts'/f'history_{seed}_{a}.json').read_text()) for a in C['arms']]
        assert all(len(h)==30 for h in histories)
        for e in range(30):assert len({h[e]['permutation_sha256'] for h in histories})==1
        for row,h in zip(sr,histories):
            e=max([x for x in h if 'valid_macro_f1' in x],key=lambda x:x['valid_macro_f1']);assert e['epoch']==row['best_epoch']
        for name in C['slices']:
            base=read(v9/'artifacts'/f'base_{seed}_{name}.npz');y=base['labels'];b=base['predictions'];common=base['common_wrong'];v=read(v9/'artifacts'/f'valid_{seed}_{name}_frozen.npz');vy=v['labels'];vz=v['logits']
            temps=C['training']['temperature_grid'];t0=min(temps,key=lambda t:float(F.cross_entropy(torch.from_numpy(vz/t),torch.from_numpy(vy))));bp=probability(base['logits'],t0)
            for arm in C['arms']+['v9_pca']:
                if arm=='v9_pca':
                    a=read(v9/'artifacts'/f'view_{seed}_{name}.npz');p=a['query_predictions'];assert np.array_equal(a['query_rows'],base['rows']);q=None
                else:
                    a=read(RUN/'artifacts'/f'query_{seed}_{name}_{arm}.npz');assert np.array_equal(a['rows'],base['rows']) and np.array_equal(a['labels'],y);row=next(r for r in sr if r['arm']==arm);q=probability(a['logits'],row['temperature']);p=q.argmax(-1)
                met=metrics(y,p,b);mf,acc=independent_metric(y,p);assert abs(met['macro_f1']-mf)<1e-12 and abs(met['accuracy']-acc)<1e-12
                cell=dict(seed=seed,slice=name,arm=arm,output='standalone',**met,**conditional(y,b,p,common));cell['delta_f1_pp']=100*(met['macro_f1']-metrics(y,b,b)['macro_f1']);cells.append(cell)
                if q is None:continue
                sq=np.sort(q,axis=1);scores=dict(view_probability_margin=sq[:,-1]-sq[:,-2],class_log_odds_gain=np.log(np.maximum(q[np.arange(len(y)),p],1e-30))-np.log(np.maximum(q[np.arange(len(y)),b],1e-30)))
                for score_name,score in scores.items():
                    f,curve=frontier(y,b,p,score);old=next(r for r in historical if r['seed']==seed and r['slice']==name and r['condition']=='smooth_balance');f.update(seed=seed,slice=name,arm=arm,score=score_name,correction_damage_auc=auc(y,b,p,score),rec1_gain_vs_v9_pp=100*(f['rec_at_nar1']-old['rec_at_nar1']));fronts.append(f);curves[f'{seed}_{name}_{arm}_{score_name}']=curve
                fp=(.5*q+.5*bp).argmax(-1);met=metrics(y,fp,b);mf,acc=independent_metric(y,fp);assert abs(met['macro_f1']-mf)<1e-12
                cells.append(dict(seed=seed,slice=name,arm=arm,output='fixed_equal_fusion',**met,**conditional(y,b,fp,common),delta_f1_pp=100*(met['macro_f1']-metrics(y,b,b)['macro_f1'])))
    summaries=[]
    for name in C['slices']:
        for arm in C['arms']+['v9_pca']:
            for output in ['standalone','fixed_equal_fusion']:
                r=[x for x in cells if x['slice']==name and x['arm']==arm and x['output']==output]
                if not r:continue
                out=dict(slice=name,arm=arm,output=output)
                for k in ['accuracy','macro_f1','nar','recovery','common_wrong_corrected','common_wrong_same','common_wrong_other','disagreement_precision','correctness_phi','Q_statistic','sourceview_error_overlap','delta_f1_pp']:
                    scale=1 if k in ['correctness_phi','Q_statistic','delta_f1_pp'] else 100;vals=np.array([x[k] for x in r])*scale;out[k+'_mean']=float(vals.mean());out[k+'_sd']=float(vals.std(ddof=1))
                if output=='standalone' and arm!='v9_pca':
                    for k in ['rec_at_nar1','rec1_gain_vs_v9_pp','correction_damage_auc']:
                        vals=[x[k] for x in fronts if x['slice']==name and x['arm']==arm and x['score']=='view_probability_margin'];scale=100 if k=='rec_at_nar1' else 1;out[k+'_mean']=float(np.mean(vals)*scale) if all(v is not None for v in vals) else None
                summaries.append(out)
    decisions=[]
    for arm in C['arms']:
        ss=[r for r in source_rows if r['arm']==arm];valid=all(r['valid_macro_f1']>=.60 for r in ss);sg=[]
        for name in C['slices']:
            r=next(r for r in summaries if r['slice']==name and r['arm']==arm and r['output']=='standalone');co=r['common_wrong_corrected_mean']>=10;au=r['correction_damage_auc_mean'] is not None and r['correction_damage_auc_mean']>=.65;fr=r['rec1_gain_vs_v9_pp_mean']>=.5;sg.append(dict(slice=name,common_wrong_pass=co,ranking_auc_pass=au,frontier_pass=fr,passed=co and au and fr))
        decisions.append(dict(arm=arm,source_semantic_pass=valid,slices=sg,passed=valid and all(s['passed'] for s in sg)))
    save('source_summary.json',source_rows);save('cells.json',cells);save('frontier_cells.json',fronts);save('summary.json',summaries);save('registered_decisions.json',dict(primary='direction_order',primary_passed=decisions[0]['passed'],arms=decisions,adaptation_started=False,external_scored=False,extra_source_supervision=True));np.savez_compressed(RUN/'artifacts/frontier_curves.npz',**curves)
    for file,data in [('cells.csv',cells),('summary.csv',summaries),('frontier_cells.csv',fronts),('source_summary.csv',source_rows)]:
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (RUN/'artifacts'/file).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    lines=['# 独立视图能力与共错互补性 v10','',f'完成9个独立纯源模型；主候选direction_order门槛：{decisions[0]["passed"]}。本轮只做能力探针，没有接入TTA。','', '所有结果为两个已观察目标切片的诊断开发；增加了源监督模型预算，不能据此宣称同预算优于原TTA。','', '|视图|源valid F1均值±SD|源train accuracy均值|','|---|---:|---:|']
    for arm in C['arms']:
        rr=[r for r in source_rows if r['arm']==arm];vv=np.array([r['valid_macro_f1'] for r in rr])*100;lines.append(f'|{arm}|{vv.mean():.3f} ± {vv.std(ddof=1):.3f}|{100*np.mean([r["source_train_accuracy"] for r in rr]):.3f}|')
    lines+=['','下表均为三训练种子均值，单位%或pp；同NAR列来自query标签诊断阈值，不可部署。共错指冻结head、源原型及双bank投票指向同一错误类别的固定子集。','', '|切片|视图|输出|F1|ΔF1 pp|NAR|Recovery|共错纠正|共错仍同错|Rec@NAR1|较v9 pp|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('|'+ '|'.join([r['slice'],r['arm'],r['output']]+[f'{r[k]:.3f}' for k in ['macro_f1_mean','delta_f1_pp_mean','nar_mean','recovery_mean','common_wrong_corrected_mean','common_wrong_same_mean']]+[f'{r[k]:.3f}' if k in r and r[k] is not None else '—' for k in ['rec_at_nar1_mean','rec1_gain_vs_v9_pp_mean']])+'|')
    lines+=['','预注册判决（主、次、对照分别报告，不互换）：','', '```json',json.dumps(decisions,ensure_ascii=False,indent=2),'```','', '条件AUROC只比较改对与改坏两组，排除旧错→其他错；完整分歧精度包含这类无效改写，详见cells.csv。源valid仅选checkpoint与温度；所有9训练完成之后才读取query评分。GroupNorm与独立逐样本变换不拟合query统计。固定等权概率融合不是风险保证。额外源训练与原主模型的不同训练日程/结构，意味着这是能力与互补性对照，而不是严格的同算力模型排名。','', '输出方向不同不等于获得正确的新证据；只有共同错误中的真实纠正与受控损伤前沿提升才支持后续整合。主候选未过门槛则不扩展；次候选通过也必须单独立项才接入适应。zero timestamp沿用缓存padding，无法恢复真实t=0包。方向IAT负时间差截零，比例见input_diagnostics.json。没有新的WTT/AWF访问，未修改既有封存。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n');(RUN/'table_main.md').write_text('\n'.join(lines[lines.index('|切片|视图|输出|F1|ΔF1 pp|NAR|Recovery|共错纠正|共错仍同错|Rec@NAR1|较v9 pp|'):lines.index('预注册判决（主、次、对照分别报告，不互换）：')])+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4));arms=C['arms'];colors=['#1976d2','#d65b29','#31986a'];x=np.arange(3)
    axes[0].bar(x,[100*np.mean([r['valid_macro_f1'] for r in source_rows if r['arm']==a]) for a in arms],color=colors);axes[0].set_xticks(x,arms,rotation=20);axes[0].set_ylabel('Source-valid Macro-F1 (%)');axes[0].axhline(60,color='gray',linestyle='--')
    for j,name in enumerate(C['slices'],1):
        for a,color in zip(arms,colors):
            k=[r for r in summaries if r['slice']==name and r['arm']==a and r['output']=='standalone'][0];idx=arms.index(a);axes[j].bar(idx-.18,k['common_wrong_corrected_mean'],width=.35,color=color,label='Common-error corrected' if idx==0 else None);axes[j].bar(idx+.18,k['rec_at_nar1_mean'],width=.35,color=color,alpha=.4,label='Recovery at NAR <= 1% (diagnostic)' if idx==0 else None)
        axes[j].set_xticks(x,arms,rotation=20);axes[j].set_title(name);axes[j].set_ylabel('Rate (%)');axes[j].legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_complementarity.{ext}',dpi=170)
    plt.close(fig);save('metric_audit.json',dict(passed=True,independent_confusion_metrics=len(cells),source_selection_and_permutations_checked=True,states=18,frontiers=len(fronts)))

if __name__=='__main__':main()
