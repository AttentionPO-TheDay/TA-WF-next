"""Frozen v9 gates: stability, performance and matched auxiliary correspondence."""
import json
import numpy as np
from runtime import RUN,C,save
from frontier_base import curve,writecsv
from scoring import read


def agg(rs,keys):
    o={}
    for k in keys:
        v=np.array([r[k] for r in rs]);scale=1 if k=='delta_f1_pp' else 100;o[k+'_mean']=float(v.mean())*scale;o[k+'_sd']=float(v.std(ddof=1))*scale
    return o


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sum([json.loads((RUN/'artifacts'/f'scores_{s}.json').read_text()) for s in C['seeds']],[]);assert len(rows)==42;save('scores.json',rows);writecsv('cells.csv',rows);methods=['frozen','v6_lora','v8_balance']+C['conditions'];summ=[];front=[];curves={};logs=[]
    for name in C['slices']:
        for cond in methods:
            rs=[v for v in rows if v['slice']==name and v['condition']==cond];ff=[]
            for seed in C['seeds']:
                d=read(RUN/'artifacts'/f'query_{seed}_{name}_{cond}.npz');f,cv=curve([dict(slice=name,y=d['labels'],bp=d['base_logits'].argmax(-1),cp=d['logits'].argmax(-1),scores=dict(margin_gain=d['margin_gain']))],'margin_gain');ff.append(f);front.append(dict(seed=seed,slice=name,condition=cond,**f));curves[(name,cond,seed)]=cv
            summ.append(dict(slice=name,condition=cond,**agg(rs,['macro_f1','accuracy','delta_f1_pp','nar','recovery','source_valid_macro_f1','cluster_purity','cluster_ari','cluster_nmi','common_same_wrong','common_corrected','common_other_wrong']),**agg(ff,['rec_at_nar1','rec_at_nar2'])))
        for cond in C['conditions']:
            for seed in C['seeds']:
                h=json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{cond}.json').read_text());late=h[200:];weight=np.array([v['geometry_weight'] for v in h]);gc=[v['gradient_previous_cosine'] for v in late if v['gradient_previous_cosine'] is not None];uc=[v['update_previous_cosine'] for v in late if v['update_previous_cosine'] is not None]
                logs.append(dict(seed=seed,slice=name,condition=cond,clip_steps=sum(v['gradient_norm']>1 for v in h),late_delta_log_weight=float(np.abs(np.diff(np.log(weight[199:]))).mean()),mean_weight=float(weight.mean()),late_gradient_cosine=float(np.mean(gc)),late_negative_gradient_fraction=float(np.mean(np.array(gc)<0)),late_update_cosine=float(np.mean(uc)),late_negative_update_fraction=float(np.mean(np.array(uc)<0)),aux_gradient_norm_mean=float(np.mean([v['aux_gradient_norm'] for v in h])),aux_loss_mean=float(np.mean([v['aux'] for v in h]))))
    save('summary.json',summ);save('frontier_cells.json',front);save('stability_cells.json',logs);writecsv('summary.csv',summ);writecsv('stability_cells.csv',logs)
    def get(name,cond):return next(v for v in summ if v['slice']==name and v['condition']==cond)
    tests=[]
    for name in C['slices']:
        smooth=get(name,'smooth_balance');ramp=get(name,'ramp_balance');base=get(name,'v8_balance');view=get(name,'smooth_view');shuffle=get(name,'smooth_shuffled_view');sl=[v for v in logs if v['slice']==name and v['condition']=='smooth_balance'];rl=[v for v in logs if v['slice']==name and v['condition']=='ramp_balance'];reduction=1-np.mean([v['late_delta_log_weight'] for v in sl])/max(np.mean([v['late_delta_log_weight'] for v in rl]),1e-15)
        t=dict(slice=name,smoothing_reduction=float(reduction),smooth_minus_ramp_f1_pp=smooth['macro_f1_mean']-ramp['macro_f1_mean'],smooth_minus_ramp_rec1_pp=smooth['rec_at_nar1_mean']-ramp['rec_at_nar1_mean'],smooth_clip_steps=sum(v['clip_steps'] for v in sl),ramp_clip_steps=sum(v['clip_steps'] for v in rl),smooth_minus_v8_f1_pp=smooth['macro_f1_mean']-base['macro_f1_mean'],smooth_minus_v8_nar_pp=smooth['nar_mean']-base['nar_mean'],smooth_minus_v8_rec1_pp=smooth['rec_at_nar1_mean']-base['rec_at_nar1_mean'],view_minus_shuffle_common_corrected_pp=view['common_corrected_mean']-shuffle['common_corrected_mean'],view_minus_shuffle_f1_pp=view['macro_f1_mean']-shuffle['macro_f1_mean'],view_minus_shuffle_nar_pp=view['nar_mean']-shuffle['nar_mean'],view_minus_smooth_f1_pp=view['macro_f1_mean']-smooth['macro_f1_mean'],view_minus_smooth_rec1_pp=view['rec_at_nar1_mean']-smooth['rec_at_nar1_mean'])
        t['stability_passed']=bool(t['smoothing_reduction']>=.3 and t['smooth_clip_steps']<=t['ramp_clip_steps'] and t['smooth_minus_ramp_f1_pp']>=-.1 and t['smooth_minus_ramp_rec1_pp']>=-.1)
        t['efficacy_passed']=bool(t['smooth_minus_v8_f1_pp']>=0 and t['smooth_minus_v8_nar_pp']<=.2 and t['smooth_minus_v8_rec1_pp']>=.2)
        t['auxiliary_passed']=bool(t['view_minus_shuffle_common_corrected_pp']>=.5 and t['view_minus_shuffle_f1_pp']>=0 and t['view_minus_shuffle_nar_pp']<=.2 and t['view_minus_smooth_f1_pp']>=0 and t['view_minus_smooth_rec1_pp']>=0);tests.append(t)
    decision=dict(stability_passed=all(v['stability_passed'] for v in tests),efficacy_passed=all(v['efficacy_passed'] for v in tests),auxiliary_passed=all(v['auxiliary_passed'] for v in tests),slices=tests,full_matrix_launched=False);save('registered_decisions.json',decision)
    lines=['# v9 全部状态','', '三源训练seed均值±样本SD；ΔF1相对frozen，Rec@NAR1为同一未饱和margin的标签诊断，不是部署操作点。','', '| Slice | Condition | ΔF1 pp | NAR % | Recovery % | Rec@NAR1 % | Common-wrong corrected % |','|---|---|---:|---:|---:|---:|---:|']
    for a in summ:lines.append('| '+a['slice']+' | '+a['condition']+' | '+' | '.join(f"{a[k+'_mean']:.3f}±{a[k+'_sd']:.3f}" for k in ['delta_f1_pp','nar','recovery','rec_at_nar1','common_corrected'])+' |')
    (RUN/'table_main.md').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));grid=np.linspace(0,.06,241)
    for ax,name in zip(axs,C['slices']):
        for cond in methods[2:]:
            vv=[]
            for seed in C['seeds']:
                n,r,_=curves[(name,cond,seed)];ix=np.maximum(np.searchsorted(n,grid,side='right')-1,0);vv.append(r[ix])
            ax.plot(grid*100,np.mean(vv,0)*100,label=cond)
        ax.set(title=name,xlabel='Diagnostic NAR (%)',ylabel='Recovery (%)');ax.grid(alpha=.2);ax.axvline(1,color='gray',ls=':')
    axs[1].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_frontier.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for j,name in enumerate(C['slices']):
        for cond in C['conditions']:
            hs=[json.loads((RUN/'artifacts'/f'updates_{s}_{name}_{cond}.json').read_text()) for s in C['seeds']];ws=np.array([[v['geometry_weight'] for v in h] for h in hs]);cs=np.array([[v['gradient_previous_cosine'] or 0 for v in h] for h in hs]);axs[0,j].plot(np.arange(1,401),ws.mean(0),label=cond);axs[1,j].plot(np.arange(1,401),np.convolve(cs.mean(0),np.ones(20)/20,mode='same'),label=cond)
        axs[0,j].set(title=name,ylabel='Geometry multiplier');axs[1,j].set(xlabel='Reference update',ylabel='Consecutive gradient cosine (20-step mean)');axs[0,j].grid(alpha=.2);axs[1,j].grid(alpha=.2)
    axs[0,1].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_stability.{ext}',dpi=170)
    plt.close(fig)
    va=json.loads((RUN/'artifacts/source_view_audit.json').read_text());text=['# 渐进平滑与辅助视图 v9','',f'24/24新适应完成。预注册稳定性={decision["stability_passed"]}、相对v8效果={decision["efficacy_passed"]}、辅助视图={decision["auxiliary_passed"]}。', '', '[完整主表](table_main.md)；[同风险诊断](figure_frontier.png)；[倍率/梯度稳定性](figure_stability.png)。', '', f'辅助raw-TAM PCA视图仅source拟合：sourcevalid MacroF1={100*va["valid_macro_f1"]:.3f}%，Accuracy={100*va["valid_accuracy"]:.3f}%，NLL={va["nll"]:.4f}，温度={va["temperature"]}，enabled={va["enabled"]}。']
    for t in tests:text.append(f"\n{t['slice']}：倍率后期变化减少{100*t['smoothing_reduction']:.1f}%；smooth/ramp F1 {t['smooth_minus_ramp_f1_pp']:+.3f}pp、Rec@NAR1 {t['smooth_minus_ramp_rec1_pp']:+.3f}pp；smooth/v8 F1 {t['smooth_minus_v8_f1_pp']:+.3f}pp、NAR {t['smooth_minus_v8_nar_pp']:+.3f}pp、Rec@NAR1 {t['smooth_minus_v8_rec1_pp']:+.3f}pp。真实视图/打乱共错纠错{t['view_minus_shuffle_common_corrected_pp']:+.3f}pp；真实视图/smooth F1 {t['view_minus_smooth_f1_pp']:+.3f}pp、Rec@NAR1 {t['view_minus_smooth_rec1_pp']:+.3f}pp。")
    text+=['', '独立输入视图终端证据（不能将单独改对能力直接当作LoRA可达上限）：']
    for name in C['slices']:
        vv=[json.loads((RUN/'artifacts'/f'view_metrics_{s}_{name}.json').read_text()) for s in C['seeds']];text.append(f"\n{name}：辅助视图query accuracy={100*np.mean([v['query_accuracy'] for v in vv]):.3f}%；原head错误集合中view正确={100*np.mean([v['frozen_wrong_corrected'] for v in vv]):.3f}%；固定共错集合中view正确={100*np.mean([v['common_corrected'] for v in vv]):.3f}%、仍同错={100*np.mean([v['common_same_wrong'] for v in vv]):.3f}%。")
    text+=['', '识别边界：v8本来已有动态EMA梯度比，本轮增加渐进和log倍率平滑，不能把35×均值误写成固定倍率。较平滑的倍率并不证明梯度方向稳定；梯度前后cosine、负cosine比例、clip次数及累计倍率分开报告。渐进改变了早期总几何梯度预算，所以ramp控制不可缺，不能只与v8比较就把收益全部归因平滑。', '', 'PCA行正交不自动产生独立语义；辅助视图来自与原模型共享的raw TAM，通过不同尺度/压缩形成证据。它不是原128D表征的等距旋转，也没有新增网站信息。温度与启用资格只由sourcevalid决定；sourcevalid通过不证明目标可靠性。真实/打乱对应关系对照检验该具体辅助证据，而非泛指所有多视角架构。', '', '打乱对照在自身每batch匹配总加权辅助质量；跨臂状态演化仍不同。source所有18553标签只用于本轮声明的原型中心，未训练新backbone/神经分类器；三源模型共享同一辅助PCA/prototype，不是三次独立辅助训练。', '', '只报告raw主输出与同一margin诊断曲线，无新拒绝阈值/极端分位/GMM，无已停分支重跑。全部原始BN/source状态冻结。query不参与任何投影、原型、记忆、归一化、温度或聚类拟合，query标签仅终端评价。1024reference无标签，KMeans拟合仅reference，query纯度不能用来改预测。', '', '两片为已观察诊断开发；三个seed为source训练重复，reference和适应流固定。Version采用既有内容支持的身份假设，USA仅地域节点。所有预定臂和阴性保留；未新增外部评分，未扩展全矩阵。']
    (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n');print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
