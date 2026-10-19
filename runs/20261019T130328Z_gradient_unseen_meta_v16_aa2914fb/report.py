from runtime import *

def main():
 cells=json.loads((RUN/'artifacts/cells.json').read_text());matches=json.loads((RUN/'artifacts/matched_frontiers.json').read_text());policies=json.loads((RUN/'artifacts/source_policy_target.json').read_text());summary=[];feas=json.loads((RUN/'artifacts/feasibility.json').read_text())
 for part in ['diagnostic','query']:
  for sl in C['slices']:
   for score in C['scores']:
    rr=[r for r in cells if r['partition']==part and r['slice']==sl and r['score']==score];out=dict(partition=part,slice=sl,score=score,states=len(rr))
    for k in ['pauc_norm','rec_at_nar1','attained_nar1','corrected','damaged','other_wrong_accepted','other_per_correction','diag_f1','oracle_recovery']:
     vv=[r[k] for r in rr if r[k] is not None];out[k+'_mean']=float(np.mean(vv)) if len(vv)==3 else None;out[k+'_sd']=float(np.std(vv,ddof=1)) if len(vv)==3 else None
    summary.append(out)
 decisions=[]
 for sl in C['slices']:
  get=lambda s:next(r for r in summary if r['partition']=='query' and r['slice']==sl and r['score']==s)
  primary=get('GU_all');control=get('IM_all');base=get('v12_primary');out=dict(slice=sl,complete=primary['states']==control['states']==3)
  for baseline,b in [('IM_all',control),('v12_primary',base)]:
   rr=[r for r in matches if r['partition']=='query' and r['slice']==sl and r['score']=='GU_all' and r['baseline']==baseline and r['mode']=='actual_damage'];gain=100*(np.mean([r['Recovery'] for r in rr])-b['rec_at_nar1_mean']) if len(rr)==3 and b['states']==3 else None;support=sum(r['R']>=r['baseline_R'] for r in rr);out[baseline+'_same_damage_rec_gain_pp']=gain;out[baseline+'_nondecreasing_seeds']=support
  pg=primary['pauc_norm_mean']-control['pauc_norm_mean'] if out['complete'] else None;out['pauc_gain_vs_IM_pp']=100*pg if pg is not None else None;out['mechanism_passed']=bool(out['complete'] and out['IM_all_same_damage_rec_gain_pp']>=.2 and pg>=0 and out['IM_all_nondecreasing_seeds']>=2)
  cor=[r for r in matches if r['partition']=='query' and r['slice']==sl and r['score']=='GU_all' and r['baseline']=='v12_primary' and r['mode']=='corrections'];ok=len(cor)==3 and all(r['feasible'] for r in cor);wr=float(np.mean([r['W'] for r in cor]))/base['other_wrong_accepted_mean'] if ok else None;dc=float(np.mean([r['D'] for r in cor]))-base['damaged_mean'] if ok else None;out.update(same_v12_R_feasible=ok,same_v12_R_W_ratio=wr,same_v12_R_D_change=dc)
  if primary['states']==3:
   out['v12_pauc_gain_pp']=100*(primary['pauc_norm_mean']-base['pauc_norm_mean']);out['v12_rec1_gain_pp']=100*(primary['rec_at_nar1_mean']-base['rec_at_nar1_mean']);out['promotion_passed']=bool(out['v12_pauc_gain_pp']>=0 and out['v12_rec1_gain_pp']>=-.1 and (out['v12_primary_same_damage_rec_gain_pp']>=.2 or (ok and wr<=.95 and dc<=0)))
  else:out['promotion_passed']=False
  decisions.append(out)
 decision=dict(primary='GU_all',mechanism_passed=all(r['mechanism_passed'] for r in decisions),promotion_passed=all(r['promotion_passed'] for r in decisions),slices=decisions,gradient_unseen_only=True,strict_out_of_model=False,integrated=False)
 save('summary.json',summary);csvout('summary.csv',summary);save('registered_decisions.json',decision)
 fmt=lambda v:'NA' if v is None else f'{v:.3f}'
 lines=['# 梯度未见纠错元监督检验 v16','',f'主臂GU_all机理门槛：{decision["mechanism_passed"]}；超越v12方法门槛：{decision["promotion_passed"]}。模型和候选固定；不因其他臂更好替换主臂。','', '## 资格与角色','', '官方valid未参与骨干/辅助编码器梯度，但参与过历史checkpoint/温度选择；本轮是gradient-unseen检验，不是严格模型外交叉拟合或新确认。已观察valid现在显式分为元监督612/校准510/源诊断612，内容组互斥；元训练与校准不得再称独立验证。IM元监督使用612既有源train行，各主要臂标签预算匹配。','', '## query低NAR诊断前沿','', '三seed均值；缺seed均值NA。pAUC/.01为FPR<=.01区间平均候选TPR，FPR分母全部frozen正确者。query标签选出的前沿不可部署。','', '|切片|方法|状态|pAUC/.01 %|Recovery@NAR1 %|W|W/R|','|---|---|---:|---:|---:|---:|---:|']
 for r in [r for r in summary if r['partition']=='query']:lines.append('|'+ '|'.join([r['slice'],r['score'],str(r['states']),fmt(100*r['pauc_norm_mean']) if r['pauc_norm_mean'] is not None else 'NA',fmt(100*r['rec_at_nar1_mean']) if r['rec_at_nar1_mean'] is not None else 'NA',fmt(r['other_wrong_accepted_mean']),fmt(r['other_per_correction_mean'])])+'|')
 lines+=['','## 匹配实际损伤预算','','|切片|对照预算|方法|Recovery增量pp|不下降seed|','|---|---|---|---:|---:|']
 for sl in C['slices']:
  for baseline in ['IM_all','v12_primary']:
   b=next(r for r in summary if r['partition']=='query' and r['slice']==sl and r['score']==baseline)
   for arm in C['arms']:
    rr=[r for r in matches if r['partition']=='query' and r['slice']==sl and r['baseline']==baseline and r['score']==arm and r['mode']=='actual_damage'];gain=100*(np.mean([r['Recovery'] for r in rr])-b['rec_at_nar1_mean']) if len(rr)==3 and b['states']==3 else None;lines.append('|'+ '|'.join([sl,baseline,arm,fmt(gain),str(sum(r['R']>=r['baseline_R'] for r in rr))])+'|')
 lines+=['','## 冻结源阈值直接迁移','','以下仅源calibration阈值，未用query调阈值；满足观察NAR并非风险保证。','','|切片|方法|状态|NAR均值 %|NAR最大 %|Recovery %|W|','|---|---|---:|---:|---:|---:|---:|']
 for sl in C['slices']:
  for score in C['scores']:
   rr=[r for r in policies if r['partition']=='query' and r['slice']==sl and r['score']==score];vals=[100*np.mean([r['NAR'] for r in rr]),100*max(r['NAR'] for r in rr),100*np.mean([r['Recovery'] for r in rr]),np.mean([r['W'] for r in rr])] if len(rr)==3 else [None]*4;lines.append('|'+ '|'.join([sl,score,str(len(rr))]+[fmt(v) for v in vals])+'|')
 lines+=['','## 未用于选模的源诊断','','|切片|方法|状态|pAUC/.01 %|Recovery@NAR1 %|','|---|---|---:|---:|---:|']
 for r in [r for r in summary if r['partition']=='diagnostic']:lines.append('|'+ '|'.join([r['slice'],r['score'],str(r['states']),fmt(100*r['pauc_norm_mean']) if r['pauc_norm_mean'] is not None else 'NA',fmt(100*r['rec_at_nar1_mean']) if r['rec_at_nar1_mean'] is not None else 'NA'])+'|')
 lines+=['','## 可行性','','```json',json.dumps(feas,ensure_ascii=False,indent=2),'```','','## 预注册判决','','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','','## 审核与限制','','preflight三个源模型/两个固定LoRA/辅助原始输出与历史完整2160valid逐位复现。每个新缓存前256行独立重提取，全部模型参数与BN缓冲冻结；所有拟合模型/source/query分数与阈值及计数复算，散列冻结清单见artifacts。原始数据只读，源训练清单未扩大，无骨干/LoRA/辅助训练。','','GU与IM仍可能在错误构成/类别难度/先前选模影响上不同。原clean与合成的消融不能独立消除这些因素；不能以此证明记忆的因果作用。源诊断不是严格新确认，历史标签暴露保留。清洁源错误数量可能过少，GU_clean不足必须记不可行而非扩样。重复扰动不是独立样本，三seed共用query，微小增幅不能作显著性结论。','','Version045/USA仅已观察开发，沿用内容支持102类身份假设，非官方映射；不能概括所有时间漂移场景。WTT/AWF关闭，未部署，未自动新增交叉拟合训练。']
 (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 with np.load(RUN/'artifacts/frontiers.npz') as z:
  fig,axes=plt.subplots(1,2,figsize=(11,4));grid=np.linspace(0,.01,201)
  for ax,sl in zip(axes,C['slices']):
   for arm in ['v12_primary','IM_all','GU_all','GU_aug','GU_clean']:
    ys=[]
    for seed in C['seeds']:
     key=f'query_{seed}_{sl}_{arm}'
     if key not in z:continue
     rr=next(r for r in cells if r['partition']=='query' and r['seed']==seed and r['slice']==sl and r['score']==arm);p=z[key];x=p[:,2]/rr['frozen_correct'];y=p[:,1]/rr['frozen_wrong'];last=np.flatnonzero(np.r_[x[1:]!=x[:-1],True]);ys.append(y[last][np.maximum(np.searchsorted(x[last],grid,side='right')-1,0)])
    if len(ys)==3:ax.step(100*grid,100*np.mean(ys,0),where='post',label=arm)
   ax.set(title=sl,xlabel='NAR budget (%)',ylabel='Recovery (%)',xlim=(0,1));ax.legend(fontsize=7)
  fig.tight_layout()
  for ext in ['png','svg','pdf']:fig.savefig(RUN/f'figure_frontier.{ext}',dpi=160)
  plt.close(fig)
 print(json.dumps(decision,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
