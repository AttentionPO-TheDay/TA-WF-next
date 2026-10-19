from runtime import *
import numpy as np

def main():
 cells=json.loads((RUN/'artifacts/cells.json').read_text());matches=json.loads((RUN/'artifacts/matched_frontiers.json').read_text());policies=json.loads((RUN/'artifacts/source_policy_target.json').read_text());summ=[];decisions=[]
 for sl in C['slices']:
  for arm in C['scores']:
   rr=[r for r in cells if r['slice']==sl and r['score']==arm];out=dict(slice=sl,score=arm,states=len(rr))
   for k in ['pauc_norm','rec_at_nar1','step_mean_recovery','attained_nar1','corrected','damaged','other_wrong_accepted','other_per_correction','oracle_recovery']:
    vv=[r[k] for r in rr if r[k] is not None];out[k+'_mean']=float(np.mean(vv)) if len(vv)==3 else None;out[k+'_sd']=float(np.std(vv,ddof=1)) if len(vv)==3 else None
   summ.append(out)
 for arm in C['arms']:
  parts=[]
  for sl in C['slices']:
   base=next(r for r in summ if r['slice']==sl and r['score']=='v12_primary');s=next(r for r in summ if r['slice']==sl and r['score']==arm);risk=[r for r in matches if r['slice']==sl and r['score']==arm and r['mode']=='v12_actual_damage'];cor=[r for r in matches if r['slice']==sl and r['score']==arm and r['mode']=='v12_corrections'];complete=s['states']==3;feasible=len(cor)==3 and all(r['feasible'] for r in cor)
   gain=100*(np.mean([r['Recovery'] for r in risk])-base['rec_at_nar1_mean']) if len(risk)==3 else None;support=sum(r['R']>=r['baseline_R'] for r in risk);wr=float(np.mean([r['W'] for r in cor]))/base['other_wrong_accepted_mean'] if feasible else None;dc=float(np.mean([r['D'] for r in cor]))-base['damaged_mean'] if feasible else None;cleanup=feasible and wr<=.95 and dc<=0;pg=s['pauc_norm_mean']-base['pauc_norm_mean'] if complete else None;rg=100*(s['rec_at_nar1_mean']-base['rec_at_nar1_mean']) if complete else None;passed=complete and pg>=0 and rg>=-.1 and (gain>=.2 or cleanup) and support>=2
   parts.append(dict(slice=sl,passed=bool(passed),complete=complete,pauc_gain_pp=100*pg if pg is not None else None,rec1_gain_pp=rg,same_damage_rec_gain_pp=gain,nondecreasing_same_damage_seeds=support,same_correction_feasible=feasible,same_correction_W_ratio=wr,same_correction_D_change=dc))
  decisions.append(dict(arm=arm,passed=all(r['passed'] for r in parts),slices=parts))
 decision=dict(primary='ABC',primary_passed=next(d['passed'] for d in decisions if d['arm']=='ABC'),single_method_tests=['A','B','C'],all_arms=decisions,integrated=False,target_role='observed_diagnostic')
 save('summary.json',summ);csvout('summary.csv',summ);save('registered_decisions.json',decision)
 # Full factorial differences; source-standardization fixed within a state. These are descriptive paired contrasts.
 contrasts=[]
 for sl in C['slices']:
  for f,pairs in [('A',[('A','ce'),('AB','B'),('AC','C'),('ABC','BC')]),('B',[('B','ce'),('AB','A'),('BC','C'),('ABC','AC')]),('C',[('C','ce'),('AC','A'),('BC','B'),('ABC','AB')])]:
   for new,old in pairs:
    a=next(r for r in summ if r['slice']==sl and r['score']==new);b=next(r for r in summ if r['slice']==sl and r['score']==old);contrasts.append(dict(slice=sl,factor=f,new=new,old=old,pauc_gain_pp=100*(a['pauc_norm_mean']-b['pauc_norm_mean']) if a['states']==b['states']==3 else None,rec1_gain_pp=100*(a['rec_at_nar1_mean']-b['rec_at_nar1_mean']) if a['states']==b['states']==3 else None))
 save('factorial_contrasts.json',contrasts);csvout('factorial_contrasts.csv',contrasts)
 fmt=lambda x:'NA' if x is None else f'{x:.3f}'
 lines=['# A/B/C低风险候选清洗 v15','',f'组合主臂ABC通过：{decision["primary_passed"]}。全部A/B/C单独及组合均预注册；不以最佳分支替换ABC。固定v9候选，零神经网络训练/新前向，复用缓存。','', '## 三seed诊断前沿','', 'pAUC/.01为低风险区间平均候选TPR，FPR分母是全部frozen正确者；不是McClish标准化。阈值由query标签确定的前沿仅诊断，不部署。','', '|切片|方法|状态|pAUC/.01 %|Recovery@NAR1 %|W|W/R|','|---|---|---:|---:|---:|---:|---:|']
 for r in summ:lines.append('|'+ '|'.join([r['slice'],r['score'],str(r['states']),fmt(100*r['pauc_norm_mean']) if r['pauc_norm_mean'] is not None else 'NA',fmt(100*r['rec_at_nar1_mean']) if r['rec_at_nar1_mean'] is not None else 'NA',fmt(r['other_wrong_accepted_mean']),fmt(r['other_per_correction_mean'])])+'|')
 lines+=['','## 同v12实际损伤预算与同纠错匹配','','|切片|方法|同损伤Recovery增量pp|不下降seed|同纠错可行seed|同纠错D|同纠错W|','|---|---|---:|---:|---:|---:|---:|']
 for sl in C['slices']:
  for arm in C['arms']:
   d=next(x for x in decisions if x['arm']==arm);v=next(x for x in d['slices'] if x['slice']==sl);rr=[r for r in matches if r['slice']==sl and r['score']==arm and r['mode']=='v12_corrections'];ff=[r for r in rr if r['feasible']];ok=len(ff)==3;lines.append('|'+ '|'.join([sl,arm,fmt(v['same_damage_rec_gain_pp']),str(v['nondecreasing_same_damage_seeds']),str(len(ff)),fmt(float(np.mean([r['D'] for r in ff]))) if ok else 'NA',fmt(float(np.mean([r['W'] for r in ff]))) if ok else 'NA'])+'|')
 lines+=['','## 源阈值直接迁移','','阈值统一按源valid的R-D-.25W选取，未使用query标签；源校准不保证目标风险。低NAR也可能来自该效用主动选择较少改写。','','|切片|方法|平均NAR %|最大NAR %|Recovery %|W|','|---|---|---:|---:|---:|---:|']
 for sl in C['slices']:
  for arm in C['scores']:
   rr=[r for r in policies if r['slice']==sl and r['score']==arm]
   if len(rr)==3:lines.append('|'+ '|'.join([sl,arm,fmt(100*np.mean([r['NAR'] for r in rr])),fmt(100*max(r['NAR'] for r in rr)),fmt(100*np.mean([r['Recovery'] for r in rr])),fmt(float(np.mean([r['W'] for r in rr])))])+'|')
 lines+=['','## reference覆盖与训练迁移检查','','```json',json.dumps(json.loads((RUN/'artifacts/density_diagnostics.json').read_text()),ensure_ascii=False,indent=2),'```','','LOPO逐扰动/逐seed结果见artifacts/lopo.csv；sourcevalid与query的分支单独/融合结果见fusion_source.json和fusion_target.json。目标结果未用于挑臂或重训。原clean组几乎没有改写，可能无纠错候选；对应pAUC未定义，保留null，不填零。','','## 判决','','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','','## 局限与完成检查','','已有三源seed，共享query样本；六状态不是六个独立数据集。微小样本计数改变不能视为显著性证据。五扰动共享原trace，非独立增样；源骨干见过train，元训练不是骨干模型外预测。LOPO仅扰动组泛化诊断。C密度比依赖边缘覆盖和条件关系不变假设，非真实纠错概率或风险保证；OOF按trace分组只防域分类器与元训练内部泄漏，不能解除骨干记忆。','','已观察Version045/USA沿用内容支持标签对应假设，非官方类别映射；不是独立确认。WTT/AWF关闭。此次目标表现不返调已冻结方法，未部署或扩大矩阵。完整重载审计见reload_audit.json；散列见freeze.json/source_fit_freeze.json/delivery_audit.json。']
 (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 with np.load(RUN/'artifacts/frontiers.npz') as z:
  fig,axes=plt.subplots(1,2,figsize=(12,4));grid=np.linspace(0,.01,201)
  for ax,sl in zip(axes,C['slices']):
   for arm in ['v12_primary','v14_primary']+C['arms']:
    ys=[]
    for seed in C['seeds']:
     k=f'{seed}_{sl}_{arm}'
     if k not in z:continue
     rr=next(r for r in cells if r['seed']==seed and r['slice']==sl and r['score']==arm);p=z[k];x=p[:,2]/rr['frozen_correct'];y=p[:,1]/rr['frozen_wrong'];last=np.flatnonzero(np.r_[x[1:]!=x[:-1],True]);ys.append(y[last][np.maximum(np.searchsorted(x[last],grid,side='right')-1,0)])
    if len(ys)==3:ax.step(grid*100,np.mean(ys,0)*100,where='post',label=arm,alpha=.8)
   ax.set(title=sl,xlabel='NAR budget (%)',ylabel='Recovery (%)',xlim=(0,1));ax.legend(fontsize=6,ncol=2)
  fig.tight_layout()
  for ext in ['png','svg','pdf']:fig.savefig(RUN/f'figure_frontier.{ext}',dpi=160)
  plt.close(fig)
 print(json.dumps(decision,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
