"""Prespecified paired contrasts; valid development evidence only."""
import subprocess
from common import *

def main():
 c=config_read();reports={};errors=[];checked=0
 for cond in c['conditions']:
  reports[cond]=[]
  for seed in c['seeds']:
   out=RUN/'artifacts'/f'{cond}_{seed}'
   try:
    r=json.loads((out/'report.json').read_text());v=json.loads((out/'verification.json').read_text())
    assert r['status']=='completed' and v['complete'];checked+=v['prediction_sets_checked'];reports[cond].append(r)
   except Exception as exc:errors.append(f'{cond}_{seed}: {type(exc).__name__}: {exc}')
 if not errors:
  for i in range(3):
   if reports[c['conditions'][0]][i]['initial_classifier_sha256']!=reports[c['conditions'][1]][i]['initial_classifier_sha256']:errors.append('paired initialization mismatch')
 complete=not errors and checked==12
 write_json(RUN/'artifacts/integrity.json',{'complete':complete,'errors':errors,'prediction_sets_checked':checked,'expected_prediction_sets':12,'original_prediction_sets_replayed':sum(2 for rs in reports.values() for r in rs),'future_access':False})
 summary={'complete':complete,'errors':errors,'conditions':{}}
 lines=['# 固定生成器后重训分类器：结果','',f'状态：{"completed" if complete else "incomplete"}。','', '本轮固定B/C生成器，仅从头训练同结构分类器。所有F1均为Macro-F1，三个seed共享开发valid。','', '| 生成器 | 新classifier source F1 | 新classifier valid F1 | 原classifier valid F1 |','|---|---:|---:|---:|']
 for cond,rs in reports.items():
  if len(rs)!=3:continue
  means={role:{k:float(np.mean([r[role][k] for r in rs])) for k in ('accuracy','macro_f1')} for role in ('source','valid')}
  means['original_valid_f1']=float(np.mean([r['original_replay']['valid']['macro_f1'] for r in rs]));summary['conditions'][cond]=means
  lines.append(f"| {cond} | {means['source']['macro_f1']*100:.3f}% | {means['valid']['macro_f1']*100:.3f}% | {means['original_valid_f1']*100:.3f}% |")
 if complete:
  cr,br=(reports[cond] for cond in c['conditions'])
  delta=lambda left,right,role,k: [(l[role][k]-r[role][k])*100 for l,r in zip(left,right)]
  main_f=delta(br,cr,'valid','macro_f1');main_a=delta(br,cr,'valid','accuracy');src=delta(br,cr,'source','macro_f1')
  recovery=lambda rs,k:[(r['valid'][k]-r['original_replay']['valid'][k])*100 for r in rs]
  bf=recovery(br,'macro_f1');ba=recovery(br,'accuracy');cf=recovery(cr,'macro_f1')
  old=[(b['original_replay']['valid']['macro_f1']-v['original_replay']['valid']['macro_f1'])*100 for b,v in zip(br,cr)]
  positive=lambda f,a:np.mean(f)>=1 and all(x>0 for x in f) and np.mean(a)>=0
  advantage=bool(positive(main_f,main_a));disadvantage=bool(positive([-x for x in main_f],[-x for x in main_a]));recovered=bool(positive(bf,ba))
  interaction=bool(advantage and recovered and np.mean(old)<0)
  bias=bool(disadvantage and np.mean(src)>=0 and np.mean(np.array(src)-np.array(main_f))>0)
  decision='SUPPORT_JOINT_OPTIMIZATION_LIMITATION' if interaction else 'SUPPORT_REPRESENTATION_TRAIN_BIAS' if bias else 'INCONCLUSIVE'
  summary.update(decision=decision,main_B_minus_C_f1_pp=main_f,main_accuracy_pp=main_a,source_f1_pp=src,B_recovery_f1_pp=bf,C_recovery_f1_pp=cf,difference_in_differences_pp=(np.array(bf)-np.array(cf)).tolist(),fixed_B_advantage=advantage,fixed_B_disadvantage=disadvantage,B_recovery=recovered,joint_optimization_support=interaction,representation_train_bias_support=bias)
  for title,values in [('新B−新C valid F1',main_f),('新B−原B valid F1',bf),('新C−原C valid F1',cf),('恢复幅度差分',summary['difference_in_differences_pp']),('新B−新C source F1',src)]:lines.extend(['',f'{title}：逐seed {values} pp；平均 {np.mean(values):+.3f} pp。'])
  lines.extend(['',f'预定裁决：{decision}。固定B优势={advantage}，固定B劣势={disadvantage}，B恢复={recovered}。','', '裁决表示在当前分类器与训练预算下的支持程度，不是唯一因果证明。若未达门槛保持未决，不据此追加搜索。'])
  text_summary=f'固定B/C生成器重训完成：{decision}；B−C valid F1 {np.mean(main_f):+.3f}pp，B恢复 {np.mean(bf):+.3f}pp；12/12新预测重载核验。'
 else:text_summary=f'固定生成器重训未完整通过；{checked}/12预测已核验，见run日志与integrity.json。';lines.extend(['','错误：'+repr(errors)])
 lines.extend(['',f'完成核验：{checked}/12组新classifier预测；正式缓存逐组重放原预测并匹配上轮已核验token；配对初始化/冻结边界/选模历史与锚点散列检查见artifacts。','', '限制：生成器checkpoint此前已按同一valid选择；C随机生成器不是成熟分类基线。B生成器阶段已有额外监督训练，仅本轮classifier预算匹配。未访问未来日期、外部数据，未进行预训练/TTA。运行代码与配置版本见artifacts/freeze.json。'])
 write_json(RUN/'artifacts/summary.json',summary);(RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state','completed' if complete else 'stopped','--summary',text_summary],check=True)
 status=ROOT/'STATUS.md';s=status.read_text();prefix=f'当前固定生成器重训（2026-09-26）：`{RUN.name}`。'
 rows=s.splitlines();rows=[prefix+text_summary if line.startswith(prefix) else line for line in rows];status.write_text('\n'.join(rows)+'\n')
 print(text_summary,flush=True)
 if not complete:raise SystemExit(2)
if __name__=='__main__':main()
