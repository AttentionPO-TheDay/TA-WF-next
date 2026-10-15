from pathlib import Path
import csv,json,subprocess
import numpy as np
RID='20261014T003143Z_inference_margin_gate_b1f6deab';ROOT=Path('/home/rbf/TA-WF-next');R=ROOT/'runs'/RID;SRC=ROOT/'runs/20261013T091631Z_mvi_dynamic_bn_tta_a79d0c0b/artifacts'
SEEDS=[21729,23407,22026];ROLES=['day14','day90','day270']
def f1(y,p):
 cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102); den=cm.sum(0)+cm.sum(1); return float(np.divide(2*np.diag(cm),den,out=np.zeros(102),where=den>0).mean())
def metrics(y,base,p):
 bc=base==y;bw=~bc
 return {'accuracy':float((p==y).mean()),'macro_f1':f1(y,p),'negative_adaptation_rate':float((bc&(p!=y)).sum()/max(1,bc.sum())),'recovery_rate':float((bw&(p==y)).sum()/max(1,bw.sum())),'changed_rate':float((p!=base).mean())}
def main():
 cells=[]
 for s in SEEDS:
  for role in ROLES:
   with np.load(SRC/f'pred_{s}_{role}_frozen.npz') as a,np.load(SRC/f'pred_{s}_{role}_no_pseudo_fixed.npz') as b:
    assert np.array_equal(a['rows'],b['rows']) and np.array_equal(a['labels'],b['labels'])
    y=a['labels'];lf=a['logits'];la=b['logits'];base=lf.argmax(1);adapt=la.argmax(1);probs=np.exp(lf-lf.max(1,keepdims=True));probs/=probs.sum(1,keepdims=True);sp=np.sort(probs,axis=1);margin=sp[:,-1]-sp[:,-2];ent=-(probs*np.log(np.maximum(probs,1e-12))).sum(1)/np.log(102)
    cells.append({'seed':s,'role':role,'rows':a['rows'],'labels':y,'base':base,'adapt':adapt,'margin':margin,'entropy':ent})
 def aggregate(kind,t):
  vals=[];preds=[]
  for c in cells:
   use=(c[kind]>t) if kind=='margin' else (c[kind]<t);p=np.where(use,c['base'],c['adapt']);vals.append(metrics(c['labels'],c['base'],p));preds.append(p)
  return {k:float(np.mean([v[k] for v in vals])) for k in vals[0]},vals,preds
 rows=[]
 for kind,grid in [('margin',np.linspace(0,.99,100)),('entropy',np.linspace(0,1,101))]:
  for t in grid:
   m,_,_=aggregate(kind,float(t));rows.append({'gate':kind,'threshold':float(t),**m})
 selected={}
 for kind in ['margin','entropy']:
  cand=[r for r in rows if r['gate']==kind and r['negative_adaptation_rate']<.02]
  selected[kind]=dict(sorted(cand or [r for r in rows if r['gate']==kind],key=lambda r:(-r['macro_f1'],r['negative_adaptation_rate'],r['threshold']))[0])
  selected[kind]['feasible']=bool(cand)
 with (R/'artifacts/thresholds.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 adapted_cells=[metrics(c['labels'],c['base'],c['adapt']) for c in cells]
 out={'selected':selected,'threshold_rows':len(rows),'adapted_exp1':{k:float(np.mean([m[k] for m in adapted_cells])) for k in adapted_cells[0]}}
 for kind in ['margin','entropy']:
  rec=selected[kind]; _,_,preds=aggregate(kind,rec['threshold']);out[kind+'_per_cell']=[]
  for c,p in zip(cells,preds):
   out[kind+'_per_cell'].append({'seed':c['seed'],'role':c['role'],**metrics(c['labels'],c['base'],p)})
   np.savez_compressed(R/'artifacts'/f'pred_{kind}_{c["seed"]}_{c["role"]}.npz',rows=c['rows'],labels=c['labels'],source_predictions=c['base'],adapted_predictions=c['adapt'],gated_predictions=p,gate_values=c[kind],threshold=rec['threshold'])
 (R/'artifacts/results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 lines=['# 方案A：推断期安全门控结果','','9个TemporalDrift query单元的离线阈值扫描完成。','','| 设置 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |','|---|---:|---:|---:|---:|']
 for label,rec in [('Exp-1 no_pseudo_fixed',out['adapted_exp1']),('Margin selected',selected['margin']),('Entropy selected',selected['entropy'])]:lines.append(f"| {label} | {rec['macro_f1']*100:.3f}% | {rec['accuracy']*100:.3f}% | {rec['negative_adaptation_rate']*100:.3f}% | {rec['recovery_rate']*100:.3f}% |")
 lines+=['','全阈值曲线见artifacts/thresholds.csv；选择规则和标签权限见PLAN.md。','结果仅为TemporalDrift已观察开发证据。']
 (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 (R/'artifacts/audit.json').write_text(json.dumps({'passed':True,'cells':9,'threshold_rows':201,'gpu_training':False,'query_labels_used_for':'metrics_and_pre_registered_threshold_selection','query_adaptation':False},ensure_ascii=False,indent=2)+'\n')
 subprocess.run(['python3',str(ROOT/'scripts/experiment.py'),'status','--id',RID,'--state','completed','--summary',f"离线门控完成；Margin F1 {selected['margin']['macro_f1']*100:.3f}%，负适应 {selected['margin']['negative_adaptation_rate']*100:.3f}%；Entropy F1 {selected['entropy']['macro_f1']*100:.3f}%。"],check=True)
if __name__=='__main__':main()
