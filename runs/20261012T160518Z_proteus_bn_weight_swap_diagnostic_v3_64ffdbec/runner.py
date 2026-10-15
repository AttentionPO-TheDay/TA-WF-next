import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np, torch
import port_base as B
R,ROOT,C=B.R,B.ROOT,B.C; DEVICE=B.DEVICE
CONDS=C['conditions']
def save(name,x): (R/'artifacts'/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def registry(state,summary): subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',R.name,'--state',state,'--summary',summary],check=True)
def metric(y,p):
 cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102); den=cm.sum(0)+cm.sum(1); f=np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0); return {'accuracy':float((y==p).mean()),'macro_f1':float(f.mean())}
def cell(y,base,p):
 a=metric(y,p); bc=base==y; bw=~bc; return {**a,'negative_adaptation_rate':float((bc&(p!=y)).sum()/max(1,bc.sum())),'recovery_rate':float((bw&(p==y)).sum()/max(1,bw.sum())),'changed_rate':float((base!=p).mean())}
def main():
 torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 assert C['status']=='frozen' and os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available()
 donor=ROOT/C['adapted_run']; scores=[]; events=[]
 for seed in C['seeds']:
  base_model,_=B.load_model(seed); base={k:v.detach().cpu().clone() for k,v in base_model.state_dict().items()}; del base_model
  # source checkpoint is same across dates; strength025 checkpoint is date-specific and loaded below per role.
  for role in C['roles']:
   target=B.load_prepared(C['prepared'][role]); split=np.load(donor/'artifacts'/f'split_{role}.npz'); q=split['query_indices']; split.close(); y=target['labels'][q]
   ck=torch.load(donor/'checkpoints'/f'{seed}_{role}_strength_025.pt',map_location='cpu',weights_only=True)['state_dict']
   states={}
   bn_keys={k for k in base if k.endswith('running_mean') or k.endswith('running_var') or k.endswith('num_batches_tracked')}
   states['frozen']=base
   states['weights_only']={k:(base[k] if k in bn_keys else ck[k]) for k in base}
   states['bn_stats_only']={k:(ck[k] if k in bn_keys else base[k]) for k in base}
   states['full_strength025']=ck
   model,_=B.load_model(seed); frozen_logits=B.predict(model,target['timestamps'][q],target['tam'][q],128); frozen=frozen_logits.argmax(1); np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_frozen.npz',rows=target['rows'][q],labels=y,predictions=frozen,logits=frozen_logits)
   assert np.isfinite(frozen_logits).all()
   for cond in CONDS:
    model.load_state_dict(states[cond],strict=True); model.eval(); z=B.predict(model,target['timestamps'][q],target['tam'][q],128); p=z.argmax(1); assert np.isfinite(z).all();
    if cond=='full_strength025':
     with np.load(donor/'artifacts'/f'pred_{seed}_{role}_strength_025.npz') as old: assert np.array_equal(p,old['predictions'])
    row={'seed':seed,'role':role,'condition':cond,'rows':len(y),**cell(y,frozen,p)}; scores.append(row); np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_{cond}.npz',rows=target['rows'][q],labels=y,predictions=p,logits=z); events.append(row)
   del model
 save('scores.json',scores); lines=['# Proteus 0.25 倍权重/BN 统计交换诊断结果','','结果用于诊断，不把混合 checkpoint 的差值解释为可加的因果贡献。','','| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |','|---|---:|---:|---:|---:|']
 def avg(c,k): return float(np.mean([x[k] for x in scores if x['condition']==c]))
 for c in CONDS: lines.append(f"| {c} | {avg(c,'macro_f1')*100:.3f}% | {avg(c,'accuracy')*100:.3f}% | {avg(c,'negative_adaptation_rate')*100:.3f}% | {avg(c,'recovery_rate')*100:.3f}% |")
 for role in C['roles']:
  lines += ['',f'## {role}','','| 条件 | Macro-F1 | 负适应率 |','|---|---:|---:|']
  for c in CONDS:
   vals=[x for x in scores if x['condition']==c and x['role']==role]; lines.append(f"| {c} | {np.mean([x['macro_f1'] for x in vals])*100:.3f}% | {np.mean([x['negative_adaptation_rate'] for x in vals])*100:.3f}% |")
 (R/'RESULTS.md').write_text('\n'.join(lines)+'\n'); registry('completed','36个固定推理状态完成；权重与BN统计交换诊断结果见RESULTS.md。')
if __name__=='__main__':
 import os
 try: main()
 except BaseException as e: save('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}); registry('failed',f'BN/权重诊断失败：{type(e).__name__}: {e}；产物保留。'); raise
