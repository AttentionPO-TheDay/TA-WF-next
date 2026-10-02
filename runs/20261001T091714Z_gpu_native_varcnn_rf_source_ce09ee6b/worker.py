"""Bounded source-only native-recipe training; no upstream entrypoints executed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import json,hashlib,time,argparse
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,f1_score
from varcnn_native import VarCNNNative
from rf_native import RFNative
from keras_compat import KerasAdam, KerasCallbacks
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def savet(p,d):
 t=p.with_suffix('.tmp');torch.save(d,t);t.replace(p)
def model(kind):return (VarCNNNative(num_classes=102) if kind=='varcnn' else RFNative(102)).cuda()
def metric(y,p):
 cm=np.bincount(y*102+p,minlength=10404).reshape(102,102);den=cm.sum(0)+cm.sum(1)
 return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0).mean())}
def predict(m,x,bs):
 m.eval();out=[]
 with torch.inference_mode():
  for i in range(0,len(x),bs):out.append(m(x[i:i+bs]).cpu().numpy())
 return np.concatenate(out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['rf','varcnn'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
 c=json.loads((R/'config.json').read_text());assert c['status']=='frozen' and not c['future_access'] and a.seed in c['seeds']
 for p,h in json.loads((R/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.manual_seed(a.seed);np.random.seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 data=torch.load(R/'artifacts/native_prepared.pt',weights_only=False,map_location='cpu');feat='direction' if a.kind=='varcnn' else 'tam'
 xs={r:e[feat].float().cuda() for r,e in data.items()};ys={r:e['labels'].long().cuda() for r,e in data.items()};labels={r:y.cpu().numpy() for r,y in ys.items()};del data
 cfg=c[a.kind];key=f'{a.kind}_{a.seed}';out=R/'artifacts'/key;out.mkdir(exist_ok=False);m=model(a.kind)
 opt=(KerasAdam(m.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['epsilon']) if a.kind=='varcnn' else torch.optim.Adam(m.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['epsilon'],weight_decay=cfg['weight_decay']))
 callbacks=KerasCallbacks() if a.kind=='varcnn' else None
 # Var-CNN uses the explicitly ported old Keras Adam recurrence.
 n=len(xs['source']);order0=torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000));savet(out/'initial_order.pt',order0)
 init={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};savet(out/'initial_state.pt',init);del init
 best=-1.;bestepoch=0;plateau_best=-np.inf;plateau_wait=0;early_best=-np.inf;early_wait=0;history=[];start=time.monotonic();steps=0
 print(json.dumps({'task':key,'samples':n,'parameters':sum(p.numel() for p in m.parameters()),'max_epochs':cfg['epochs'],'budget_seconds':c['job_seconds']}),flush=True)
 stopreason='max_epochs'
 for epoch in range(1,cfg['epochs']+1):
  if a.kind=='rf':
   for g in opt.param_groups:g['lr']=cfg['lr']*(.2**((epoch-1)/30))
   order=torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000+epoch-1))
  else:order=order0
  m.train();tot=0.;correct=0;lr=opt.param_groups[0]['lr']
  for idx in order.split(cfg['batch_size']):
   if time.monotonic()-start>c['job_seconds']:raise TimeoutError(key+' job budget')
   idx=idx.cuda();opt.zero_grad(set_to_none=True);z=m(xs['source'][idx]);loss=F.cross_entropy(z,ys['source'][idx]);assert torch.isfinite(loss);loss.backward()
   if steps==0:
    grads={k:float(p.grad.norm()) for k,p in m.named_parameters() if p.grad is not None};assert all(np.isfinite(list(grads.values()))) and sum(grads.values())>0;save(out/'first_gradients.json',grads)
   opt.step();tot+=float(loss.detach())*len(idx);correct+=int((z.argmax(1)==ys['source'][idx]).sum());steps+=1
  row={'epoch':epoch,'steps':steps,'learning_rate':lr,'train_ce':tot/n,'online_accuracy':correct/n,'elapsed_seconds':time.monotonic()-start}
  shouldstop=False
  if epoch in cfg['eval_epochs']:
   z=predict(m,xs['valid'],cfg['eval_batch_size']);score=metric(labels['valid'],z.argmax(1));row['valid']=score;acc=score['accuracy']
   if acc>best:
    best=acc;bestepoch=epoch;savet(R/'checkpoints'/f'{key}_best.pt',{'state_dict':m.state_dict(),'epoch':epoch,'steps':steps,'config_sha256':sha(R/'config.json')})
   if a.kind=='varcnn':
    row['callbacks']=callbacks.step(acc,opt)
    shouldstop=row['callbacks']['stop']
    if shouldstop:stopreason='early_stopping'
  history.append(row);save(out/'history.json',history);save(out/'progress.json',{'task':key,'epoch':epoch,'steps':steps,'best_epoch':bestepoch,'elapsed_seconds':time.monotonic()-start})
  savet(R/'checkpoints'/f'{key}_latest.pt',{'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'steps':steps,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'history':history,'callbacks':callbacks.state_dict() if callbacks else None,'config_sha256':sha(R/'config.json')})
  print(json.dumps({'task':key,**row}),flush=True)
  if shouldstop:break
 last={r:predict(m,x,cfg['eval_batch_size']) for r,x in xs.items()};np.savez_compressed(out/'predictions_last.npz',**{r:z.argmax(1) for r,z in last.items()})
 ck=torch.load(R/'checkpoints'/f'{key}_best.pt',weights_only=True,map_location='cuda');m.load_state_dict(ck['state_dict']);bestz={r:predict(m,x,cfg['eval_batch_size']) for r,x in xs.items()};np.savez_compressed(out/'predictions_best.npz',**{r:z.argmax(1) for r,z in bestz.items()});np.savez_compressed(out/'logits_best.npz',**bestz)
 del m,opt;torch.cuda.empty_cache();reload=model(a.kind);reload.load_state_dict(ck['state_dict']);verified=[]
 for r,x in xs.items():
  p=predict(reload,x,cfg['eval_batch_size']).argmax(1);assert np.array_equal(p,bestz[r].argmax(1));v=metric(labels[r],p)
  assert abs(v['accuracy']-accuracy_score(labels[r],p))<1e-12 and abs(v['macro_f1']-f1_score(labels[r],p,labels=np.arange(102),average='macro',zero_division=0))<1e-12;verified.append(r)
 report={'task':key,'kind':a.kind,'seed':a.seed,'best_epoch':bestepoch,'epochs_completed':epoch,'steps_completed':steps,'validation_opportunities':sum('valid' in row for row in history),'stop_reason':stopreason,'elapsed_seconds':time.monotonic()-start,'parameters':sum(p.numel() for p in reload.parameters()),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'verified_roles':verified,'best':{r:metric(labels[r],z.argmax(1)) for r,z in bestz.items()},'last':{r:metric(labels[r],z.argmax(1)) for r,z in last.items()}}
 save(out/'report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
