"""Run-local 2x2 generator/head training. All conditions share audited inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import json,hashlib,time,argparse
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,f1_score
from model import MultiViewModel
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
def metric(y,p):
 cm=np.bincount(y*102+p,minlength=10404).reshape(102,102);den=cm.sum(0)+cm.sum(1)
 return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0).mean())}
def new_model(kind):
 mode,head=kind.split('_');return MultiViewModel(generator_mode=mode,head=head).cuda()
def logits(m,e,idx):return m(e['timestamps'][idx],e['tam'][idx])
def predict(m,e,bs=64):
 m.eval();out=[]
 with torch.inference_mode():
  for i in range(0,len(e['labels']),bs):out.append(logits(m,e,slice(i,i+bs)).cpu().numpy())
 return np.concatenate(out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
 c=json.loads((R/'config.json').read_text());assert c['status']=='frozen' and a.kind in c['conditions'] and a.seed in c['seeds'];assert c['roles']==['source','valid'] and not c['future_access']
 for p,h in json.loads((R/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 raw=torch.load(R/'artifacts/prepared.pt',map_location='cpu',weights_only=False);data={r:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for r,e in raw.items()};del raw
 key=f'{a.kind}_{a.seed}';out=R/'artifacts'/key;out.mkdir(exist_ok=False);m=new_model(a.kind);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};savet(out/'initial_state.pt',initial)
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])
 n=len(data['source']['labels']);bs=c['batch_size'];steps=c['optimizer_steps'];pieces=[];count=0;cycle=0
 while count<steps*bs:
  pieces.append(torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000+cycle)));count+=n;cycle+=1
 indices=torch.cat(pieces)[:steps*bs].reshape(steps,bs);savet(out/'index_stream.pt',indices)
 diag={k:v[:32] for k,v in data['source'].items()}
 with torch.inference_mode():initial_tokens={k:v.clone() for k,v in m.generator(diag['timestamps'],diag['tam']).items() if k in ['packet','time']}
 labels={r:e['labels'].cpu().numpy() for r,e in data.items()};start=time.monotonic();history=[];best=-1.;beststep=0;loss_sum=0.;prev=0
 print(json.dumps({'task':key,'parameters_total':sum(p.numel() for p in m.parameters()),'parameters_trainable':sum(p.numel() for p in m.parameters() if p.requires_grad),'steps':steps}),flush=True)
 for step in range(1,steps+1):
  if time.monotonic()-start>c['job_seconds']:raise TimeoutError(key+' budget exceeded')
  lr=next(seg[2] for seg in c['lr_schedule'] if seg[0]<=step<=seg[1])
  for g in opt.param_groups:g['lr']=lr
  m.train();idx=indices[step-1].cuda();opt.zero_grad(set_to_none=True);z=logits(m,data['source'],idx);loss=F.cross_entropy(z,data['source']['labels'][idx]);assert torch.isfinite(loss);loss.backward()
  if step==1:
   grad={k:float(p.grad.norm()) if p.grad is not None else None for k,p in m.generator.named_parameters()};save(out/'generator_first_gradients.json',grad)
   if a.kind.startswith('learned'):assert all(v is not None and np.isfinite(v) and v>0 for v in grad.values())
   else:assert all(v is None for v in grad.values()) and not any(p.requires_grad for p in m.generator.parameters())
  opt.step();loss_sum+=float(loss.detach())
  if step%100==0:save(out/'progress.json',{'task':key,'step':step,'total_steps':steps,'best_step':beststep,'elapsed_seconds':time.monotonic()-start})
  if step in c['eval_steps']:
   # Select only by fixed valid accuracy; source scores at the same checkpoint.
   pred={r:predict(m,e,c['eval_batch_size']) for r,e in data.items()};scores={r:metric(labels[r],z.argmax(1)) for r,z in pred.items()}
   row={'step':step,'lr':lr,'train_ce':loss_sum/(step-prev),'elapsed_seconds':time.monotonic()-start,**scores};loss_sum=0.;prev=step
   with torch.inference_mode():
    tokens=m.generator(diag['timestamps'],diag['tam']);row['generator_token_delta_rms']={k:float((tokens[k]-initial_tokens[k]).square().mean().sqrt()) for k in initial_tokens}
   if a.kind.startswith('fixed'):assert all(v==0 for v in row['generator_token_delta_rms'].values())
   else:assert all(v>0 for v in row['generator_token_delta_rms'].values())
   if scores['valid']['accuracy']>best:
    best=scores['valid']['accuracy'];beststep=step;savet(R/'checkpoints'/f'{key}_best.pt',{'state_dict':m.state_dict(),'step':step,'config_sha256':sha(R/'config.json')})
   history.append(row);save(out/'history.json',history);savet(R/'checkpoints'/f'{key}_latest.pt',{'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'history':history,'config_sha256':sha(R/'config.json')});print(json.dumps({'task':key,**row}),flush=True)
 last={r:predict(m,e,c['eval_batch_size']) for r,e in data.items()};np.savez_compressed(out/'predictions_last.npz',**{r:z.argmax(1) for r,z in last.items()})
 ck=torch.load(R/'checkpoints'/f'{key}_best.pt',weights_only=True,map_location='cuda');m.load_state_dict(ck['state_dict']);pred={r:predict(m,e,c['eval_batch_size']) for r,e in data.items()};np.savez_compressed(out/'predictions_best.npz',**{r:z.argmax(1) for r,z in pred.items()});np.savez_compressed(out/'logits_best.npz',**pred)
 del m,opt;torch.cuda.empty_cache();reload=new_model(a.kind);reload.load_state_dict(ck['state_dict']);verified=[]
 for r,e in data.items():
  pr=predict(reload,e,c['eval_batch_size']).argmax(1);assert np.array_equal(pr,pred[r].argmax(1));sc=metric(labels[r],pr);assert abs(sc['accuracy']-accuracy_score(labels[r],pr))<1e-12;assert abs(sc['macro_f1']-f1_score(labels[r],pr,labels=np.arange(102),average='macro',zero_division=0))<1e-12;verified.append(r)
 gdelta={}
 for k,v in reload.generator.state_dict().items():gdelta[k]=float((v.cpu()-initial['generator.'+k]).square().sum())
 if a.kind.startswith('fixed'):assert sum(gdelta.values())==0
 else:assert sum(gdelta.values())>0
 report={'task':key,'kind':a.kind,'seed':a.seed,'best_step':beststep,'steps_completed':steps,'validation_opportunities':len(history),'parameters_total':sum(p.numel() for p in reload.parameters()),'parameters_trainable':sum(p.numel() for p in reload.parameters() if p.requires_grad),'generator_parameter_delta_l2':sum(gdelta.values())**.5,'elapsed_seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'verified_roles':verified,'best':{r:metric(labels[r],z.argmax(1)) for r,z in pred.items()},'last':{r:metric(labels[r],z.argmax(1)) for r,z in last.items()}}
 save(out/'report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
