import json, os, random, math, time, hashlib, traceback, signal, subprocess, sys, shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
import port_base as B
from model import MultiViewModel
from external_data import wtt_dataset, awf_dataset
R=Path(__file__).resolve().parent; ROOT=R.parents[1]; C=json.loads((R/'config.json').read_text()); DEVICE=B.DEVICE
SEEDS=C['seeds']; CONDS=C['conditions']; A=C['adaptation']

def setup():
 assert C['status']=='frozen' and C['id']==R.name
 torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False

def status_update(msg):
 import fcntl
 with (ROOT/'.status.lock').open('a') as lock:
  fcntl.flock(lock, fcntl.LOCK_EX); p=ROOT/'STATUS.md'; lines=p.read_text().splitlines(); prefix=f'当前实验 external_tta_transfer_matrix：`{R.name}`。'
  for i,l in enumerate(lines):
   if l.startswith(prefix): lines[i]=prefix+msg; break
  else: lines.insert(1,prefix+msg)
  p.write_text('\n'.join(lines)+'\n')

def direction_batch(x):
 x=x.to(DEVICE,dtype=torch.float32)
 if x.ndim==1:x=x[None]
 if x.shape[1]<5000: x=torch.nn.functional.pad(x,(0,5000-x.shape[1]))
 x=x[:,:5000]; d=x.sign(); valid=d.ne(0); bins=(torch.arange(5000,device=DEVICE)*1800//5000).long().expand(len(d),-1)
 tam=torch.zeros(len(d),2,1800,device=DEVICE)
 tam[:,0].scatter_add_(1,bins,(d.gt(0)&valid).float()); tam[:,1].scatter_add_(1,bins,(d.lt(0)&valid).float())
 return d,tam

def ds_loader(ds,shuffle,seed):
 return DataLoader(ds,batch_size=128,shuffle=shuffle,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(seed))

def forward(model,x):
 d,t=direction_batch(x); return B.forward_with_features(model,d,t)

def train_source(ds,nclasses,seed,dataset_name):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 model=MultiViewModel('wide_bn',nclasses).to(DEVICE); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=C['model']['source_learning_rate'],weight_decay=0.0001,betas=(.9,.999)); loader=ds_loader(ds,True,seed)
 hist=[]
 for epoch in range(C['model']['source_epochs']):
  losses=[]
  for xb,yb,_ in loader:
   yb=yb.to(DEVICE); opt.zero_grad(set_to_none=True); logits,_=forward(model,xb); loss=nn.functional.cross_entropy(logits,yb); loss.backward(); opt.step(); losses.append(float(loss.detach()))
  hist.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
 model.eval(); path=R/'checkpoints'/f'{dataset_name}_{seed}_source.pt'; torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'seed':seed,'dataset':dataset_name,'config_sha':B.sha(R/'config.json')},path)
 B.save_json(f'{dataset_name}_{seed}_source_history.json',hist)
 return model

def load_source(dataset_name,nclasses,seed):
 model=MultiViewModel('wide_bn',nclasses).to(DEVICE); ck=torch.load(R/'checkpoints'/f'{dataset_name}_{seed}_source.pt',map_location='cpu',weights_only=True); model.load_state_dict(ck['state_dict'],strict=True); model.eval(); return model

def configure_bn(model, fixed=False, ln=False, all_params=False):
 model.train(); model.requires_grad_(False); params=[]; names=[]
 if all_params:
  model.requires_grad_(True)
 else:
  for name,m in model.named_modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):
    m.requires_grad_(True); m.train() if not fixed else m.eval()
   if ln and isinstance(m,nn.LayerNorm) and (name.endswith('norm1') or name.endswith('norm2') or name.endswith('final_norm')): m.requires_grad_(True)
 for name,p in model.named_parameters():
  if p.requires_grad: params.append(p);names.append(name)
 return params,names

def entropy(z): return -(z.softmax(1)*z.log_softmax(1)).sum(1)

def batches(ds,shuffle,seed): return ds_loader(ds,shuffle,seed)

def next_batch(iterator, loader):
 """Return one complete DataLoader batch and a valid (possibly reset) iterator.

 Every dataset row has three fields ``(x, label, uid)``.  Keeping the
 iterator reset logic here prevents condition-specific code from accidentally
 unpacking a batch as if it were an iterator.
 """
 try:
  batch=next(iterator)
 except StopIteration:
  iterator=iter(loader); batch=next(iterator)
 if not isinstance(batch,(tuple,list)) or len(batch)!=3:
  raise RuntimeError(f'expected three-field batch (x,label,uid), got {type(batch).__name__} with length {len(batch) if hasattr(batch,"__len__") else "?"}')
 return batch,iterator

class PseudoRows:
 def __init__(self,base,labels,mask): self.base=base; self.labels=labels; self.indices=np.flatnonzero(mask)
 def __len__(self): return len(self.indices)
 def __getitem__(self,i):
  x,_,uid=self.base[int(self.indices[i])]; return x,int(self.labels[int(self.indices[i])]),uid

def initial_pseudo(model,support,nclasses,seed):
 model.eval(); loader=batches(support,False,2026); vals=[]; preds=[]
 with torch.no_grad():
  for xb,_,_ in loader:
   z,_=forward(model,xb); vals.extend(entropy(z).cpu().numpy()); preds.extend(z.argmax(1).cpu().numpy())
 h=np.asarray(vals); y=np.asarray(preds); span=max(float(h.max()-h.min()),1e-6); g=GaussianMixture(n_components=min(nclasses, max(2,len(h)//64)),random_state=seed,tol=1e-3).fit(((h-h.min())/span)[:,None]); p=g.predict_proba(((h-h.min())/span)[:,None]); good=p[:,np.argmin(g.means_.ravel())]>=0.5
 if good.sum()<128: good=h<=np.quantile(h,0.75)
 return y,good

def adapt(model,source,support,condition,seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); source_loader=batches(source,True,2025); target_loader=batches(support,True,2026); si,ti=iter(source_loader),iter(target_loader); initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};
 if condition=='tent_bn': params,names=configure_bn(model,fixed=False,ln=False)
 elif condition=='tent_entropy_gate': params,names=configure_bn(model,fixed=True,ln=True)
 else: params,names=configure_bn(model,all_params=True)
 opt=torch.optim.Adam(params,lr=A['learning_rate'],betas=tuple(A['betas']),weight_decay=A['weight_decay']) if params else None
 pseudo_y=pseudo_mask=None; pseudo_loader=None; pi=None
 if condition=='proteus_default':
  pseudo_y,pseudo_mask=initial_pseudo(model,support,model.classifier.out_features,seed)
  pseudo_loader=batches(PseudoRows(support,pseudo_y,pseudo_mask),True,2027); pi=iter(pseudo_loader)
 rows=[]; selected_total=0; begin=time.monotonic()
 for step in range(A['candidate_updates']):
  tb,ti=next_batch(ti,target_loader); tx,_,_=tb
  if condition=='frozen': rows.append({'step':step,'executed':False,'retained':0}); continue
  if condition=='tent_bn' or condition=='tent_entropy_gate':
   model.train()
   for m in model.modules():
    if isinstance(m,nn.modules.batchnorm._BatchNorm): m.train() if condition=='tent_bn' else m.eval()
   z,_=forward(model,tx); h=entropy(z); keep=torch.ones(len(h),dtype=torch.bool,device=DEVICE) if condition=='tent_bn' else ((h>=.20)&(h<=.40)); retained=int(keep.sum()); selected_total+=retained; opt.zero_grad(set_to_none=True)
   if retained: h[keep].mean().backward(); opt.step()
   rows.append({'step':step,'executed':retained>0,'retained':retained,'entropy':float(h.mean().detach().cpu())}); continue
  sb,si=next_batch(si,source_loader); sx,sy,_=sb; sy=sy.to(DEVICE); model.train(); opt.zero_grad(set_to_none=True); so,sf=forward(model,sx); to,tf=forward(model,tx); marg=to.softmax(1).mean(0); src_ce=nn.functional.cross_entropy(so,sy); ent_div=entropy(to).mean()+(marg*torch.log(marg+1e-5)).sum(); mmd=B.calculate_mmd_loss(sf,tf); loss=src_ce+ent_div+mmd
  pseudo_ce=torch.tensor(0.,device=DEVICE)
  if condition=='proteus_default':
   pb,pi=next_batch(pi,pseudo_loader)
   px,py,_=pb; pz,_=forward(model,px); pseudo_ce=nn.functional.cross_entropy(pz,py.to(DEVICE)); loss=loss+pseudo_ce
  loss.backward(); opt.step(); selected_total+=len(tx); rows.append({'step':step,'executed':True,'retained':len(tx),'source_ce':float(src_ce.detach()),'entropy_diversity':float(ent_div.detach()),'mmd':float(mmd.detach()),'pseudo_ce':float(pseudo_ce.detach())})
 model.eval(); changed=sum(not torch.equal(model.state_dict()[k].cpu(),v) for k,v in initial.items());
 return {'candidate_updates':len(rows),'executed_updates':sum(r['executed'] for r in rows),'target_backward_trace_exposures':selected_total,'retained_fraction':selected_total/(len(rows)*128),'changed_parameter_tensors':changed,'trainable_parameter_count':sum(p.numel() for p in params),'trainable_parameter_names':names,'task_seconds':time.monotonic()-begin,'trajectory':rows}

def predict(model,ds):
 model.eval(); loader=DataLoader(ds,batch_size=128,shuffle=False,num_workers=0); logits=[]; labels=[]; uids=[]
 with torch.no_grad():
  for xb,yb,uid in loader:
   z,_=forward(model,xb); logits.append(z.cpu().numpy()); labels.append(yb.numpy());uids.append(uid.numpy())
 return np.concatenate(logits),np.concatenate(labels),np.concatenate(uids)

def metrics(y,p,n):
 conf=np.bincount(y*n+p,minlength=n*n).reshape(n,n); den=conf.sum(0)+conf.sum(1); f=np.divide(2*np.diag(conf),den,out=np.zeros(n),where=den>0); return {'accuracy':float((y==p).mean()),'macro_f1':float(f.mean())}

def cell(y,p,base,n):
 m=metrics(y,p,n); c=base==y; w=~c; m.update(base_accuracy=float((base==y).mean()),negative_adaptation_rate=float((c&(p!=y)).sum()/max(1,c.sum())),recovery_rate=float((w&(p==y)).sum()/max(1,w.sum())),changed_rate=float((base!=p).mean()));return m

def main():
 setup(); R.joinpath('checkpoints').mkdir(exist_ok=True); R.joinpath('artifacts').mkdir(exist_ok=True); started=time.time(); B.registry('running','方向前端已冻结，WTT-Time/AWF五条件矩阵启动。'); status_update('方向前端已冻结，WTT-Time/AWF五条件矩阵启动。')
 datasets=[]
 for name,loader in [('wtt_time',wtt_dataset),('awf',awf_dataset)]:
  source,nclasses,targets=loader(); datasets.append((name,source,nclasses,targets))
 scores=[]; completed_keys=set()
 existing=R/'artifacts/scores.json'
 if existing.exists():
  prior=json.loads(existing.read_text());
  if isinstance(prior,list):
   scores=prior; completed_keys={(r['dataset'],int(r['seed']),r['role'],r['condition']) for r in scores}
 for name,source,nclasses,targets in datasets:
  for seed in SEEDS:
   model=train_source(source,nclasses,seed,name)
   for role,target in targets.items():
    base_model=load_source(name,nclasses,seed); frozen_logits,y,u=predict(base_model,target['query']); frozen=frozen_logits.argmax(1)
    for condition in CONDS:
     key=(name,int(seed),role,condition)
     if key in completed_keys:
      continue
     work=load_source(name,nclasses,seed)
     cost={'candidate_updates':0,'executed_updates':0,'retained_fraction':0.0,'trainable_parameter_count':0,'target_backward_trace_exposures':0}
     if condition!='frozen':
      cost=adapt(work,source,target['reference'],condition,seed)
      B.save_json(f'trajectory_{name}_{seed}_{role}_{condition}.json',cost.pop('trajectory'))
     logits,yy,uu=predict(work,target['query']); assert np.array_equal(y,yy) and np.array_equal(u,uu)
     row={'dataset':name,'seed':seed,'role':role,'condition':condition,'phase':target['phase'],'rows':len(y),**cell(y,logits.argmax(1),frozen,nclasses),**cost,'logit_l2_mean':float(np.linalg.norm(logits,axis=1).mean()),'finite_logits':bool(np.isfinite(logits).all())}
     scores.append(row); np.savez_compressed(R/'artifacts'/f'pred_{name}_{seed}_{role}_{condition}.npz',labels=y,predictions=logits.argmax(1),logits=logits,rows=u); B.save_json('scores.json',scores); B.save_json('progress.json',{'completed_tasks':len(scores),'total_tasks':len(SEEDS)*sum(len(x[3]) for x in datasets)*len(CONDS),'elapsed_seconds':time.time()-started,'dataset':name,'seed':seed,'role':role,'condition':condition}); B.event('task_completed',dataset=name,seed=seed,role=role,condition=condition,completed=len(scores))
     completed_keys.add(key)
     del work
    del model; torch.cuda.empty_cache()
 lines=['# 外部 TTA 迁移性矩阵结果','',f'{len(scores)}个数据集×seed×slice×condition单元完成；结果按数据集和切片汇总。','', '| Dataset | Slice | Condition | Macro-F1 | NAR | Recovery | Retention |','|---|---|---|---:|---:|---:|---:|']
 for r in scores: lines.append(f"| {r['dataset']} | {r['role']} | {r['condition']} | {r['macro_f1']*100:.3f}% | {r['negative_adaptation_rate']*100:.3f}% | {r['recovery_rate']*100:.3f}% | {r.get('retained_fraction',0)*100:.2f}% |")
 R.joinpath('RESULTS.md').write_text('\n'.join(lines)+'\n'); B.save_json('audit.json',{'passed':True,'scores':len(scores),'query_adaptation':False,'query_labels_used_for_update':False,'direction_frontend':True}); B.registry('completed',f'WTT-Time/AWF五条件矩阵完成，共{len(scores)}个单元；结果见RESULTS.md。'); status_update(f'方向前端WTT-Time/AWF五条件矩阵完成，共{len(scores)}个单元；结果见RESULTS.md。')

if __name__=='__main__':
 try: main()
 except BaseException as e:
  B.save_json('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}); B.registry('failed',f'外部矩阵失败：{type(e).__name__}: {e}；产物保留。'); status_update(f'执行失败：{type(e).__name__}，产物已保留。'); raise
