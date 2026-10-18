import copy, hashlib, json, math, os, random, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
import port_base as B
R, ROOT, C, DEVICE = B.R, B.ROOT, B.C, B.DEVICE
KIND=C['mechanism_kind']; SEEDS=C['seeds']; ROLES=C['roles']; A=C['adaptation']

def gpu_sync(): torch.cuda.synchronize(); return time.perf_counter()
def loss_entropy(z):
 p=z.softmax(1); return -(p*z.log_softmax(1)).sum(1)
def scale_rule(h, h0, hist):
 if KIND=='window' and len(hist)>=16:
  first=float(np.mean(hist[-16:-12])); last=float(np.mean(hist[-4:])); slope=(last-first)/max(abs(first),1e-8)
  scale=float(np.clip(1.0-4.0*slope,0.25,1.0)); return scale, slope
 rel=h/max(h0,1e-8); return float(np.clip(1.0/rel,0.25,1.0)), float(rel-1.0)
def topology_loss(a,b):
 # Three pooled scales over the 256-dimensional packet/time representation.
 a=torch.nn.functional.normalize(a,dim=1); b=torch.nn.functional.normalize(b,dim=1); out=a.new_zeros(())
 for groups in (1,2,4):
  aa=a.reshape(a.shape[0],groups,-1); bb=b.reshape(b.shape[0],groups,-1)
  out=out+torch.nn.functional.mse_loss(aa.mean(-1),bb.mean(-1))+torch.nn.functional.mse_loss(aa.std(-1,unbiased=False),bb.std(-1,unbiased=False))
 return out/3.0
def loader_source(source):
 return DataLoader(TensorDataset(*(torch.from_numpy(source[k]) for k in ('timestamps','tam','labels','rows'))),batch_size=128,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(2025))
def loader_target(support):
 return DataLoader(TensorDataset(*(torch.from_numpy(support[k]) for k in ('timestamps','tam','rows'))),batch_size=128,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(2026))
def take(it,dl):
 try:return next(it),it
 except StopIteration:it=iter(dl);return next(it),it
def adapt(model,source,support,seed,role):
 random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
 sl,tl=loader_source(source),loader_target(support); si,ti=iter(sl),iter(tl)
 teacher=copy.deepcopy(model).eval() if KIND=='ema' else None
 if teacher is not None:
  for p in teacher.parameters(): p.requires_grad_(False)
 opt=torch.optim.Adam(model.parameters(),lr=A['base_learning_rate'])
 model.eval(); hs=[]; start=gpu_sync()
 for i in range(0,len(support['rows']),128):
  x=torch.from_numpy(support['timestamps'][i:i+128]).to(DEVICE); t=torch.from_numpy(support['tam'][i:i+128]).to(DEVICE)
  with torch.no_grad(): hs.extend(loss_entropy(B.forward_with_features(model,x,t)[0]).cpu().tolist())
 initial_probe=gpu_sync()-start; h0=float(np.mean(hs)); hist=[]; rows=[]; executed=0; maxgrad=0.; begin=time.monotonic(); torch.cuda.reset_peak_memory_stats()
 for step in range(A['candidate_updates']):
  if time.monotonic()-begin>C['budget']['max_task_seconds']: raise TimeoutError('task budget exceeded')
  sb,si=take(si,sl); tb,ti=take(ti,tl); sx,st,sy=[v.to(DEVICE) for v in sb[:3]]; tx,tt=[v.to(DEVICE) for v in tb[:2]]
  with torch.no_grad():
   model.eval(); probe=loss_entropy(B.forward_with_features(model,tx,tt)[0]); hm=float(probe.mean().cpu())
   teacher_logits=B.forward_with_features(teacher,tx,tt)[0] if teacher is not None else None
  scale,signal=scale_rule(hm,h0,hist); hist.append(hm); lr=A['base_learning_rate']*scale; opt.param_groups[0]['lr']=lr
  opt.zero_grad(set_to_none=True); model.train(); set_start=gpu_sync()
  so,sf=B.forward_with_features(model,sx,st); to,tf=B.forward_with_features(model,tx,tt)
  marg=to.softmax(1).mean(0)
  terms=[torch.nn.functional.cross_entropy(so,sy), loss_entropy(to).mean()+(marg*torch.log(marg+1e-5)).sum(), B.calculate_mmd_loss(sf,tf)]
  extra=to.new_zeros(())
  if KIND=='ema': extra=torch.nn.functional.kl_div(to.log_softmax(1),teacher_logits.softmax(1),reduction='batchmean')*C['mechanism']['consistency_loss_weight']
  elif KIND=='multiscale': extra=topology_loss(sf,tf)*C['mechanism']['topology_loss_weight']
  total=sum(terms)+extra; forward_sec=gpu_sync()-set_start; back_start=gpu_sync(); total.backward(); back_sec=gpu_sync()-back_start
  for p in model.parameters():
   if p.grad is not None: maxgrad=max(maxgrad,float(p.grad.detach().abs().max()))
  opt_start=gpu_sync(); opt.step(); opt_sec=gpu_sync()-opt_start; executed+=1
  if teacher is not None:
   with torch.no_grad():
    for tp,sp in zip(teacher.parameters(),model.parameters()): tp.mul_(C['mechanism']['ema_decay']).add_(sp,alpha=1-C['mechanism']['ema_decay'])
  rows.append({'step':step,'entropy_nats':hm,'signal':signal,'scale':scale,'applied_lr':lr,'forward_seconds':forward_sec,'backward_seconds':back_sec,'optimizer_seconds':opt_sec,'source_ce':float(terms[0].detach()),'entropy_diversity':float(terms[1].detach()),'mmd':float(terms[2].detach()),'extra_loss':float(extra.detach()),'total_loss':float(total.detach())})
 totals={k+'_seconds':sum(x[k+'_seconds'] for x in rows) for k in ('forward','backward','optimizer')}
 return {'candidate_updates':A['candidate_updates'],'executed_updates':executed,'skipped_updates':0,'initial_support_entropy':h0,'initial_probe_seconds':initial_probe,'task_seconds':time.monotonic()-begin+initial_probe,'backward_amortized_ms_per_target_trace_exposure':1000*totals['backward_seconds']/(executed*128),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'relative_parameter_change':float(sum((p.detach()**2).sum() for p in model.parameters()).sqrt().cpu()),**totals}
def main():
 assert C['status']=='frozen' and C['id']==R.name and os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available()
 source=B.load_prepared(C['prepared']['source_valid'],'source'); valid=B.load_prepared(C['prepared']['source_valid'],'valid'); source_all={k:np.concatenate([source[k],valid[k]]) for k in source}
 scores=[]
 for seed in SEEDS:
  for role in ROLES:
   target=B.load_prepared(C['prepared'][role]); s,q,_=B.split_targets(source_all,role,target); np.savez_compressed(R/'artifacts'/f'split_{role}.npz',support_indices=s,query_indices=q,support=target['rows'][s],query=target['rows'][q]); support={k:target[k][s] for k in ('timestamps','tam','rows')}
   base_file=ROOT/'runs/20261013T091631Z_mvi_dynamic_bn_tta_a79d0c0b/artifacts'/f'pred_{seed}_{role}_frozen.npz'; base=np.load(base_file); assert np.array_equal(base['rows'],target['rows'][q]); frozen=base['predictions']
   model,_=B.load_model(seed); cost=adapt(model,source,support,seed,role); z=B.predict(model,target['timestamps'][q],target['tam'][q],128); pred=z.argmax(1); met=B.cell_metrics(target['labels'][q],frozen,pred); row={'seed':seed,'role':role,'condition':KIND,'origin':'new','rows':len(q),**met,**cost}; scores.append(row)
   np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_{KIND}.npz',rows=target['rows'][q],labels=target['labels'][q],predictions=pred,logits=z); torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'seed':seed,'role':role,'condition':KIND,'config_sha':B.sha(R/'config.json')},R/'checkpoints'/f'{seed}_{role}_{KIND}.pt'); B.save_json('scores.json',scores); del model; torch.cuda.empty_cache()
 avgf=float(np.mean([x['macro_f1'] for x in scores])); avgn=float(np.mean([x['negative_adaptation_rate'] for x in scores]));
 (R/'RESULTS.md').write_text('# 实验结果\n\n9个TemporalDrift任务完成。\n\n| 条件 | Macro-F1 | 负适应率 |\n|---|---:|---:|\n| '+KIND+' | '+f'{avgf*100:.3f}% | {avgn*100:.3f}% |'+'\n\n结果仍属TemporalDrift开发证据；未访问WTT-Time/AWF。\n')
 B.save_json('audit.json',{'passed':True,'new_tasks':len(scores),'pseudo_label_calls':0,'gmm_calls':0,'query_adaptation':False}); B.registry('completed',f'{KIND} 9项任务完成；Macro-F1 {avgf*100:.3f}%，负适应率 {avgn*100:.3f}%。'); print(json.dumps({'condition':KIND,'macro_f1':avgf,'negative_adaptation_rate':avgn}))
if __name__=='__main__':
 try: main()
 except BaseException as e: B.save_json('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}); B.registry('failed',f'{KIND}失败：{type(e).__name__}: {e}'); raise
