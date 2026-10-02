import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,gc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import RUN,ROOT,sha,save,deterministic
from model import MultiViewModel,span_mask,mix_batch,mixed_ce
from worker import predict

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0';deterministic('cuda',2)
 c=json.loads((RUN/'config.json').read_text());old=ROOT/c['historical_run'];references={}
 for p,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
 raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');assert set(raw)=={'source','valid'}
 data={role:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()};checks=0
 for seed in c['seeds']:
  d=RUN/'artifacts'/f'baseline_{seed}';rep=json.loads((d/'report.json').read_text());torch.manual_seed(seed);m=MultiViewModel('baseline').cuda()
  state=torch.load(d/'initial_state.pt',weights_only=True);assert set(state)==set(m.state_dict()) and all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
  ckpath=ROOT/'runs'/rep['origin_run']/'checkpoints'/f"{rep['origin_task']}_best.pt";references[str(ckpath.relative_to(ROOT))]=sha(ckpath)
  m.load_state_dict(torch.load(ckpath,weights_only=True,map_location='cuda')['state_dict'])
  with np.load(d/'predictions_best.npz') as f:
   for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),f[role]);checks+=1
  del m
 # Synthetic Mixup independently replayed input/label mapping and CE identity.
 rng=np.random.default_rng(41);rr=np.random.default_rng(41);x=torch.arange(48.).reshape(4,2,6);y=torch.arange(4)
 mixed,other,w=mix_batch(x,y,rng);ww=float(rr.beta(.2,.2));perm=torch.tensor(rr.permutation(4));assert w==ww and torch.equal(other,y[perm]) and torch.equal(mixed,w*x+(1-w)*x[perm])
 z=torch.randn(4,5);manual=-(w*z.log_softmax(1)[torch.arange(4),y]+(1-w)*z.log_softmax(1)[torch.arange(4),other]).mean();assert torch.allclose(mixed_ce(z,y,other,w),manual)
 states={};params={};peaks={};e=data['source'];seed=c['seeds'][0];bn_checks={}
 for kind in c['reported_conditions']:
  gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.manual_seed(seed);m=MultiViewModel(kind).cuda()
  states[kind]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};params[kind]={'total':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad),'generator_trainable':sum(p.numel() for p in m.generator.parameters() if p.requires_grad)}
  m.eval()
  with torch.no_grad():
   g=m.generator(e['timestamps'][:2],e['tam'][:2]);assert g['time'].shape==(2,120,110)
   fixed=e['tam'][:2].log1p().reshape(2,2,120,15).permute(0,2,1,3).reshape(2,120,30);assert torch.equal(g['time'][...,:30],fixed)
   assert torch.equal(m(e['timestamps'][:2],e['tam'][:2]),m(torch.zeros_like(e['timestamps'][:2]),e['tam'][:2]))
   assert torch.isfinite(m(torch.zeros_like(e['timestamps'][:2]),torch.zeros_like(e['tam'][:2]))).all()
  before={k:v.clone() for k,v in m.named_buffers()};m.train();tam=span_mask(e['tam'][:64],torch.Generator().manual_seed(seed+50000));y=e['labels'][:64];other=y;w=1.
  if c['factors'][kind]['mixup']:tam,other,w=mix_batch(tam,y,np.random.default_rng(seed+80000))
  z=m(e['timestamps'][:64],tam);loss=mixed_ce(z,y,other,w);assert torch.isfinite(loss);loss.backward()
  for name,p in m.named_parameters():
   if not p.requires_grad:assert p.grad is None,name
   elif name.startswith('generator.'):
    assert p.grad is not None and torch.isfinite(p.grad).all(),name
    if name.endswith('weight'):assert p.grad.norm()>0,name
  if c['factors'][kind]['bn']:
   bns=[mod for mod in m.modules() if isinstance(mod,nn.BatchNorm1d)];assert len(bns)==6
   assert all(int(mod.num_batches_tracked)==1 for mod in bns)
   assert any(not torch.equal(v,dict(m.named_buffers())[k]) for k,v in before.items() if k.endswith('running_mean'))
   after={k:v.clone() for k,v in m.named_buffers()}
   predict(m,{k:v[:65] for k,v in e.items()},64)
   assert all(torch.equal(v,dict(m.named_buffers())[k]) for k,v in after.items())
   with torch.no_grad():
    a=m(e['timestamps'][:1],e['tam'][:1]);b=m(e['timestamps'][:2],e['tam'][:2])[:1];assert torch.allclose(a,b,atol=1e-4,rtol=1e-4)
   bn_checks[kind]={'source_train_updates':1,'source_eval_no_updates':True,'eval_batch_independent':True}
  moments=[torch.zeros_like(p) for p in m.parameters() if p.requires_grad for _ in range(2)]
  torch.cuda.synchronize();peaks[kind]=torch.cuda.max_memory_allocated();del moments,m,z,loss,tam
 ref=states['baseline']
 for kind,state in states.items():
  for k,v in ref.items():
   if not k.startswith('generator.') or c['factors'][kind]['width']==1:assert torch.equal(v,state[k]),(kind,k)
 assert all(torch.equal(v,states['wide_bn'][k]) for k,v in states['wide'].items())
 for kind in ['bn','wide_bn']:
  for k,v in states[kind].items():
   if '.norm' in k:
    expected=torch.ones_like(v) if k.endswith(('weight','running_var')) else torch.zeros_like(v);assert torch.equal(v,expected),k
 estimate=3*max(peaks.values())+7*1024**3;total=torch.cuda.mem_get_info()[1];assert estimate<total,(estimate,total)
 save(RUN/'artifacts/preflight.json',{'passed':True,'baseline_replays':checks,'shared_initialization_checked':True,'mixup_semantics_checked':True,'bn_checks':bn_checks,'source_backward_examples_per_condition':64,'optimizer_updates':0,'parameter_counts':params,'peak_bytes_per_worker':peaks,'aggregate_gpu_bytes_with_reserve':estimate,'gpu_total_bytes':total,'historical_reference_hashes':references,'future_access':False})
 print(json.dumps({'passed':True,'baseline_replays':checks,'parameter_counts':params,'gpu_budget_gib':estimate/1024**3}))
if __name__=='__main__':main()
