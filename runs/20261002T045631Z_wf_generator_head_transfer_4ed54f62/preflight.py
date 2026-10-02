import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,gc
import numpy as np
import torch
from torch.nn import functional as F
from common import RUN,ROOT,sha,save,deterministic
from model import MultiViewModel,span_mask
from worker import predict
from rf_native import RFNative

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0';deterministic('cuda',2)
 c=json.loads((RUN/'config.json').read_text());old=ROOT/c['historical_run'];references={}
 for p,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
 raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');assert set(raw)=={'source','valid'}
 data={role:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()}
 checks=0
 for seed in c['seeds']:
  d=RUN/'artifacts'/f'shallow_transformer_{seed}';rep=json.loads((d/'report.json').read_text());torch.manual_seed(seed);m=MultiViewModel('shallow_transformer').cuda()
  initial=torch.load(d/'initial_state.pt',weights_only=True);assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items())
  ckpath=ROOT/'runs'/rep['origin_run']/'checkpoints'/f"{rep['origin_task']}_best.pt";references[str(ckpath.relative_to(ROOT))]=sha(ckpath)
  m.load_state_dict(torch.load(ckpath,weights_only=True,map_location='cuda')['state_dict'])
  with np.load(d/'predictions_best.npz') as f:
   for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),f[role]);checks+=1
  del m
 params={};peaks={};states={};e=data['source'];seed=c['seeds'][0]
 for kind in c['reported_conditions']:
  gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.manual_seed(seed);m=MultiViewModel(kind).cuda()
  states[kind]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  params[kind]={'total':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad),'generator_trainable':sum(p.numel() for p in m.generator.parameters() if p.requires_grad) if hasattr(m,'generator') else None}
  m.eval()
  with torch.no_grad():
   z=m(e['timestamps'][:2],e['tam'][:2]);assert torch.isfinite(z).all() and z.shape==(2,102)
   assert torch.equal(z,m(torch.zeros_like(e['timestamps'][:2]),e['tam'][:2]))
   assert torch.isfinite(m(torch.zeros_like(e['timestamps'][:2]),torch.zeros_like(e['tam'][:2]))).all()
   if hasattr(m,'generator'):
    g=m.generator(e['timestamps'][:2],e['tam'][:2]);assert g['time'].shape==(2,120,110)
    fixed=e['tam'][:2].log1p().reshape(2,2,120,15).permute(0,2,1,3).reshape(2,120,30)
    assert torch.equal(g['time'][...,:30],fixed);assert not g['packet_mask'].any() and g['time_mask'].all()
    one=m.generator(e['timestamps'][:1],e['tam'][:1])['time'];assert torch.allclose(one,g['time'][:1],atol=1e-5,rtol=1e-5)
   if kind.endswith('_mlp'):
    x=torch.randn(2,120,128,device='cuda');mask=torch.zeros(2,120,dtype=torch.bool,device='cuda');a=m.head(x,mask);x[:,0]+=1;b=m.head(x,mask);assert torch.equal(a[:,1:],b[:,1:])
   if kind=='rf_matched':
    direct=RFNative(102).cuda().eval();direct.load_state_dict(m.rf.state_dict());assert torch.equal(z,direct(e['tam'][:2].log1p()[:,None]));del direct
  m.train();tam=span_mask(e['tam'][:64],torch.Generator().manual_seed(seed+50000));loss=F.cross_entropy(m(e['timestamps'][:64],tam),e['labels'][:64]);assert torch.isfinite(loss);loss.backward()
  assert any(p.grad is not None and p.grad.norm()>0 for p in m.parameters() if p.requires_grad)
  for name,p in m.named_parameters():
   if p.grad is not None:assert torch.isfinite(p.grad).all(),name
   if not p.requires_grad:assert p.grad is None,name
   if name.startswith('generator.') and p.requires_grad:assert p.grad is not None and p.grad.norm()>0,name
  # Materialize two Adam moment buffers for memory accounting, no optimizer update.
  moments=[torch.zeros_like(p) for p in m.parameters() if p.requires_grad for _ in range(2)]
  torch.cuda.synchronize();peaks[kind]=torch.cuda.max_memory_allocated()
  del m,z,loss,tam,moments
 # Shared parameter equality within same functional modules.
 ref=states['shallow_transformer']
 for kind,state in states.items():
  if kind=='rf_matched':continue
  for k,v in ref.items():
   if not k.startswith(('generator.','head.')):assert torch.equal(v,state[k]),(kind,k)
   if k.startswith('generator.') and kind.startswith('shallow'):assert torch.equal(v,state[k])
   if k.startswith('head.') and kind.endswith('transformer'):assert torch.equal(v,state[k])
 assert all(torch.equal(v,states['progressive_mlp'][k]) for k,v in states['progressive_transformer'].items() if k.startswith('generator.'))
 assert all(torch.equal(v,states['progressive_mlp'][k]) for k,v in states['shallow_mlp'].items() if k.startswith('head.'))
 # Generator interface independent of number of classifier classes (architecture check).
 a=MultiViewModel('progressive_mlp',7);b=MultiViewModel('progressive_mlp',102)
 assert {k:tuple(v.shape) for k,v in a.generator.state_dict().items()}=={k:tuple(v.shape) for k,v in b.generator.state_dict().items()}
 student_peak=max(v for k,v in peaks.items() if k!='rf_matched');estimate=max(3*student_peak,2*student_peak+peaks['rf_matched'])+7*1024**3;total=torch.cuda.mem_get_info()[1];assert estimate<total,(estimate,total)
 save(RUN/'artifacts/preflight.json',{'passed':True,'baseline_replays':checks,'fixed_counts_masks_class_independence_checked':True,'shared_initialization_checked':True,'rf_wrapper_parity':True,'real_source_backward_examples_per_condition':64,'optimizer_updates':0,'parameter_counts':params,'per_worker_peak_bytes_including_data_and_moment_buffers':peaks,'aggregate_gpu_bytes_with_reserve':estimate,'gpu_total_bytes':total,'historical_reference_hashes':references,'future_access':False})
 print(json.dumps({'passed':True,'baseline_replays':checks,'parameters':params,'gpu_budget_gib':estimate/1024**3}))
if __name__=='__main__':main()
