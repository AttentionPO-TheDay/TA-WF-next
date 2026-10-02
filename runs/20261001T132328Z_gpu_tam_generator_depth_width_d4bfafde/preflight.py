"""Source-only GPU checks for unchanged classifier, depth support and gradients."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import gc,json,time,torch
from model import MultiViewModel,CONDITIONS,parameter_counts
from encoder_base import MultiViewModel as PreviousModel
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
OLD=ROOT/'runs/20261001T124113Z_gpu_tam_multiscale_generator_101ab9e9'
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
raw=torch.load(R/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
cache_bytes=sum(raw[role][k].numel()*raw[role][k].element_size() for role in ('source','valid') for k in ('timestamps','tam','labels'))
x,t,y=[raw['source'][k][:64].cuda() for k in ('timestamps','tam','labels')];del raw
initial=torch.load(OLD/'artifacts/multiscale_d139_21729/initial_state.pt',weights_only=True)
records={};fixed=None;shallow_logits={}
for kind in CONDITIONS:
 torch.manual_seed(21729);m=MultiViewModel(kind).cuda()
 assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items() if not k.startswith('generator.'))
 if kind=='d2_w16':assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items())
 torch.cuda.reset_peak_memory_stats()
 for _ in range(3):
  m.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(m(x,t),y).backward()
 torch.cuda.synchronize();start=time.monotonic()
 for _ in range(10):
  m.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x,t),y);assert torch.isfinite(loss);loss.backward()
 torch.cuda.synchronize();seconds=(time.monotonic()-start)/10
 grads={k:None if p.grad is None else float(p.grad.norm()) for k,p in m.generator.named_parameters()}
 for k,p in m.generator.named_parameters():
  if p.requires_grad:assert grads[k] is not None and grads[k]>0 and torch.isfinite(p.grad).all(),k
  else:assert grads[k] is None,k
 m.eval()
 with torch.inference_mode():
  z=m(x,t);assert torch.isfinite(z).all() and torch.equal(z,m(torch.zeros_like(x),t))
  tok=m.generator(x,t);assert not tok['packet_mask'].any() and tok['time_mask'].all()
  stat=tok['time'][...,:30].clone()
  if fixed is None:fixed=stat
  else:assert torch.equal(stat,fixed)
  depth,width=CONDITIONS[kind]
  if depth==2:shallow_logits[width]=z.clone()
  else:torch.testing.assert_close(z,shallow_logits[width],atol=0,rtol=0)
  if kind=='d2_w16':
   torch.manual_seed(21729);previous=PreviousModel('multiscale_d139').cuda().eval()
   torch.testing.assert_close(z,previous(x,t),atol=0,rtol=0);del previous
 records[kind]={**parameter_counts(m),'step_seconds':seconds,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'generator_gradient_norms':grads}
 del m,loss,tok,stat;gc.collect();torch.cuda.empty_cache()
free,total=torch.cuda.mem_get_info();estimate=2*(max(v['peak_cuda_bytes'] for v in records.values())+cache_bytes+2*1024**3)
assert estimate<.85*free,(estimate,free)
out={'checks':'passed','real_source_rows':64,'optimizer_steps':0,'new_valid_scoring':False,'future_access':False,
     'shared_classifier_initialization':'passed','historical_baseline_exact_parity':True,'zero_residual_depth_initial_parity':True,
     'fixed_stats_and_masks_equal':True,'packet_invariance':True,'records':records,
     'two_worker_memory_estimate_bytes':estimate,'free_cuda_bytes':free,'total_cuda_bytes':total}
(R/'artifacts/preflight_gpu.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
