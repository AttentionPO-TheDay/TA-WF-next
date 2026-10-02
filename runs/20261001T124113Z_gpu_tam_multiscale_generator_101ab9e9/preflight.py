"""Source-only CUDA checks; no optimizer or new valid scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import gc,json,time,torch
from model import MultiViewModel,ENCODERS,parameter_counts
from view_base import MultiViewModel as ViewModel
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
OLD=ROOT/'runs/20261001T113740Z_gpu_generator_view_ablation_35fd3e98'
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
raw=torch.load(R/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
cache_bytes=sum(raw[r][k].numel()*raw[r][k].element_size() for r in ('source','valid') for k in ('timestamps','tam','labels'))
x,t,y=[raw['source'][k][:64].cuda() for k in ('timestamps','tam','labels')];del raw
initial=torch.load(OLD/'artifacts/tam_only_21729/initial_state.pt',weights_only=True)
records={};fixed=None
for encoder in ENCODERS:
 torch.manual_seed(21729);m=MultiViewModel(encoder).cuda()
 assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items())
 torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 for _ in range(10):
  m.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x,t),y);assert torch.isfinite(loss);loss.backward()
 torch.cuda.synchronize();secs=(time.monotonic()-start)/10
 grads={k:None if p.grad is None else float(p.grad.norm()) for k,p in m.generator.named_parameters()}
 for k,p in m.generator.named_parameters():
  if p.requires_grad:assert grads[k] is not None and grads[k]>0 and torch.isfinite(p.grad).all(),k
  else:assert grads[k] is None,k
 m.eval()
 with torch.inference_mode():
  z=m(x,t);assert torch.isfinite(z).all() and torch.equal(z,m(torch.zeros_like(x),t))
  tok=m.generator(x,t);assert not tok['packet_mask'].any() and tok['time_mask'].all()
  summary=tok['time'][...,:30].clone()
  if fixed is None:fixed=summary
  else:assert torch.equal(summary,fixed)
  if encoder=='local_d1':
   torch.manual_seed(21729);previous=ViewModel('tam_only').cuda().eval()
   torch.testing.assert_close(z,previous(x,t),atol=0,rtol=0);del previous
 records[encoder]={**parameter_counts(m),'step_seconds':secs,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'generator_gradient_norms':grads}
 del m,loss,tok,summary;gc.collect();torch.cuda.empty_cache()
free,total=torch.cuda.mem_get_info();estimated=2*(max(v['peak_cuda_bytes'] for v in records.values())+cache_bytes+2*1024**3)
assert estimated<.85*free,(estimated,free)
out={'checks':'passed','source_rows':64,'optimizer_steps':0,'new_valid_scoring':False,'future_access':False,
     'historical_initialization_equal':True,'local_forward_exact_parity':True,'fixed_stats_identical':True,
     'parameter_counts_equal':True,'packet_invariance':True,'records':records,
     'two_worker_memory_estimate_bytes':estimated,'free_cuda_bytes':free,'total_cuda_bytes':total}
(R/'artifacts/preflight_gpu.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
