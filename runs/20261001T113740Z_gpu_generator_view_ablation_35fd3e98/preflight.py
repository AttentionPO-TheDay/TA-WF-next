"""Real-source CUDA audit, no optimizer or valid scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import gc,json,time,torch
from model import MultiViewModel,VIEWS,parameter_counts
from base_model import MultiViewModel as BaseModel
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
OLD=ROOT/'runs/20261001T104959Z_gpu_multiview_generator_head_factorial_cf47d62c'
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
raw=torch.load(R/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
data_bytes=sum(raw[r][k].numel()*raw[r][k].element_size() for r in ('source','valid') for k in ('timestamps','tam','labels'))
x,t,y=[raw['source'][k][:64].cuda() for k in ('timestamps','tam','labels')];del raw
initial=torch.load(OLD/'artifacts/learned_transformer_21729/initial_state.pt',weights_only=True)
records={}
for view in VIEWS:
 torch.manual_seed(21729);m=MultiViewModel(view).cuda()
 assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items())
 torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 for _ in range(8):
  m.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x,t),y);assert torch.isfinite(loss);loss.backward()
 torch.cuda.synchronize();seconds=(time.monotonic()-start)/8
 grad={k:None if p.grad is None else float(p.grad.norm()) for k,p in m.generator.named_parameters()}
 for k,p in m.generator.named_parameters():
  if p.requires_grad:assert grad[k] is not None and grad[k]>0 and torch.isfinite(p.grad).all(),k
  else:assert grad[k] is None,k
 m.eval()
 with torch.inference_mode():
  z=m(x,t);assert torch.isfinite(z).all()
  if view.startswith('packet_'):assert torch.equal(z,m(x,torch.zeros_like(t)))
  if view=='packet_direction':assert torch.equal(z,m(x.sign()*23,torch.ones_like(t)))
  if view=='tam_only':assert torch.equal(z,m(torch.zeros_like(x),t))
  if view=='fusion':
   torch.manual_seed(21729);base=BaseModel('learned','transformer').cuda().eval()
   torch.testing.assert_close(z,base(x,t),atol=0,rtol=0)
   del base
 records[view]={**parameter_counts(m),'step_seconds':seconds,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'generator_gradient_norms':grad}
 del m,loss;gc.collect();torch.cuda.empty_cache()
free,total=torch.cuda.mem_get_info()
estimate=2*(max(d['peak_cuda_bytes'] for d in records.values())+data_bytes+2*1024**3)
assert estimate < .85*free,(estimate,free)
out={'checks':'passed','real_source_rows':64,'optimizer_steps':0,'valid_scored':False,'future_access':False,
     'historical_initialization_equal':True,'fusion_exact_forward_parity':True,'removed_input_invariance':True,
     'records':records,'two_worker_memory_estimate_bytes':estimate,'free_cuda_bytes':free,'total_cuda_bytes':total}
(R/'artifacts/preflight_gpu.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
