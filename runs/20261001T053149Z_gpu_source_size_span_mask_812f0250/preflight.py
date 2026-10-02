from pathlib import Path
import sys,json
import torch
from augmentation import span_mask
from model import PacketModel
r=Path(__file__).resolve().parent;torch.set_num_threads(2)
x=torch.ones(8,5000);length=torch.tensor([0,1,2,20,50,100,1000,5000]);mask=torch.arange(5000)[None]<length[:,None];x*=mask
before=x.clone()
for seed in range(20):
 a=span_mask(x,mask,torch.Generator().manual_seed(seed));b=span_mask(x,mask,torch.Generator().manual_seed(seed));assert torch.equal(a,b) and torch.equal(x,before)
 erased=((a==0)&mask).sum(1);assert (erased<=32).all() and (erased[length<=1]==0).all();assert (a[~mask]==0).all();assert ((a!=0).sum(1)[length>0]>=1).all()
assert any(not torch.equal(span_mask(x,mask,torch.Generator().manual_seed(s)),x) for s in range(20))
result={'augmentation_checks':'passed','training_steps':0,'valid_scored':False}
if '--gpu' in sys.argv:
 import os
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 m=PacketModel(head='mlp').cuda();v=torch.ones(64,5000,device='cuda');observed=v.bool();v=span_mask(v,observed,torch.Generator(device='cuda').manual_seed(171));torch.nn.functional.cross_entropy(m(v,observed),torch.arange(64,device='cuda')%102).backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 m.eval()
 with torch.inference_mode():assert torch.isfinite(m(torch.ones(128,5000,device='cuda'))).all()
 result['gpu_checks']='passed';result['peak_bytes']=torch.cuda.max_memory_allocated()
(r/'artifacts'/('preflight_gpu.json' if '--gpu' in sys.argv else 'preflight_cpu.json')).write_text(json.dumps(result,indent=2)+'\n');print(result)
