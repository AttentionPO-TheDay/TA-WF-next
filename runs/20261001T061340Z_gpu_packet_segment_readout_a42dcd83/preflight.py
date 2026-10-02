from pathlib import Path
import json,sys,importlib.util,os
import torch
from model import PacketModel,pool_tokens
R=Path(__file__).resolve().parent;ROOT=R.parents[1];torch.set_num_threads(2)
x=torch.arange(8.).reshape(1,8,1);mask=torch.ones(1,8,dtype=torch.bool)
assert torch.equal(pool_tokens(x,mask,'segments'),torch.tensor([[.5,2.5,4.5,6.5]]))
mask[:,5:]=False
assert torch.equal(pool_tokens(x,mask,'segments'),torch.tensor([[.5,2.,3.,4.]]))
assert torch.equal(pool_tokens(x,torch.zeros_like(mask),'segments'),torch.zeros(1,4))
# Shared initialization including the widened head, original backbone exact.
spec=importlib.util.spec_from_file_location('previous',ROOT/'runs/20261001T053149Z_gpu_source_size_span_mask_812f0250/model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.manual_seed(171);old=mod.PacketModel(head='mlp').eval()
torch.manual_seed(171);a=PacketModel(head='mlp',pooling='global_repeat').eval()
torch.manual_seed(171);b=PacketModel(head='mlp',pooling='segments').eval()
for k,v in a.state_dict().items():
 assert torch.equal(v,b.state_dict()[k])
 if not k.startswith('readout.'):assert torch.equal(v,old.state_dict()[k])
d=torch.zeros(3,5000);d[0,:321]=1;d[1,:1400]=-1
with torch.inference_mode():
 torch.testing.assert_close(a(d),old(d),rtol=1e-5,atol=1e-6)
 obs=d!=0;dirty=d.clone();dirty[~obs]=float('nan');torch.testing.assert_close(b(d),b(dirty,obs),rtol=0,atol=0)
 torch.testing.assert_close(b(d)[:1],b(d[:1]),rtol=1e-5,atol=1e-6)
assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
result={'pool_reference':'passed','shared_initialization':'passed','global_initial_function_parity':'passed','padding_and_batch':'passed','parameters':sum(p.numel() for p in a.parameters()),'optimizer_steps':0,'valid_scored':False}
if '--gpu' in sys.argv:
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 for mode in ['global_repeat','segments']:
  model=PacketModel(head='mlp',pooling=mode).cuda();d=torch.ones(64,5000,device='cuda');d[-1]=0
  torch.nn.functional.cross_entropy(model(d),torch.arange(64,device='cuda')).backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  model.eval()
  with torch.inference_mode():assert torch.isfinite(model(torch.ones(128,5000,device='cuda'))).all()
  del model,d;torch.cuda.empty_cache()
 result['cuda']='passed';result['peak_bytes']=torch.cuda.max_memory_allocated()
(R/'artifacts'/('preflight_gpu.json' if '--gpu' in sys.argv else 'preflight_cpu.json')).write_text(json.dumps(result,indent=2)+'\n');print(result)
