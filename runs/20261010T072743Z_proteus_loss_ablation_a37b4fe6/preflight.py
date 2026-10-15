"""Formula/gradient parity to the upstream text and real-input GPU smoke."""
import ast
import json
from pathlib import Path
import torch
import numpy as np
import port_base as B
from runner import setup, loader

setup()
upstream=Path('/tmp/proteus_audit.HOKVvy/Adaptive-WF-Attack-4cdab4163bf3de7036a2498dac533c888b97664d/exp/proteus.py')
tree=ast.parse(upstream.read_text())
names={'compute_gaussian_kernel','calculate_mmd_loss','compute_softmax_entropy'}
subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
namespace={'torch':torch}
exec(compile(subset,str(upstream),'exec'),namespace)
torch.manual_seed(93)
x=torch.randn(8,12,requires_grad=True); y=torch.randn(8,12,requires_grad=True)
a=B.calculate_mmd_loss(x,y); b=namespace['calculate_mmd_loss'](x,y)
assert torch.equal(a,b)
ga=torch.autograd.grad(a,(x,y),retain_graph=True); gb=torch.autograd.grad(b,(x,y))
assert all(torch.equal(u,v) for u,v in zip(ga,gb))
z=torch.randn(8,102)
assert torch.equal(B.entropy(z),namespace['compute_softmax_entropy'](z))

# A pseudo pool change must not change the independent target stream.
x=np.zeros((1024,2),np.float32); t=np.zeros((1024,2),np.float32)
l1=loader(x,t,np.zeros(1024,dtype=np.int64),np.arange(1024),2026)
l2=loader(x,t,np.zeros(1024,dtype=np.int64),np.arange(1024),2026)
assert torch.equal(next(iter(l1))[3],next(iter(l2))[3])

source=B.load_prepared(B.C['prepared']['source_valid'],'source')
model,_=B.load_model(21729)
tx=torch.from_numpy(source['timestamps'][:128]).to(B.DEVICE)
tt=torch.from_numpy(source['tam'][:128]).to(B.DEVICE)
labels=torch.from_numpy(source['labels'][:128]).to(B.DEVICE)
model.train(); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
so,sf=B.forward_with_features(model,tx,tt)
to,tf=B.forward_with_features(model,tx,tt)
po,_=B.forward_with_features(model,tx,tt)
m=to.softmax(1).mean(0)
loss=torch.nn.functional.cross_entropy(so,labels)+torch.nn.functional.cross_entropy(po,po.detach().argmax(1))+B.entropy(to).mean()+(m*torch.log(m+1e-5)).sum()+B.calculate_mmd_loss(sf,tf)
loss.backward(); optimizer.step()
assert torch.isfinite(loss)
B.save_json('preflight.json',{'passed':True,'mmd_value_and_gradient_upstream_parity':True,'entropy_upstream_parity':True,
    'source_target_loader_rng_isolated':True,'gpu_smoke_batch':128,'gpu':torch.cuda.get_device_name(0),
    'peak_memory_bytes':torch.cuda.max_memory_allocated(),'upstream_sha256':B.sha(upstream)})
print('Preflight passed; peak GiB',torch.cuda.max_memory_allocated()/1024**3)
