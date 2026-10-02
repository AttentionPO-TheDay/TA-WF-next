from runner import *
from df_reference import SamePool
from sklearn.metrics import f1_score

torch.set_num_threads(4)
torch.manual_seed(171)
checks=[]
for n in [1,4,7,8,9,20,79,313,1250,5000]:
    x=torch.randn(2,3,n);total=max(0,((n+3)//4-1)*4+8-n);left=total//2
    ref=torch.stack([x[:,:,max(0,j*4-left):min(n,j*4-left+8)].max(-1).values for j in range((n+3)//4)],-1)
    torch.testing.assert_close(SamePool()(x),ref,rtol=0,atol=0)
checks.append('SAME pool numeric reference across odd/even lengths')
m=DFReference().cuda();x=torch.randn(4,1,5000,device='cuda');shape=[]
hooks=[l.register_forward_hook(lambda mod,ins,out:shape.append(out.shape[-1])) for l in m.features if isinstance(l,SamePool)]
z=m(x);assert shape==[1250,313,79,20] and z.shape==(4,102)
loss=F.cross_entropy(z,torch.tensor([0,1,2,3],device='cuda'));loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
checks.extend(['DF pool lengths 1250/313/79/20','DF gradients finite'])
for h in hooks:h.remove()
m.eval()
with torch.inference_mode():a=m(x);b=m(x)
torch.testing.assert_close(a,b,rtol=0,atol=0);checks.append('DF eval repeat exact')
# CPU/GPU import capability and current model source-only forward/gradient.
p=ROOT/'runs/20260927T032819Z_source_size_local_depth_factorial_dae714b7/artifacts/prepared.pt'
d=torch.load(p,map_location='cpu',weights_only=False);e=device_entry(subset(d['source150'],slice(0,4)));t=new_model('transformer',True)
z=logits(t,e,slice(None),'transformer');F.cross_entropy(z,e['labels']).backward()
assert z.shape==(4,102) and all(torch.isfinite(q.grad).all() for q in t.parameters() if q.grad is not None)
checks.append('current Transformer source-only CUDA forward and gradient')
savej(RUN/'artifacts/preflight.json',{'checks':checks,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'training_steps':0,'valid_scored':False})
print(json.dumps(checks))
