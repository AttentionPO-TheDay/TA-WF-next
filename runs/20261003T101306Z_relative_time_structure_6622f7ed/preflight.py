import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch, numpy as np
from common import RUN,config,verify_freeze,deterministic,save
from model import MultiViewModel

def main():
    c=config();verify_freeze();deterministic('cuda',2)
    assert torch.cuda.is_available()
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
    assert set(raw)=={'source','valid'}
    for role,n in [('source',15300),('valid',510)]:
        assert raw[role]['tam'].shape==(n,2,1800)
        assert raw[role]['timestamps'].shape==(n,5000)
    models=[]
    for kind in c['conditions']:
        torch.manual_seed(21729);torch.cuda.manual_seed_all(21729)
        models.append(MultiViewModel(kind).cuda())
    a,b=models
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    ts=raw['source']['timestamps'][:64].cuda();tam=raw['source']['tam'][:64].cuda();y=raw['source']['labels'][:64].cuda()
    f=b.relative_time_features(ts)
    assert f.shape==(64,120,4) and torch.isfinite(f).all()
    # Independent CPU reference, including the declared left-endpoint pair-bin rule.
    x=ts[:2].cpu().numpy();bins=np.arange(5000)*120//5000;ref=np.zeros((2,120,4))
    for j in range(120):
        ix=bins==j;v=x[:,ix];valid=v!=0;count=valid.sum(1).clip(1)
        ref[:,j,0]=(np.clip(v,-80,80)/80*valid).sum(1)/count
        ref[:,j,1]=valid.mean(1)
        ref[:,j,2]=(np.log1p(np.clip(abs(v),0,80))/np.log(81)*valid).sum(1)/count
        pair=(x[:,:-1]!=0)&(x[:,1:]!=0);pi=bins[:-1]==j
        ref[:,j,3]=(((np.sign(x[:,:-1])!=np.sign(x[:,1:]))&pair)[:,pi]).sum(1)/pair[:,pi].sum(1).clip(1)
    assert np.allclose(f[:2].cpu().numpy(),ref,atol=1e-6)
    assert torch.equal(b.relative_time_features(torch.zeros_like(ts)),torch.zeros_like(f))
    for model in models:
        model.train();model.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(ts,tam),y);loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert b.relative_projection.weight.grad.abs().sum()>0
    assert torch.equal(a.relative_projection.weight.grad,torch.zeros_like(a.relative_projection.weight.grad))
    result={'passed':True,'freeze_files':16,'same_initialization':True,'batch_size':64,'feature_cpu_reference':True,'deterministic_forward_backward':True,'gpu':torch.cuda.get_device_name(0),'future_access':False}
    save(RUN/'artifacts/preflight.json',result);print(result,flush=True)
if __name__=='__main__':main()
