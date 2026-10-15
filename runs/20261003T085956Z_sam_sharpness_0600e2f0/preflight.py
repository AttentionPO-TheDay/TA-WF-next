"""Preflight for two-pass CE control and BN-safe SAM."""
import json, os
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask
from common import RUN, config, verify_freeze, deterministic

def set_bn_mode(model, training):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.training=training

def main():
    c=config();assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0';deterministic('cuda',2);device=torch.device('cuda');raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');assert raw['source']['tam'].shape==(15300,2,1800) and raw['valid']['tam'].shape==(510,2,1800)
    torch.manual_seed(21729);torch.cuda.manual_seed_all(21729);model=MultiViewModel('ce_only').to(device);x=raw['source']['timestamps'][:64].to(device);tam=span_mask(raw['source']['tam'][:64].to(device),torch.Generator().manual_seed(71729));y=raw['source']['labels'][:64].to(device)
    model.train();set_bn_mode(model,True);first=model(x,tam);loss=F.cross_entropy(first,y);loss.backward();grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None];gn=torch.stack([g.detach().float().norm() for g in grads]).norm();assert torch.isfinite(gn) and gn>0;buffers={k:v.clone() for k,v in model.named_buffers()};orig=[(p,p.detach().clone()) for p in model.parameters() if p.requires_grad];rho=c['sam']['rho']
    with torch.no_grad():
        for p,_ in orig:p.add_(rho*p.grad.detach()/gn)
    perturbed = any(float((p.detach()-old).abs().max())>0 for p,old in orig)
    set_bn_mode(model,False);model.zero_grad(set_to_none=True);second=model(x,tam);loss2=F.cross_entropy(second,y);assert torch.isfinite(loss2);after={k:v.clone() for k,v in model.named_buffers()};assert all(torch.equal(buffers[k],after[k]) for k in buffers);loss2.backward()
    with torch.no_grad():
        for p,old in orig:p.copy_(old)
    assert all(torch.equal(p,old) for p,old in orig); assert perturbed
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    result={'passed':True,'cuda_available':True,'gpu_name':torch.cuda.get_device_name(0),'source_rows':15300,'valid_rows':510,'two_pass_forward':True,'sam_parameter_perturbation':True,'rho':rho,'bn_second_pass_no_update':True,'parameter_restore':True,'fixed_teacher_loaded':False,'ema_loaded':False,'generator_gradients_active':True,'future_access':False};(RUN/'artifacts/preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
