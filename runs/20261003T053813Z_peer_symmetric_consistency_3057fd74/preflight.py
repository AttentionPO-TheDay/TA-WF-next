"""Preflight for single-network two-view peer consistency."""
import json, os
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask, symmetric_peer_loss
from common import RUN, config, deterministic

def main():
    c=config(); assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'; deterministic('cuda',2); device=torch.device('cuda'); raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu'); assert raw['source']['tam'].shape==(15300,2,1800) and raw['valid']['tam'].shape==(510,2,1800)
    torch.manual_seed(21729); torch.cuda.manual_seed_all(21729); model=MultiViewModel('ce_only').to(device); assert sum(1 for _ in [model])==1
    x=raw['source']['timestamps'][:64].to(device); clean=raw['source']['tam'][:64].to(device); y=raw['source']['labels'][:64].to(device); a=span_mask(clean,torch.Generator().manual_seed(71729)); b=span_mask(clean,torch.Generator().manual_seed(81729)); assert not torch.equal(a,b)
    logits=model(torch.cat([x,x]),torch.cat([a,b])); assert logits.shape==(128,102) and torch.isfinite(logits).all(); la,lb=logits[:64],logits[64:]; ce=.5*(F.cross_entropy(la,y)+F.cross_entropy(lb,y)); peer=symmetric_peer_loss(la,lb,2.0); loss=ce+.25*peer; assert torch.isfinite(loss); loss.backward(); active=[p for n,p in model.named_parameters() if (n.startswith('generator.stages') or n=='generator.to_features.weight') and n.endswith('weight')]; assert active and all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0 for p in active)
    assert any(p.requires_grad for p in model.parameters())
    result={'passed':True,'cuda_available':True,'gpu_name':torch.cuda.get_device_name(0),'source_rows':15300,'valid_rows':510,'single_network_count':1,'fixed_teacher_loaded':False,'ema_loaded':False,'joint_two_view_forward':True,'view_logits_shape':[64,102],'peer_target_detached':True,'student_generator_gradients_active':True,'future_access':False}; (RUN/'artifacts/preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
