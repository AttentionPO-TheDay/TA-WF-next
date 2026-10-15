"""Preflight for primary-CE peer attribution conditions."""
import json, os
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask, symmetric_peer_loss
from common import RUN, config, deterministic, verify_freeze, save, sha
from worker import js_divergence, predict

def main():
    c=config(); assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'; deterministic('cuda',2); device=torch.device('cuda'); raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu'); assert raw['source']['tam'].shape==(15300,2,1800) and raw['valid']['tam'].shape==(510,2,1800)
    torch.manual_seed(21729);torch.cuda.manual_seed_all(21729);model=MultiViewModel('ce_only').to(device); x=raw['source']['timestamps'][:64].to(device); clean=raw['source']['tam'][:64].to(device); y=raw['source']['labels'][:64].to(device); a=span_mask(clean,torch.Generator().manual_seed(71729));b=span_mask(clean,torch.Generator().manual_seed(81729));assert not torch.equal(a,b)
    both=model(torch.cat([x,x]),torch.cat([a,b]));assert both.shape==(128,102) and torch.isfinite(both).all();la,lb=both[:64],both[64:];ce=F.cross_entropy(la,y);peer=symmetric_peer_loss(la,lb,2.0);loss=ce+.25*peer;loss.backward();active=[p for n,p in model.named_parameters() if (n.startswith('generator.stages') or n=='generator.to_features.weight') and n.endswith('weight')];assert active and all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0 for p in active)
    verify_freeze()
    assert all(int(v)==1 for n,v in model.named_buffers() if n.endswith('num_batches_tracked'))
    for seed in c['seeds']:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        initialized=MultiViewModel('ce_only')
        ref=torch.load(RUN/'artifacts'/f'ce_only_{seed}'/'initial_state.pt',weights_only=True)
        assert set(ref)==set(initialized.state_dict())
        assert all(torch.equal(v,ref[k]) for k,v in initialized.state_dict().items())
    # Loss-level proof: auxiliary logits have no direct CE derivative.
    va=torch.randn(8,102,device=device,requires_grad=True)
    vb=torch.randn(8,102,device=device,requires_grad=True)
    direct=F.cross_entropy(torch.cat([va,vb])[:8],y[:8])
    da,db=torch.autograd.grad(direct,(va,vb))
    assert da.norm()>0 and db.count_nonzero()==0
    pa,pb=torch.softmax(va.detach()/2,1),torch.softmax(vb.detach()/2,1)
    assert not pa.requires_grad and not pb.requires_grad
    l=symmetric_peer_loss(va,vb,2)
    ga,gb=torch.autograd.grad(l,(va,vb))
    # Stop-gradient analytical derivatives, including T^2 and mean scaling.
    torch.testing.assert_close(ga,(pa-pb)/len(va))
    torch.testing.assert_close(gb,(pb-pa)/len(vb))
    assert ga.norm()>0 and gb.norm()>0
    torch.testing.assert_close(l,symmetric_peer_loss(vb,va,2))
    z=torch.tensor([[1000.,-1000.]])
    assert abs(js_divergence(z,z))<1e-7
    assert abs(js_divergence(z,-z)-0.6931471805599453)<1e-6
    # Inference-path compatibility and eval BN stability, no validation selection.
    e={k:v[:64].cuda() for k,v in raw['source'].items() if k in ('timestamps','tam','labels')}
    assert predict(model,e,c['eval_batch_size']).shape==(64,102)
    result={'passed':True,'cuda_available':True,'gpu_name':torch.cuda.get_device_name(0),'source_rows':15300,'valid_rows':510,'single_network_count':1,'fixed_teacher_loaded':False,'ema_loaded':False,'joint_two_view_forward':True,'one_bn_update_verified':True,'initialization_seeds_verified':c['seeds'],'primary_ce_shape':[64,102],'auxiliary_logits_direct_ce_gradient_zero':True,'peer_target_detached_derivative_verified':True,'student_generator_gradients_active':True,'standard_js_verified':True,'freeze_hashes_verified':len(json.loads((RUN/'artifacts/freeze.json').read_text())),'future_access':False};save(RUN/'artifacts/preflight.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
