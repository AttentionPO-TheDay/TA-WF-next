"""Semantic ablation checks: removed information cannot influence predictions."""
import unittest
import torch
from model import MultiViewModel,VIEWS,parameter_counts
from base_model import MultiViewModel as BaseModel
torch.set_num_threads(2)

def example():
 x=torch.zeros(3,5000)
 x[0,:81]=torch.arange(1,82).float()/11
 x[0,1:81:2]*=-1
 x[1,:112]=torch.arange(1,113).float()/7
 t=torch.zeros(3,2,1800)
 t[0,:,10:21]=3;t[1,0,30:70]=2
 return x,t

class Checks(unittest.TestCase):
 def test_fusion_matches_original_and_initialization(self):
  torch.manual_seed(21729);base=BaseModel('learned','transformer').eval()
  x,t=example()
  for view in VIEWS:
   torch.manual_seed(21729);m=MultiViewModel(view).eval()
   self.assertEqual(set(m.state_dict()),set(base.state_dict()))
   for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,base.state_dict()[k]),k)
   if view=='fusion':
    with torch.no_grad():
     torch.testing.assert_close(m(x,t),base(x,t),atol=0,rtol=0)
     a,b=m.generator(x,t),base.generator(x,t)
     for k in a:torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)

 def test_removed_inputs_cannot_influence_predictions(self):
  x,t=example();changed=x.sign()*torch.rand_like(x)*600
  for view in VIEWS:
   torch.manual_seed(23407);m=MultiViewModel(view).eval()
   with torch.no_grad():
    reference=m(x,t)
    if view=='tam_only':torch.testing.assert_close(reference,m(torch.randn_like(x),t),atol=0,rtol=0)
    if view.startswith('packet_'):torch.testing.assert_close(reference,m(x,torch.rand_like(t)*20),atol=0,rtol=0)
    if view=='packet_direction':torch.testing.assert_close(reference,m(changed,torch.rand_like(t)*20),atol=0,rtol=0)

 def test_masks_and_empty_input(self):
  x,t=example()
  for view in VIEWS:
   m=MultiViewModel(view).eval()
   with torch.no_grad():
    p,z,pm,tm=m.forward_tokens(x,t)
    empty=m(torch.zeros_like(x),torch.zeros_like(t))
    batch=m(x,t);single=m(x[:1],t[:1])
   self.assertTrue(torch.isfinite(empty).all())
   self.assertFalse(p[~pm].any());self.assertFalse(z[~tm].any())
   if view=='tam_only':self.assertFalse(pm.any())
   elif view!='fusion':self.assertFalse(tm.any())
   torch.testing.assert_close(batch[:1],single,atol=3e-6,rtol=3e-5)

 def test_active_generator_gradients_and_frozen_parameters(self):
  x,t=example()
  for view in VIEWS:
   torch.manual_seed(22026);m=MultiViewModel(view)
   before={k:p.detach().clone() for k,p in m.named_parameters()}
   opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
   torch.nn.functional.cross_entropy(m(x,t),torch.tensor([2,9,20])).backward()
   for k,p in m.generator.named_parameters():
    if p.requires_grad:
     self.assertIsNotNone(p.grad,k);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0)
    else:self.assertIsNone(p.grad,k)
   if view=='packet_direction':self.assertFalse(m.generator.packet_conv[0].weight.grad[:,2,:].any())
   opt.step()
   for k,p in m.named_parameters():
    if not p.requires_grad:self.assertTrue(torch.equal(p,before[k]),k)

if __name__=='__main__':
 for view in VIEWS:
  print(view,parameter_counts(MultiViewModel(view)))
 unittest.main()
