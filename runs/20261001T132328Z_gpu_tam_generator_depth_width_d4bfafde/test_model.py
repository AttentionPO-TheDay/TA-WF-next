"""Checks for controlled depth/width change and nonblocking residual gradients."""
import unittest,torch
from model import MultiViewModel,CONDITIONS,parameter_counts
from encoder_base import MultiViewModel as PreviousModel
torch.set_num_threads(2)

def example():
 x=torch.zeros(2,5000);x[:,:81]=torch.arange(1,82).float()/7
 t=torch.zeros(2,2,1800);t[0,0,20:63]=2;t[0,1,64:101]=3;t[1,:,100:111]=4
 return x,t

class Checks(unittest.TestCase):
 def test_shared_classifier_and_spatial_initialization(self):
  models={}
  for kind in CONDITIONS:
   torch.manual_seed(21729);models[kind]=MultiViewModel(kind)
  state=models['d2_w16'].state_dict()
  for kind,m in models.items():
   for k,v in m.state_dict().items():
    if not k.startswith('generator.'):self.assertTrue(torch.equal(v,state[k]),k)
  for a,b in [('d2_w16','d4_w16'),('d2_w32','d4_w32')]:
   for k,v in models[a].generator.time_conv.state_dict().items():self.assertTrue(torch.equal(v,models[b].generator.time_conv.state_dict()[k]),k)

 def test_original_and_identity_depth_parity(self):
  x,t=example();torch.manual_seed(22026);previous=PreviousModel('multiscale_d139').eval()
  torch.manual_seed(22026);baseline=MultiViewModel('d2_w16').eval()
  self.assertEqual(previous.state_dict().keys(),baseline.state_dict().keys())
  for k,v in baseline.state_dict().items():self.assertTrue(torch.equal(v,previous.state_dict()[k]),k)
  with torch.no_grad():torch.testing.assert_close(baseline(x,t),previous(x,t),atol=0,rtol=0)
  for width in (16,32):
   torch.manual_seed(22026);shallow=MultiViewModel(f'd2_w{width}').eval()
   torch.manual_seed(22026);deep=MultiViewModel(f'd4_w{width}').eval()
   with torch.no_grad():torch.testing.assert_close(shallow(x,t),deep(x,t),atol=0,rtol=0)

 def test_token_interface_fixed_stats_and_no_packet_information(self):
  x,t=example();reference=None
  for kind in CONDITIONS:
   m=MultiViewModel(kind).eval()
   with torch.no_grad():
    tok=m.generator(x,t);out=m(x,t)
    torch.testing.assert_close(out,m(torch.ones_like(x)*-100,t),atol=0,rtol=0)
    torch.testing.assert_close(out[:1],m(x[:1],t[:1]),atol=3e-6,rtol=3e-5)
    self.assertTrue(torch.isfinite(m(x,torch.zeros_like(t))).all())
   self.assertEqual(tuple(tok['time'].shape),(2,120,110))
   self.assertFalse(tok['packet'].any());self.assertFalse(tok['packet_mask'].any());self.assertTrue(tok['time_mask'].all())
   stats=tok['time'][...,:30]
   if reference is None:reference=stats
   else:torch.testing.assert_close(stats,reference,atol=0,rtol=0)

 def test_added_pointwise_depth_preserves_spatial_support(self):
  # Force nonlinear branch weights nonzero to verify support after identity init.
  g=MultiViewModel('d4_w32').generator
  with torch.no_grad():
   for p in g.time_conv.parameters():p.fill_(.02)
   for layer in g.depth_mix:layer.weight.fill_(.02);layer.bias.zero_()
   z=torch.zeros(1,2,1800)
   for d in (1,3,9):
    edge=z.clone();edge[0,0,900+4*d]=1
    self.assertGreater((g.encode_scale(edge,d)-g.encode_scale(z,d))[0,:,900].abs().sum(),0)
    outside=z.clone();outside[0,0,900+4*d+1]=1
    self.assertEqual((g.encode_scale(outside,d)-g.encode_scale(z,d))[0,:,900].abs().sum(),0)

 def test_all_active_gradients_including_zero_initialized_branches(self):
  x,t=example()
  for kind in CONDITIONS:
   torch.manual_seed(23407);m=MultiViewModel(kind)
   before={k:p.detach().clone() for k,p in m.named_parameters()}
   opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
   torch.nn.functional.cross_entropy(m(x,t),torch.tensor([4,7])).backward()
   for k,p in m.generator.named_parameters():
    if p.requires_grad:self.assertIsNotNone(p.grad,k);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0,k)
    else:self.assertIsNone(p.grad,k)
   opt.step()
   for k,p in m.named_parameters():
    if not p.requires_grad:self.assertTrue(torch.equal(p,before[k]),k)
   for k,p in m.generator.depth_mix.named_parameters():self.assertFalse(torch.equal(p,before['generator.depth_mix.'+k]),k)

if __name__=='__main__':
 for kind in CONDITIONS:print(kind,parameter_counts(MultiViewModel(kind)))
 unittest.main()
