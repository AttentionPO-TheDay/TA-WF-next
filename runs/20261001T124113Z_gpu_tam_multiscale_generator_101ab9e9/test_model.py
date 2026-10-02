"""Semantic and numerical checks for context expansion with shared weights."""
import unittest,torch
from model import MultiViewModel,ENCODERS,parameter_counts
from view_base import MultiViewModel as ViewModel
torch.set_num_threads(2)

def example():
 torch.manual_seed(543)
 x=torch.randn(2,5000)
 tam=torch.zeros(2,2,1800)
 tam[0,0,10:49]=2;tam[0,1,55:77]=3;tam[1,:,63:91]=1
 return x,tam

class Checks(unittest.TestCase):
 def test_parameter_initialization_and_exact_local_parity(self):
  x,t=example();torch.manual_seed(21729);old=ViewModel('tam_only').eval()
  counts=parameter_counts(old)
  for encoder in ENCODERS:
   torch.manual_seed(21729);m=MultiViewModel(encoder).eval()
   self.assertEqual(parameter_counts(m),counts)
   self.assertEqual(m.state_dict().keys(),old.state_dict().keys())
   for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,old.state_dict()[k]),k)
   with torch.no_grad():
    tok=m.generator(x,t);prev=old.generator(x,t)
    torch.testing.assert_close(tok['time'][...,:30],prev['time'][...,:30],atol=0,rtol=0)
    if encoder=='local_d1':torch.testing.assert_close(m(x,t),old(x,t),atol=0,rtol=0)
   self.assertEqual(tuple(tok['time'].shape),(2,120,110))

 def test_shared_scale_reconstruction_and_order(self):
  x,t=example();m=MultiViewModel('multiscale_d139')
  log=t.log1p();z=sum(m.generator.encode_scale(log,d) for d in (1,3,9))/3
  expected=z.reshape(2,16,120,5,3).mean(-1).permute(0,2,3,1).reshape(2,120,80)
  torch.testing.assert_close(m.generator(x,t)['time'][...,30:],expected,atol=0,rtol=0)
  fixed=m.generator(x,t)['time'][...,:30]
  torch.testing.assert_close(fixed,t.log1p().reshape(2,2,120,15).permute(0,2,1,3).reshape(2,120,30))
  self.assertEqual(len(list(m.generator.time_conv.parameters())),4)

 def test_receptive_fields(self):
  # Positive synthetic weights avoid relying on accidental random cancellation.
  x=torch.zeros(1,5000);t=torch.zeros(1,2,1800)
  base=MultiViewModel('local_d1').generator
  with torch.no_grad():
   for layer in base.time_conv:layer.weight.fill_(.05);layer.bias.zero_()
   for d in (1,3,9):
    changed=t.clone();changed[0,0,900+4*d]=1
    diff=base.encode_scale(changed.log1p(),d)-base.encode_scale(t.log1p(),d)
    self.assertGreater(diff[0,:,900].abs().sum(),0)
    changed=t.clone();changed[0,0,900+4*d+1]=1
    diff=base.encode_scale(changed.log1p(),d)-base.encode_scale(t.log1p(),d)
    self.assertEqual(diff[0,:,900].abs().sum(),0)

 def test_removed_packet_and_batch_independence(self):
  x,t=example()
  for encoder in ENCODERS:
   m=MultiViewModel(encoder).eval()
   with torch.no_grad():
    z=m(x,t);changed=m(torch.zeros_like(x),t)
    torch.testing.assert_close(z,changed,atol=0,rtol=0)
    torch.testing.assert_close(z[:1],m(x[:1],t[:1]),atol=3e-6,rtol=3e-5)
    self.assertTrue(torch.isfinite(m(x,torch.zeros_like(t))).all())
    tokens=m.generator(x,t)
    self.assertFalse(tokens['packet'].any());self.assertFalse(tokens['packet_mask'].any());self.assertTrue(tokens['time_mask'].all())

 def test_active_gradients_all_scales_and_frozen_state(self):
  x,t=example()
  for encoder in ENCODERS:
   m=MultiViewModel(encoder);before={k:v.detach().clone() for k,v in m.named_parameters()}
   opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
   torch.nn.functional.cross_entropy(m(x,t),torch.tensor([4,5])).backward()
   for k,p in m.generator.named_parameters():
    if p.requires_grad:self.assertIsNotNone(p.grad,k);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0)
    else:self.assertIsNone(p.grad,k)
   opt.step()
   for k,p in m.named_parameters():
    if not p.requires_grad:self.assertTrue(torch.equal(p,before[k]),k)

if __name__=='__main__':
 for kind in ENCODERS:print(kind,parameter_counts(MultiViewModel(kind)))
 unittest.main()
