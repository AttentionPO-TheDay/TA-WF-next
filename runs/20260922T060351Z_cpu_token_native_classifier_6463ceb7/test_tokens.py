import unittest
import numpy as np
import torch
from train import tokens,TokenNet
class Checks(unittest.TestCase):
 def test_fields(self):
  x=np.zeros((2,5000)); x[0,:4]=1; x[1,:7]=1
  d,e,c=tokens(x); np.testing.assert_array_equal(c[0],c[1]); self.assertNotEqual(e[0,0],e[1,0])
  self.assertEqual(c[0,0],3); self.assertEqual(d[0,0],2)
 def test_mask_gradient(self):
  torch.set_num_threads(2)
  for view in ['exact','coarse']:
   torch.manual_seed(1); m=TokenNet(view)
   d=torch.zeros(2,512,dtype=torch.long); d[:,:10]=1
   l=torch.ones(2,512,dtype=torch.float32 if view=='exact' else torch.long)
   a=m(d,l); other=l.clone(); other[:,10:]=5
   torch.testing.assert_close(a,m(d,other))
   a.sum().backward(); self.assertGreater(m.direction.weight.grad.abs().sum(),0)
if __name__=='__main__': unittest.main()
