import unittest
import numpy as np
import torch
from train import targets,Model
class Check(unittest.TestCase):
 def test_targets(self):
  x=np.zeros((1,5000),np.float32); x[0,:5]=[1,1,-1,-1,1]
  w,r=targets(x)
  np.testing.assert_allclose(w[0,:2],[.6,.5])
  np.testing.assert_allclose(w[0,200:202],[.6,.5])
  np.testing.assert_allclose(r[0],np.log1p([2,1.5,.5,2,1,2,0,2]),rtol=1e-6)
 def test_aux_gradient_and_inference(self):
  torch.set_num_threads(2); torch.manual_seed(1); m=Model()
  x=torch.randn(2,5000); aux=torch.nn.Linear(1024,8)
  loss=aux(m.features(x)).square().mean(); loss.backward()
  self.assertGreater(m.body[0].weight.grad.abs().sum(),0)
  self.assertIsNone(m.head.weight.grad)
  before=m(x).detach(); aux.weight.data.zero_()
  torch.testing.assert_close(before,m(x))
if __name__=='__main__': unittest.main()
