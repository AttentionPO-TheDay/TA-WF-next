import unittest
import torch
from train import RunNet

class Checks(unittest.TestCase):
    def test_padding_and_order(self):
        torch.set_num_threads(2); torch.manual_seed(1729)
        for c in (4,8):
            net=RunNet(c); x=torch.randn(2,32,c); m=torch.zeros(2,32,dtype=torch.bool); m[:,:16]=True
            y=net(x,m); other=x.clone(); other[:,16:]=999
            torch.testing.assert_close(y,net(other,m))
            shuffled=x.clone(); shuffled[:,:16]=x[:,:16].flip(1)
            self.assertGreater((net(shuffled,m)-y).abs().max().item(),1e-5)
            y.sum().backward(); self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
            padded=torch.cat([x,torch.randn(2,8,c)],1); pm=torch.cat([m,torch.zeros(2,8,dtype=torch.bool)],1)
            torch.testing.assert_close(y,net(padded,pm))

if __name__=='__main__': unittest.main()
