import unittest
import numpy as np
import torch
from train import extract, RunNet, PacketNet, WindowNet

class Checks(unittest.TestCase):
    def test_windows_fixed_slots(self):
        a=np.zeros((1,5000)); a[0,:3]=[1,1,-1]
        f=extract(a)
        np.testing.assert_allclose(f['windows'][0,:2],[2/3,1/2])
        np.testing.assert_allclose(f['windows'][0,200:202],[2/3,1/2])
        self.assertEqual(np.count_nonzero(f['windows']),4)
    def test_padding_ignored_and_gradient(self):
        torch.set_num_threads(2)
        net=RunNet(4); x=torch.randn(2,10,4); m=torch.zeros(2,10,dtype=torch.bool); m[:,:3]=True
        y=net(x,m); altered=x.clone(); altered[:,3:]=99
        torch.testing.assert_close(y,net(altered,m))
        y.sum().backward(); self.assertTrue(all(p.grad is not None for p in net.parameters()))
    def test_shapes_and_coarse_collision(self):
        a=np.zeros((2,5000)); a[0,:4]=1; a[1,:7]=1
        f=extract(a); np.testing.assert_array_equal(f['coarse'][0],f['coarse'][1])
        self.assertFalse(np.array_equal(f['exact'][0],f['exact'][1]))
        for model,x in [(PacketNet(),torch.from_numpy(f['packet'])),(WindowNet(),torch.from_numpy(f['windows']))]:
            self.assertEqual(tuple(model(x).shape),(2,102))

if __name__=='__main__': unittest.main()
