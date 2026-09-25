import unittest
import numpy as np
import torch
from train import permuted, RunNet

class Checks(unittest.TestCase):
    def test_permutation(self):
        a=np.arange(2*512*4).reshape(2,512,4).astype(np.float32)
        m=np.zeros((2,512),bool); m[0,:31]=True; m[1,:71]=True
        d={'exact':(a,None,None),'coarse':(np.concatenate([a,a],2),None,None),'mask':m}
        s,p=permuted(d,'source'); t,q=permuted(d,'source')
        np.testing.assert_array_equal(p,q)
        for i,n in enumerate([31,71]):
            np.testing.assert_array_equal(np.sort(p[i,:n]),np.arange(n))
            np.testing.assert_array_equal(p[i,n:],np.arange(n,512))
            np.testing.assert_array_equal(s['exact'][0][i],a[i,p[i]])
        np.testing.assert_array_equal(s['exact'][0],s['coarse'][0][:,:,:4])
        self.assertIs(s['mask'],m)
    def test_paired_initialization_and_padding(self):
        torch.set_num_threads(2)
        for c in (4,8):
            torch.manual_seed(17); a=RunNet(c)
            torch.manual_seed(17); b=RunNet(c)
            for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y)
            x=torch.randn(2,32,c); m=torch.zeros(2,32,dtype=torch.bool); m[:,:16]=True
            y=x.clone(); y[:,16:]=999
            torch.testing.assert_close(a(x,m),a(y,m))

if __name__=='__main__': unittest.main()
