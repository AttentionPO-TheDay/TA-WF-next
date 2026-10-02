"""Mechanism checks before training; synthetic data only, no valid scoring."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from model import MultiViewModel,supervised_contrastive,span_mask,learning_rate,KINDS
from prepare import relative_tam

class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def batch(self):
        rng=torch.Generator().manual_seed(7)
        return torch.randn(4,5000,generator=rng),torch.randint(0,4,(4,2,1800),generator=rng).float()

    def test_initialization_and_identity(self):
        x,tam=self.batch();torch.manual_seed(21729);base=MultiViewModel('baseline').eval()
        expected=base(x,tam)
        for kind in KINDS[1:]:
            torch.manual_seed(21729);m=MultiViewModel(kind).eval()
            for k,v in m.state_dict().items():
                if kind=='cpu_temporal_cnn' and k.startswith('head.'):continue
                if k in base.state_dict():self.assertTrue(torch.equal(v,base.state_dict()[k]),(kind,k))
            if kind!='cpu_temporal_cnn':torch.testing.assert_close(m(x,tam),expected,rtol=1e-5,atol=1e-6)

    def test_packet_invariance_and_batch_independence(self):
        x,tam=self.batch()
        for kind in ['scale_concat','attention_readout','cpu_temporal_cnn']:
            m=MultiViewModel(kind).eval()
            a=m(x,tam)
            torch.testing.assert_close(a,m(x*100,tam),rtol=0,atol=0)
            torch.testing.assert_close(a[:1],m(x[:1],tam[:1]),atol=1e-5,rtol=1e-5)
            self.assertTrue(torch.isfinite(m(torch.zeros_like(x),torch.zeros_like(tam))).all())

    def test_joint_gradients(self):
        x,tam=self.batch()
        for kind in ['scale_concat','attention_readout','cpu_temporal_cnn','supcon']:
            m=MultiViewModel(kind);y=torch.tensor([1,1,2,2]);z,f=m(x,tam,return_features=True)
            loss=F.cross_entropy(z,y)+(.05*supervised_contrastive(f,y) if kind=='supcon' else 0)
            loss.backward()
            for name,p in m.generator.named_parameters():
                if p.requires_grad:self.assertIsNotNone(p.grad);self.assertGreater(p.grad.norm().item(),0,name)
                else:self.assertIsNone(p.grad)
            if kind=='attention_readout':self.assertGreater(m.readout_query.grad.norm().item(),0)

    def test_contrastive_formula_and_singletons(self):
        f=torch.tensor([[1.,0],[.8,.2],[0,1],[.1,.9]],requires_grad=True);y=torch.tensor([0,0,1,1])
        z=F.normalize(f,dim=-1);scores=z@z.T/.1;terms=[]
        for i in range(4):
            others=[j for j in range(4) if i!=j];pos=[j for j in others if y[j]==y[i]]
            terms.append(sum(-(scores[i,j]-torch.logsumexp(scores[i,others],0)) for j in pos)/len(pos))
        torch.testing.assert_close(supervised_contrastive(f,y),torch.stack(terms).mean())
        loss=supervised_contrastive(f,torch.arange(4));self.assertEqual(float(loss),0);loss.backward();self.assertTrue(torch.isfinite(f.grad).all())

    def test_relative_time_mass_direction_and_scale_invariance(self):
        x=np.array([.25,-.5,1.,-.125,0],np.float32)
        a=relative_tam(x);b=relative_tam(x*8)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a.sum(1),[2,2])
        self.assertEqual(a[0,1799],1);self.assertEqual(a.sum(),4)
        self.assertEqual(relative_tam(np.zeros(5,np.float32)).sum(),0)

    def test_mask_joint_interval_and_rng(self):
        tam=torch.ones(64,2,1800)
        a=span_mask(tam,torch.Generator().manual_seed(7));b=span_mask(tam,torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a[:,0],a[:,1]));self.assertTrue(tam.eq(1).all())
        for row in a[:,0]:
            idx=(row==0).nonzero().flatten()
            self.assertIn(len(idx),[0,90])
            if len(idx):self.assertEqual(int(idx[-1]-idx[0]),89)

    def test_schedule(self):
        self.assertAlmostEqual(learning_rate('cosine_lr',1),.001)
        self.assertAlmostEqual(learning_rate('cosine_lr',12800),.0001)
        self.assertEqual(learning_rate('baseline',6400),.001)
        self.assertEqual(learning_rate('baseline',6401),.0003)

if __name__=='__main__':unittest.main()
