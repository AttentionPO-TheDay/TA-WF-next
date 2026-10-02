import unittest
import torch
from torch.nn import functional as F
from model import MultiViewModel,make_optimizer,learning_rate,span_mask,FACTORS
from candidate_base import MultiViewModel as Historical

class FactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def inputs(self):
        g=torch.Generator().manual_seed(4)
        return torch.randn(4,5000,generator=g),torch.randint(0,100,(4,2,1800),generator=g).float()

    def test_shared_initialization_and_historical_parity(self):
        x,tam=self.inputs();torch.manual_seed(21729);old=Historical('span_mask').eval()
        for kind in FACTORS:
            torch.manual_seed(21729);m=MultiViewModel(kind).eval()
            self.assertEqual(set(old.state_dict()),set(m.state_dict()))
            self.assertTrue(all(torch.equal(v,old.state_dict()[k]) for k,v in m.state_dict().items()))
            if FACTORS[kind][0]=='log1p':torch.testing.assert_close(m(x,tam),old(x,tam),atol=0,rtol=0)

    def test_input_mapping_and_no_packet_effect(self):
        x,tam=self.inputs()
        for kind,(scale,recipe) in FACTORS.items():
            m=MultiViewModel(kind).eval();tokens=m.generator(x,tam)
            expected=(tam.log1p() if scale=='log1p' else tam).reshape(4,2,120,15).permute(0,2,1,3).reshape(4,120,30)
            torch.testing.assert_close(tokens['time'][...,:30],expected,atol=0,rtol=0)
            torch.testing.assert_close(m(x,tam),m(x*100,tam),atol=0,rtol=0)
            torch.testing.assert_close(m(x,tam)[:1],m(x[:1],tam[:1]),atol=1e-5,rtol=1e-5)
            self.assertTrue(torch.isfinite(m(torch.zeros_like(x),torch.zeros_like(tam))).all())

    def test_gradient_and_optimizer_update(self):
        x,tam=self.inputs()
        for kind in FACTORS:
            m=MultiViewModel(kind);before={k:p.detach().clone() for k,p in m.named_parameters()}
            opt=make_optimizer(m,kind);F.cross_entropy(m(x,tam),torch.tensor([0,1,2,3])).backward()
            for k,p in m.generator.named_parameters():
                if p.requires_grad:self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.norm().item(),0,k)
                else:self.assertIsNone(p.grad)
            opt.step()
            for k,p in m.named_parameters():
                if not p.requires_grad:self.assertTrue(torch.equal(before[k],p.detach()))
            self.assertTrue(any(not torch.equal(before['generator.'+k],p.detach()) for k,p in m.generator.named_parameters() if p.requires_grad))

    def test_recipe_and_schedule(self):
        for kind,(scale,recipe) in FACTORS.items():
            opt=make_optimizer(MultiViewModel(kind),kind)
            self.assertEqual(type(opt),torch.optim.Adam if recipe=='rfstyle' else torch.optim.AdamW)
            self.assertEqual(opt.defaults['weight_decay'],.001 if recipe=='rfstyle' else .0001)
            self.assertAlmostEqual(learning_rate(kind,1),.0005 if recipe=='rfstyle' else .001)
            if recipe=='rfstyle':self.assertAlmostEqual(learning_rate(kind,12800),.0005*.2**(12799/12800))

    def test_mask_matches_historical(self):
        x,tam=self.inputs();g1=torch.Generator().manual_seed(71729);g2=torch.Generator().manual_seed(71729)
        self.assertTrue(torch.equal(span_mask(tam,g1),span_mask(tam,g2)))

if __name__=='__main__':unittest.main()
