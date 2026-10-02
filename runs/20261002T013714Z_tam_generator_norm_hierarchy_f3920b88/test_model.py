import unittest
import torch
from torch.nn import functional as F
from model import MultiViewModel,FACTORS,make_optimizer,span_mask
from input_base import MultiViewModel as OldModel

class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def inputs(self):
        g=torch.Generator().manual_seed(7)
        return torch.randn(4,5000,generator=g),torch.randint(0,10,(4,2,1800),generator=g).float()

    def test_initialization_and_baseline_identity(self):
        x,tam=self.inputs();torch.manual_seed(21729);old=OldModel('log_current').eval()
        for kind in FACTORS:
            torch.manual_seed(21729);m=MultiViewModel(kind).eval()
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in old.state_dict().items()))
            if kind=='flat_none':torch.testing.assert_close(m(x,tam),old(x,tam),atol=0,rtol=0)
            if FACTORS[kind][0]:
                self.assertEqual(len(m.generator.channel_norms),2)
                for norm in m.generator.channel_norms:
                    self.assertEqual(norm.normalized_shape,(16,));self.assertTrue(norm.weight.eq(1).all());self.assertTrue(norm.bias.eq(0).all())

    def test_shape_fixed_features_batch_independence_and_no_packet(self):
        x,tam=self.inputs();fixed=tam.log1p().reshape(4,2,120,15).permute(0,2,1,3).reshape(4,120,30)
        for kind in FACTORS:
            m=MultiViewModel(kind).eval();g=m.generator(x,tam)
            self.assertEqual(g['time'].shape,(4,120,110));torch.testing.assert_close(g['time'][...,:30],fixed,atol=0,rtol=0)
            torch.testing.assert_close(m(x,tam),m(x*100,tam),atol=0,rtol=0)
            torch.testing.assert_close(m(x,tam)[:1],m(x[:1],tam[:1]),atol=1e-5,rtol=1e-5)
            self.assertTrue(g['time_mask'].all());self.assertFalse(g['packet_mask'].any())
            self.assertTrue(torch.isfinite(m(torch.zeros_like(x),torch.zeros_like(tam))).all())

    def test_gradient_and_update(self):
        x,tam=self.inputs()
        for kind in FACTORS:
            m=MultiViewModel(kind);initial={k:p.detach().clone() for k,p in m.named_parameters()};opt=make_optimizer(m,kind)
            F.cross_entropy(m(x,tam),torch.tensor([0,1,2,3])).backward()
            for name,p in m.generator.named_parameters():
                if p.requires_grad:self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.norm().item(),0,name)
                else:self.assertIsNone(p.grad)
            opt.step()
            for name,p in m.named_parameters():
                if not p.requires_grad:self.assertTrue(torch.equal(initial[name],p.detach()))
                elif name.startswith('generator.'):self.assertFalse(torch.equal(initial[name],p.detach()),name)

    def test_channel_norm_uses_no_batch_or_time_moments(self):
        m=MultiViewModel('flat_norm').eval();z=torch.randn(2,16,19)
        norm=m.generator.channel_norms[0]
        out=norm(z.transpose(1,2)).transpose(1,2)
        modified=z.clone();modified[1]*=100;modified[0,:,1:]+=100
        other=norm(modified.transpose(1,2)).transpose(1,2)
        torch.testing.assert_close(out[0,:,0],other[0,:,0],atol=0,rtol=0)
        self.assertFalse(any('running' in k for k in norm.state_dict()))

    def test_support_and_grid_alignment(self):
        for kind,hier in [('flat_none',False),('hier_none',True)]:
            m=MultiViewModel(kind).eval()
            for dilation in [1,3,9]:
                x=torch.ones(1,2,1800,requires_grad=True)
                index=200
                z=m.generator.encode_scale(x,dilation)
                if not hier:z=z.reshape(1,16,600,3).mean(-1)
                z[0,:,index].sum().backward()
                support=x.grad.abs().sum(1)[0].nonzero().flatten()
                lower=600-(8 if hier else 4)*dilation
                upper=602+(8 if hier else 4)*dilation
                self.assertEqual(int(support[0]),lower);self.assertEqual(int(support[-1]),upper)
                self.assertEqual(int(support[-1]-support[0]+1),16*dilation+3 if hier else 8*dilation+3)

    def test_mask_rng_and_optimizer_unchanged(self):
        x,tam=self.inputs()
        for kind in FACTORS:
            a=span_mask(tam,torch.Generator().manual_seed(71729));b=span_mask(tam,torch.Generator().manual_seed(71729))
            self.assertTrue(torch.equal(a,b));opt=make_optimizer(MultiViewModel(kind),kind)
            self.assertEqual(type(opt),torch.optim.AdamW);self.assertEqual(opt.defaults['lr'],.001);self.assertEqual(opt.defaults['weight_decay'],.0001)

if __name__=='__main__':unittest.main()
