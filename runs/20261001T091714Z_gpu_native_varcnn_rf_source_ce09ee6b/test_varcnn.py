"""Synthetic CPU port checks only: no real data and no optimizer training."""
import unittest
import torch
from varcnn_native import CausalConv1d, SameMaxPool1d, VarCNNNative


class VarCNNPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_causal_manual_stride_dilation(self):
        for length in (8, 9):
            for stride, dilation in ((1, 1), (2, 1), (1, 2), (1, 4), (1, 8)):
                conv = CausalConv1d(1, 1, 3, stride=stride, dilation=dilation, bias=False)
                with torch.no_grad():
                    conv.weight.copy_(torch.tensor([[[2., 3., 5.]]]))
                x = torch.arange(1., length + 1).reshape(1, 1, -1)
                expected = []
                for endpoint in range(0, length, stride):
                    expected.append(sum(w * float(x[0, 0, endpoint - (2 - j) * dilation])
                                        for j, w in enumerate((2., 3., 5.))
                                        if endpoint - (2 - j) * dilation >= 0))
                torch.testing.assert_close(conv(x).flatten(), torch.tensor(expected))

    def test_same_pool_independent_windows(self):
        for values, expected in (([-8., -7., -6., -5.], [-6., -5.]),
                                 ([-8., -7., -6., -5., -4.], [-7., -5., -4.])):
            result = SameMaxPool1d()(torch.tensor(values).reshape(1, 1, -1))
            torch.testing.assert_close(result.flatten(), torch.tensor(expected))

    def test_architecture_gradients_eval_and_reload(self):
        model = VarCNNNative()
        self.assertIsInstance(model.stages[0][0].shortcut, torch.nn.Sequential)
        self.assertIsInstance(model.stages[0][1].shortcut, torch.nn.Identity)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                self.assertEqual(module.eps, 1e-5)
                self.assertEqual(module.momentum, .01)
        x = torch.randn(2, 1, 5000)
        shapes = []
        handles = [stage.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
                   for stage in model.stages]
        y = model(x)
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes, [(2, 64, 1250), (2, 128, 625), (2, 256, 313), (2, 512, 157)])
        self.assertEqual(tuple(y.shape), (2, 102))
        torch.nn.functional.cross_entropy(y, torch.tensor([0, 101])).backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        model.eval()
        with torch.no_grad():
            a, b = model(x), model(x)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            restored = VarCNNNative().eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(a, restored(x), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 5000))
        print('VarCNNNative parameters:', sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
