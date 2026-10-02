"""CPU synthetic invariants; no real data or training runs."""
import unittest
import torch
from model import PacketModel


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        torch.manual_seed(9)
        x = torch.randint(0, 2, (3, 5000)).float() * 2 - 1
        x[0, 731:] = 0
        x[1, 4011:] = 0
        x[2] = 0
        return x

    def test_shapes_mask_and_batch_independence(self):
        x = self.inputs()
        for head in ('transformer', 'mlp'):
            with self.subTest(head=head), torch.no_grad():
                model = PacketModel(head=head).eval()
                logits = model(x)
                self.assertEqual(logits.shape, (3, 102))
                self.assertTrue(torch.isfinite(logits).all())
                tokens, mask = model.generator(x, x != 0)
                self.assertEqual(tokens.shape, (3, 79, 128))
                self.assertEqual(mask.sum(1).tolist(), [12, 63, 0])
                self.assertEqual(tokens[~mask].abs().sum().item(), 0)
                garbage = x.clone()
                garbage[x == 0] = float('nan')
                torch.testing.assert_close(logits, model(garbage, observed=x != 0), rtol=0, atol=0)
                for i in range(3):
                    torch.testing.assert_close(logits[i:i+1], model(x[i:i+1]), rtol=1e-5, atol=1e-6)

    def test_generator_gradients(self):
        x = self.inputs()
        for head in ('transformer', 'mlp'):
            with self.subTest(head=head):
                model = PacketModel(head=head)
                torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2])).backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    if 'convs' in name and name.endswith('weight'):
                        self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_shared_initialization(self):
        torch.manual_seed(1729)
        attention = PacketModel('transformer')
        after_attention = torch.rand(3)
        torch.manual_seed(1729)
        mlp = PacketModel('mlp')
        after_mlp = torch.rand(3)
        torch.testing.assert_close(after_attention, after_mlp, rtol=0, atol=0)
        common = [name for name in attention.state_dict() if not name.startswith('blocks.')]
        for name in common:
            torch.testing.assert_close(attention.state_dict()[name], mlp.state_dict()[name], rtol=0, atol=0)

    def test_all_padding_backward(self):
        for head in ('transformer', 'mlp'):
            model = PacketModel(head)
            model(torch.zeros(2, 5000)).square().mean().backward()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__':
    unittest.main()
