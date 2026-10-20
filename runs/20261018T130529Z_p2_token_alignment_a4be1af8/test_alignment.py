import unittest
import torch
from token_alignment import AdaINToken, Moments, ResidualTokenProjector, moment_alignment_loss


class AlignmentChecks(unittest.TestCase):
    def test_reference_affine_recovery_and_query_batch_independence(self):
        torch.manual_seed(42)
        clean = torch.randn(16, 120, 110)
        shifted = clean * 1.5 + 0.7
        source, reference = Moments(), Moments()
        source.add(clean)
        reference.add(shifted)
        layer = AdaINToken(source.finish(), reference.finish())
        self.assertTrue(torch.allclose(layer(shifted), clean, atol=2e-6))
        self.assertTrue(torch.equal(layer(shifted[:1]), layer(shifted)[:1]))

    def test_residual_identity_and_unsupervised_gradient(self):
        torch.manual_seed(42)
        clean = torch.randn(8, 120, 110)
        source = Moments()
        source.add(clean)
        reference = clean * 1.2 + 0.5
        layer = ResidualTokenProjector()
        self.assertTrue(torch.equal(layer(reference), reference))
        opt = torch.optim.Adam(layer.parameters(), lr=0.01)
        initial = float(moment_alignment_loss(reference, layer(reference), source.finish()).detach())
        for _ in range(20):
            opt.zero_grad()
            loss = moment_alignment_loss(reference, layer(reference), source.finish())
            loss.backward()
            opt.step()
        final = float(moment_alignment_loss(reference, layer(reference), source.finish()).detach())
        self.assertLess(final, initial)

    def test_streaming_moments_and_empty_rejection(self):
        torch.manual_seed(42)
        x = torch.randn(9, 120, 110)
        full, parts = Moments(), Moments()
        full.add(x)
        parts.add(x[:4])
        parts.add(x[4:])
        self.assertTrue(torch.equal(full.finish()['mean'], parts.finish()['mean']))
        self.assertTrue(torch.equal(full.finish()['variance'], parts.finish()['variance']))
        with self.assertRaises(ValueError):
            Moments().finish()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
