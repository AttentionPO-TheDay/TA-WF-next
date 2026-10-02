"""Synthetic CPU checks only. Run directly; reads no dataset/checkpoint."""
import unittest
import torch
from model import MultiViewModel, MultiViewGenerator, parameter_counts

torch.set_num_threads(2)


def example():
    x = torch.zeros(2, 5000)
    x[0, :73] = torch.arange(1, 74).float() / 7
    x[0, 1:73:2] *= -1
    # Non-monotonic timestamps must remain in original packet order.
    x[0, 3] = -0.01
    x[1, :113] = torch.arange(1, 114).float() / 3
    tam = torch.zeros(2, 2, 1800)
    tam[0, 0, 3:11] = 2
    tam[0, 1, 27:41] = 1
    tam[1, :, :117] = 3
    return x, tam


class ModelChecks(unittest.TestCase):
    def test_fixed_scalar_reference(self):
        x, tam = example()
        x_orig, tam_orig = x.clone(), tam.clone()
        g = MultiViewGenerator("fixed")
        result = g(x, tam)
        p = result["packet"]
        for row in range(2):
            for token in (0, 1, 2, 99):
                for sub in range(5):
                    v = x[row, token * 50 + sub * 10:token * 50 + sub * 10 + 10]
                    vals = [float(a) for a in v if a != 0]
                    pairs = [(float(a), float(b)) for a, b in zip(v[:-1], v[1:]) if a != 0 and b != 0]
                    expected = [sum(1 if a > 0 else -1 for a in vals) / max(len(vals), 1),
                                len(vals) / 10,
                                sum(min(abs(a), 80) / 80 for a in vals) / max(len(vals), 1),
                                sum((a > 0) != (b > 0) for a, b in pairs) / max(len(pairs), 1)]
                    torch.testing.assert_close(p[row, token, sub * 4:sub * 4 + 4], torch.tensor(expected))
        torch.testing.assert_close(result["time"][0, 0, :30], tam[0, :, :15].log1p().flatten())
        self.assertEqual(result["packet_mask"].sum().item(), 5)
        self.assertTrue(result["time_mask"].all().item())
        self.assertFalse(p[..., 20:].any().item())
        self.assertFalse(result["time"][..., 30:].any().item())
        self.assertTrue(torch.equal(x, x_orig))
        self.assertTrue(torch.equal(tam, tam_orig))
        self.assertTrue(all(not a.requires_grad for a in g.parameters()))

    def test_shared_initialization_and_fixed_statistics(self):
        models = []
        for mode in ("fixed", "learned"):
            for head in ("mlp", "transformer"):
                torch.manual_seed(2026)
                models.append(MultiViewModel(mode, head))
        reference = models[0].state_dict()
        for m in models[1:]:
            for key, value in m.state_dict().items():
                if not key.startswith("head."):
                    self.assertTrue(torch.equal(value, reference[key]), key)
        x, tam = example()
        fixed, learned = models[0].generator(x, tam), models[2].generator(x, tam)
        torch.testing.assert_close(fixed["packet"][..., :20], learned["packet"][..., :20])
        torch.testing.assert_close(fixed["time"][..., :30], learned["time"][..., :30])
        for key in ("packet_mask", "time_mask"):
            self.assertTrue(torch.equal(fixed[key], learned[key]))

    def test_all_padding_eval_repeat_and_batch_independence(self):
        x, tam = example()
        for mode in ("fixed", "learned"):
            for head in ("mlp", "transformer"):
                torch.manual_seed(1729)
                m = MultiViewModel(mode, head).eval()
                with torch.no_grad():
                    batch = m(x, tam)
                    repeat = m(x, tam)
                    single = m(x[:1], tam[:1])
                    empty = m(torch.zeros_like(x), torch.zeros_like(tam))
                    p, t, p_mask, t_mask = m.forward_tokens(x, tam)
                self.assertTrue(torch.isfinite(empty).all().item())
                self.assertEqual(tuple(batch.shape), (2, 102))
                torch.testing.assert_close(batch, repeat, atol=0, rtol=0)
                torch.testing.assert_close(batch[:1], single, atol=3e-6, rtol=3e-5)
                self.assertFalse(p[~p_mask].any().item())
                self.assertTrue(t_mask.all().item())

    def test_learning_reaches_both_generator_branches(self):
        x, tam = example()
        for head in ("mlp", "transformer"):
            torch.manual_seed(3407)
            m = MultiViewModel("learned", head)
            before = {name: p.detach().clone() for name, p in m.generator.named_parameters()}
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            logits = m(x, tam)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([4, 8]))
            loss.backward()
            for name, p in m.generator.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all().item(), name)
                self.assertGreater(p.grad.abs().sum().item(), 0, name)
            opt.step()
            for name, p in m.generator.named_parameters():
                self.assertFalse(torch.equal(p, before[name]), name)

    def test_input_contract(self):
        m = MultiViewModel("fixed", "mlp")
        with self.assertRaises(ValueError):
            m(torch.zeros(2, 4999), torch.zeros(2, 2, 1800))
        with self.assertRaises(ValueError):
            m(torch.zeros(2, 5000), torch.zeros(2, 1800, 2))


if __name__ == "__main__":
    for mode in ("fixed", "learned"):
        for head in ("mlp", "transformer"):
            torch.manual_seed(1729)
            print(mode, head, parameter_counts(MultiViewModel(mode, head)))
    unittest.main()
