import copy
import unittest

import torch
from torch.nn import functional as F

from ta_wf_next.learnable_generator import GeneratorClassifier
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views


def example():
    rng = torch.Generator().manual_seed(91)
    directions = (torch.randint(0, 2, (3, 100), generator=rng) * 2 - 1).float()
    directions[0, 63:] = 0
    directions[1] = 0
    views = [generate_views(row.tolist(), input_kind="direction", budget=100, window_sizes=(50,))
             for row in directions]
    return batch_from_views(views, packet_patch=50, max_runs=12), directions, directions != 0


def make(pool="attention", trainable=True):
    return GeneratorClassifier(pool=pool, generator_trainable=trainable,
                               packet_budget=100, max_runs=12, window_tokens=2, dropout=0.1)


class LearnableGeneratorTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_paired_initial_state_and_tokens(self):
        batch, x, valid = example()
        outputs = []; states = []
        for pool, trainable in [("mean", True), ("attention", True), ("attention", False), ("attention", True)]:
            torch.manual_seed(1729)
            model = make(pool, trainable).eval()
            states.append(copy.deepcopy(model.state_dict()))
            outputs.append(model(batch, x, valid))
        for state, output in zip(states[1:], outputs[1:]):
            self.assertTrue(all(torch.equal(v, state[k]) for k, v in states[0].items()))
            self.assertTrue(torch.equal(outputs[0], output))

    def test_padding_empty_mask_and_batch_independence(self):
        batch, x, valid = example(); model = make().eval()
        tokens = model.generator(x, valid, batch.packet)
        altered = x.clone(); altered[~valid] = 999
        self.assertTrue(torch.equal(tokens, model.generator(altered, valid, batch.packet)))
        self.assertTrue(torch.equal(tokens[1], torch.zeros_like(tokens[1])))
        self.assertTrue(torch.isfinite(model(batch, altered, valid)).all())
        alone = model.generator(x[:1], valid[:1], batch.packet[:1])
        self.assertTrue(torch.allclose(tokens[:1], alone, atol=1e-6, rtol=1e-6))

    def test_feedback_and_whole_generator_freeze(self):
        batch, x, valid = example()
        for trainable in (True, False):
            torch.manual_seed(7); model = make(trainable=trainable)
            before = copy.deepcopy(model.generator.state_dict())
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            loss = F.cross_entropy(model(batch, x, valid), torch.tensor([0, 1, 2]))
            loss.backward()
            if trainable:
                for name in ("conv1.weight", "conv2.weight", "score.weight", "projection.weight"):
                    grad = dict(model.generator.named_parameters())[name].grad
                    self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0, name)
            else:
                self.assertTrue(all(p.grad is None for p in model.generator.parameters()))
            optimizer.step()
            changed = any(not torch.equal(v, model.generator.state_dict()[k]) for k, v in before.items())
            self.assertEqual(changed, trainable)

    def test_mean_scorer_is_unused(self):
        batch, x, valid = example(); model = make(pool="mean")
        model(batch, x, valid).sum().backward()
        self.assertFalse(model.generator.score.weight.requires_grad)
        self.assertIsNone(model.generator.score.weight.grad)


if __name__ == "__main__":
    unittest.main()
