import math
import unittest

import torch

from keras_compat import KerasAdam, KerasCallbacks


class KerasCompatibilityTests(unittest.TestCase):
    def test_adam_two_steps_and_resume(self):
        p = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
        opt = KerasAdam([p])
        expected = [1.0, -2.0]
        m, v = [0.0, 0.0], [0.0, 0.0]
        for t, grads in enumerate(([1e-8, -0.4], [2e-8, 0.2]), 1):
            for i, g in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                expected[i] -= .001 * math.sqrt(1 - .999**t) / (1 - .9**t) * m[i] / (math.sqrt(v[i]) + 1e-8)
            p.grad = torch.tensor(grads, dtype=p.dtype)
            opt.step()
            torch.testing.assert_close(p, torch.tensor(expected, dtype=p.dtype), rtol=1e-12, atol=1e-12)
        clone = KerasAdam([p])
        clone.load_state_dict(opt.state_dict())
        self.assertEqual(clone.param_groups[0]["iterations"], 2)

    def test_callback_ties_order_and_resume(self):
        opt = KerasAdam([torch.tensor(0., requires_grad=True)])
        callbacks = KerasCallbacks()
        self.assertTrue(callbacks.step(.5, opt)["improved"])
        for tie in range(1, 12):
            result = callbacks.step(.5, opt)
            self.assertFalse(result["improved"])
            self.assertEqual(result["reduced_lr"], tie in (6, 11))
            self.assertEqual(result["stop"], tie == 11)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], .0001)
        self.assertEqual(callbacks.plateau_wait, 1)
        self.assertEqual(callbacks.early_wait, 11)
        restored = KerasCallbacks()
        restored.load_state_dict(callbacks.state_dict())
        self.assertEqual(restored.state_dict(), callbacks.state_dict())

    def test_callback_independent_thresholds_and_floor(self):
        opt = KerasAdam([torch.tensor(0., requires_grad=True)], lr=1e-5)
        callbacks = KerasCallbacks()
        callbacks.step(.5, opt)
        result = callbacks.step(.50005, opt)
        self.assertTrue(result["improved"])
        self.assertEqual(callbacks.plateau_wait, 1)
        self.assertEqual(callbacks.early_wait, 0)
        for _ in range(10):
            result = callbacks.step(.50005, opt)
            self.assertFalse(result["reduced_lr"])
        self.assertEqual(callbacks.plateau_wait, 11)


if __name__ == "__main__":
    unittest.main()
