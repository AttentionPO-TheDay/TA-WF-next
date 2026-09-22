import importlib.util
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "run_support_selection.py"
SPEC = importlib.util.spec_from_file_location("support_selection", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SupportSelectionTests(unittest.TestCase):
    def test_metric_perfect(self):
        truth = np.repeat(np.arange(M.CLASSES), 2)
        got = M.metric(truth, truth)
        self.assertEqual(got["accuracy"], 1.0)
        self.assertEqual(got["macro_f1"], 1.0)

    def test_conservative_exact_tie(self):
        stats = {c: {"valid": True, "macro_f1_mean": .5, "accuracy_mean": .5, "macro_f1_sd": .1}
                 for c in M.PROTO}
        self.assertEqual(M.choose(stats, M.PROTO), "G_source")

    def test_shrink_gradient(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(7, 3)); y = rng.integers(0, M.CLASSES, size=7)
        w0 = rng.normal(scale=.01, size=(M.CLASSES, 3)); b0 = rng.normal(scale=.01, size=M.CLASSES)
        theta = np.concatenate([w0.ravel(), b0]) + rng.normal(scale=.01, size=w0.size+b0.size)
        value, grad = M.shrink_objective(theta, x, y, w0, b0, .1)
        for index in (0, 13, len(theta)-1):
            eps = 1e-6; plus = theta.copy(); minus = theta.copy()
            plus[index] += eps; minus[index] -= eps
            vp = M.shrink_objective(plus, x, y, w0, b0, .1)[0]
            vm = M.shrink_objective(minus, x, y, w0, b0, .1)[0]
            self.assertAlmostEqual(grad[index], (vp-vm)/(2*eps), places=5)
        self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
