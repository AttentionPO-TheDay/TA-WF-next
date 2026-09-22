import importlib.util
from pathlib import Path
import unittest
import numpy as np

P = Path(__file__).resolve().parents[1] / "run_region_relevance.py"
S = importlib.util.spec_from_file_location("region", P)
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)


class RegionTests(unittest.TestCase):
    def test_conflict_rule(self):
        exemplars = np.zeros((102, 3, 512), np.float32)
        support = np.zeros_like(exemplars)
        for c in range(102):
            exemplars[c, :, c % 512] = 1
            support[c, :, c % 512] = 1
        thresholds = np.full(102, .1, np.float32)
        w, own, alien, alien_class = M.conflict_weights(exemplars, thresholds, support)
        self.assertEqual(w.shape, (102, 3))
        self.assertTrue(np.all(w == 1))
        self.assertTrue(np.all(own == 1))
        self.assertEqual(alien_class.shape, (102, 3))

    def test_method_shapes_and_no_conflict_equivalence(self):
        rng = np.random.default_rng(4)
        e = rng.normal(size=(102, 3, 512)).astype(np.float32)
        e /= np.linalg.norm(e, axis=2, keepdims=True)
        p = rng.normal(size=(102, 512)).astype(np.float32)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        q = rng.normal(size=(7, 512)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        out = M.predict_methods(q, e, p, np.ones((102, 3), np.float32))
        self.assertTrue(all(v.shape == (7,) for v in out.values()))
        self.assertTrue(np.array_equal(out["B_plain_multiprototype"], out["C_site_global_weight"]))
        self.assertTrue(np.array_equal(out["B_plain_multiprototype"], out["D_region_conflict_suppression"]))
        self.assertTrue(np.array_equal(out["B_plain_multiprototype"], out["D_ablation_uniform_site_retention"]))


if __name__ == "__main__": unittest.main()
