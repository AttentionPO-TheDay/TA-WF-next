import importlib.util
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "run_history_retention.py"
SPEC = importlib.util.spec_from_file_location("history_retention", PATH)
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class HistoryRetentionTests(unittest.TestCase):
    def test_normalized_zero(self):
        got=M.normalized(np.asarray([[3.,4.],[0.,0.]])); np.testing.assert_allclose(got[0],[.6,.8]); np.testing.assert_array_equal(got[1],[0,0])

    def test_shrink_gradient(self):
        rng=np.random.default_rng(1); x=rng.normal(size=(7,4)); y=rng.integers(0,M.CLASSES,size=7)
        w0=rng.normal(size=(M.CLASSES,4)); b0=rng.normal(size=M.CLASSES); theta=np.concatenate([w0.ravel(),b0])+.01
        value,grad=M.shrink_objective(theta,x,y,w0,b0,.1)
        for idx in (0,17,len(theta)-1):
            step=np.zeros_like(theta); step[idx]=1e-6
            plus=M.shrink_objective(theta+step,x,y,w0,b0,.1)[0]; minus=M.shrink_objective(theta-step,x,y,w0,b0,.1)[0]
            self.assertAlmostEqual(grad[idx],(plus-minus)/2e-6,places=5)
        self.assertTrue(np.isfinite(value))

    def test_list_hash_order_sensitive(self):
        self.assertNotEqual(M.list_hash([1,2]),M.list_hash([2,1]))


if __name__ == "__main__": unittest.main()
