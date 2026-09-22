import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "run_replication.py"
SPEC = importlib.util.spec_from_file_location("run_replication", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class ReplicationUnitTests(unittest.TestCase):
    def test_normalized_zero_safe(self):
        x=np.array([[3.0,4.0],[0.0,0.0]])
        got=MOD.normalized(x)
        np.testing.assert_allclose(got[0],[0.6,0.8])
        np.testing.assert_array_equal(got[1],[0.0,0.0])

    def test_metric_identity(self):
        y=np.tile(np.arange(MOD.CLASSES),2)
        m=MOD.metric(y,y,True)
        self.assertEqual(m["accuracy"],1.0)
        self.assertEqual(m["macro_f1"],1.0)
        self.assertEqual(len(m["per_website"]),MOD.CLASSES)

    def test_tiebreak_is_conservative(self):
        stats={c:{"macro_f1_mean":.5,"accuracy_mean":.5,"macro_f1_sd":.1} for c in MOD.GLOBAL}
        self.assertEqual(MOD.choose(stats,MOD.GLOBAL),"G_source")

    def test_preregistered_training_grid(self):
        self.assertEqual(MOD.TRAIN_SEEDS,(1013,2024))
        self.assertEqual(set(MOD.TRAIN_SEEDS)&{6238},set())
        self.assertEqual(len(MOD.MODELS)*len(MOD.TRAIN_SEEDS),4)


if __name__ == "__main__": unittest.main()
