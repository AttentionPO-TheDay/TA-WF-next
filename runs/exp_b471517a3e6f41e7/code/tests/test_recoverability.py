import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_recoverability.py"
SPEC = importlib.util.spec_from_file_location("recoverability", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecoverabilityTests(unittest.TestCase):
    def test_normalized_rows_and_zero(self):
        value = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        got = MODULE.normalized(value)
        np.testing.assert_allclose(got[0], [0.6, 0.8])
        np.testing.assert_array_equal(got[1], [0.0, 0.0])

    def test_prototype_predict(self):
        support_x = np.eye(MODULE.CLASSES, dtype=np.float32).repeat(2, axis=0)
        support_y = np.arange(MODULE.CLASSES).repeat(2)
        query = np.eye(MODULE.CLASSES, dtype=np.float32)[:3]
        np.testing.assert_array_equal(MODULE.prototype_predict(support_x, support_y, query), [0, 1, 2])

    def test_list_hash_is_order_sensitive(self):
        self.assertNotEqual(MODULE.int_list_hash([1, 2]), MODULE.int_list_hash([2, 1]))
        self.assertNotEqual(MODULE.string_list_hash(["a", "b"]), MODULE.string_list_hash(["b", "a"]))


if __name__ == "__main__":
    unittest.main()
