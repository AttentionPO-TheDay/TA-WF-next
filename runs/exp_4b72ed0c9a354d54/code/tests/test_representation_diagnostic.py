import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "run_representation_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("representation_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RepresentationDiagnosticTests(unittest.TestCase):
    def test_exact_layer_masks(self):
        full = torch.ones(2, 1, 5000)
        full[1, 0, 400:] = 0
        for model_specs in MODULE.LAYERS.values():
            for spec in model_specs.values():
                mask = MODULE.layer_mask(full, spec)
                for i in range(spec["positions"]):
                    left = spec["left0"] + spec["stride"] * i
                    right = spec["right0"] + spec["stride"] * i
                    self.assertEqual(bool(mask[0, i]), left >= 0 and right < 5000)
                    self.assertEqual(bool(mask[1, i]), left >= 0 and right < 400)

    def test_ordered_pool_and_short_samples_are_retained_as_zero(self):
        spec = {"channels": 1, "positions": 8, "stride": 1, "left0": 0, "right0": 0, "rf": 1}
        local = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
        inputs = torch.ones(2, 1, 5000)
        inputs[1, 0, 3:] = 0
        global_mean, ordered, regions, counts = MODULE.pool_representations(local, inputs, spec)
        self.assertEqual(counts.tolist(), [8, 3])
        torch.testing.assert_close(global_mean[0], torch.ones(1))
        raw = torch.tensor([0.5, 2.5, 4.5, 6.5])
        torch.testing.assert_close(ordered[0], raw / torch.linalg.vector_norm(raw))
        self.assertTrue((ordered[1] == 0).all())
        self.assertTrue((regions[1] == 0).all())
        self.assertTrue(torch.isfinite(global_mean).all() and torch.isfinite(ordered).all() and torch.isfinite(regions).all())

    def test_no_training_primitive_in_entrypoint(self):
        text = SCRIPT.read_text()
        self.assertNotIn(".backward(", text)
        self.assertNotIn("torch.optim", text)


if __name__ == "__main__":
    unittest.main()
