"""CPU-only synthetic checks; no datasets, checkpoints, or optimizer steps."""

import importlib.util
import os
from pathlib import Path
import unittest

import torch

from ta_wf_next.models import DF, VarCNN, VarCNNDirection


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_direction_interfaces_and_gradients(self):
        for model_type, channels, positions in (
            (DF, 256, 18), (VarCNNDirection, 512, 157),
        ):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(7)
                model = model_type(5).eval()
                x = torch.randn(2, 1, 5000)
                with torch.no_grad():
                    logits, features = model(x)
                    local = model.forward_local(x)
                    self.assertEqual(logits.shape, (2, 5))
                    self.assertEqual(features.shape, (2, 512))
                    self.assertEqual(local.shape, (2, channels, positions))
                    torch.testing.assert_close(features, model.forward_features(x))
                    torch.testing.assert_close(logits, model.mlp(features))
                model.train()
                logits, _ = model(x)
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_direction_variable_length_and_channel_guard(self):
        model = VarCNNDirection(5).eval()
        with torch.no_grad():
            for length in (3000, 4097):
                self.assertEqual(model(torch.randn(1, 1, length))[0].shape, (1, 5))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 2, 5000))

    def test_dual_branch(self):
        model = VarCNN(5).eval()
        with torch.no_grad():
            logits, features = model(torch.randn(2, 2, 5000))
        self.assertEqual(logits.shape, (2, 5))
        self.assertEqual(features.shape, (2, 1024))

    @unittest.skipUnless(os.environ.get("WF_MIGRATION_SOURCE"),
                         "optional explicit legacy-source parity check")
    def test_migration_parity(self):
        # Only this opt-in migration check loads standalone legacy source files.
        # Production imports and normal tests never depend on the old workspace.
        source = Path(os.environ["WF_MIGRATION_SOURCE"])
        for name, model_type, channels in (("DF", DF, 1), ("VarCNN", VarCNN, 2)):
            with self.subTest(model=name):
                spec = importlib.util.spec_from_file_location(
                    "migration_reference_" + name, source / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                torch.manual_seed(11)
                reference = getattr(module, name)(5).eval()
                migrated = model_type(5).eval()
                migrated.load_state_dict(reference.state_dict(), strict=True)
                x = torch.randn(2, channels, 5000)
                with torch.no_grad():
                    expected = reference(x)
                    actual = migrated(x)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    if name == "VarCNN":
                        direction = VarCNNDirection(5).eval()
                        direction.dir_encoder.load_state_dict(
                            reference.dir_encoder.state_dict(), strict=True)
                        torch.testing.assert_close(
                            direction.dir_encoder(x[:, :1]),
                            reference.dir_encoder(x[:, :1]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
