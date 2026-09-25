import unittest
import torch

from ta_wf_next.batch_views import (coarse_run_batch, exact_run_batch,
    packet_batch, views_from_rows, window_batch)


class BatchViewTests(unittest.TestCase):
    def setUp(self):
        self.views = views_from_rows([[1, 1, -1, -1, 1, 0], [1, -1, 0, 0]],
                                     input_kind='direction', observation_budget=8,
                                     window_sizes=(4,))

    def test_packet_mask_and_observation_budget(self):
        b = packet_batch(self.views, width=4)
        self.assertEqual(tuple(b.values.shape), (2, 4, 1))
        self.assertEqual(b.mask.tolist(), [[True, True, True, True], [True, True, False, False]])
        self.assertEqual(b.token_count.tolist(), [5, 2])
        self.assertEqual(b.source_observed_count.tolist(), [5, 2])
        self.assertEqual(b.truncated.tolist(), [True, False])

    def test_exact_and_coarse_are_separate(self):
        exact = exact_run_batch(self.views, width=4)
        coarse = coarse_run_batch(self.views, width=4)
        self.assertEqual(tuple(exact.values.shape), (2, 4, 4))
        self.assertEqual(tuple(coarse.values.shape), (2, 4, 8))
        self.assertEqual(exact.values[0, 0, 1].item(), 2.)
        self.assertEqual(coarse.values[0, 0, 1].item(), 1.)

    def test_window_partial_is_explicit(self):
        w = window_batch(self.views, size=4, width=4)
        self.assertEqual(w.mask[0].tolist(), [True, True, False, False])
        self.assertEqual(w.values[0, 1, 3].item(), 1.)
        self.assertEqual(w.token_count.tolist(), [2, 1])
        self.assertEqual(w.source_observed_count.tolist(), [5, 2])

    def test_empty_and_bad_width_rejected(self):
        with self.assertRaises(ValueError):
            packet_batch(())
        with self.assertRaises(ValueError):
            exact_run_batch(self.views, width=0)


if __name__ == '__main__':
    unittest.main()
