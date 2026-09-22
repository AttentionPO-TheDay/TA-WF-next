import itertools
import unittest
from ta_wf_next.burst_tokens import encode_directions


class BurstTokenTests(unittest.TestCase):
    def test_roundtrip_exhaustive(self):
        for n in range(10):
            for seq in itertools.product((-1,1), repeat=n):
                e=encode_directions(seq, budget=12)
                self.assertEqual(e.decode(), seq)
                self.assertEqual(sum(r.count for r in e.runs), n)
                self.assertTrue(all(a.direction != b.direction for a,b in zip(e.runs,e.runs[1:])))

    def test_padding(self):
        e=encode_directions([1,1,-1,0,0]); self.assertEqual(e.decode(),(1,1,-1))
        self.assertEqual(e.end_reason,'padding')

    def test_internal_zero(self):
        for seq in ([1,0,-1],[0,1]):
            with self.assertRaises(ValueError): encode_directions(seq)

    def test_nonfinite(self):
        for v in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ValueError): encode_directions([v])

    def test_budget_validation(self):
        for b in (0,-1,True,1.5):
            with self.assertRaises(ValueError): encode_directions([1],budget=b)

    def test_modes(self):
        with self.assertRaises(ValueError): encode_directions([2])
        with self.assertRaises(ValueError): encode_directions([1],input_kind='size')

    def test_timestamp_order_preserved(self):
        self.assertEqual(encode_directions([1.1,-1.0,-1.2,.9],input_kind='signed_timestamp').decode(),(1,-1,-1,1))

    def test_no_lookahead(self):
        def stream():
            yield 1; yield -1
            raise AssertionError('read outside budget')
        e=encode_directions(stream(),budget=2)
        self.assertEqual(e.end_reason,'budget')
        self.assertEqual(e.decode(),(1,-1))

    def test_boundary(self):
        e=encode_directions([1]*10,budget=5)
        self.assertEqual(e.runs[0].count,5)
        self.assertTrue(e.runs[0].touches_left_boundary and e.runs[0].touches_right_boundary)
        e=encode_directions([1,-1,1,0])
        self.assertFalse(e.runs[1].touches_left_boundary or e.runs[1].touches_right_boundary)
        self.assertTrue(e.runs[-1].touches_right_boundary)

    def test_empty(self):
        for seq in ([],[0,0]):
            e=encode_directions(seq); self.assertEqual(e.runs,()); self.assertEqual(e.coarse(),())

    def test_bins_and_lossy_collision(self):
        a=encode_directions([1]*4+[-1]*8)
        b=encode_directions([1]*7+[-1]*15)
        self.assertEqual(a.coarse(),b.coarse())
        self.assertNotEqual(a.decode(),b.decode())
        self.assertEqual([t.log2_count_bin for t in a.coarse()],[2,3])
        self.assertIsNone(a.coarse()[0].previous_bin_delta)
        self.assertEqual(a.coarse()[0].next_bin_delta,1)
        self.assertEqual(a.coarse()[1].previous_bin_delta,1)

    def test_bin_boundaries(self):
        self.assertEqual([encode_directions([1]*n).coarse()[0].log2_count_bin for n in [1,2,3,4,7,8]],[0,1,1,2,2,3])

if __name__ == '__main__': unittest.main()
