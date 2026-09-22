import unittest
from ta_wf_next.traffic_views import generate_views


class TrafficViewTests(unittest.TestCase):
    def test_views_and_windows(self):
        v=generate_views([1,1,-1,-1,1,0],input_kind='direction',window_sizes=(2,4))
        self.assertEqual(v.packet_direction,(1,1,-1,-1,1))
        self.assertEqual(v.runs.decode(),v.packet_direction)
        w=dict(v.direction_windows)[4]
        self.assertEqual(w[0].positive_fraction,.5)
        self.assertAlmostEqual(w[0].transition_fraction,1/3)
        self.assertTrue(w[1].partial)
        self.assertEqual(w[1].transition_fraction,0.)

    def test_cross_sign_reversal_not_clipped_iat(self):
        v=generate_views([2.,-1.,3.,-2.],input_kind='signed_timestamp',enable_timing=True)
        self.assertEqual(v.timing.same_direction_interval,(None,None,1.,1.))
        self.assertEqual(v.timing.run_span,(0.,0.,0.,0.))
        self.assertEqual(v.packet_direction,(1,-1,1,-1))
        self.assertEqual(v.timing.invalid_same_direction_intervals,0)

    def test_bad_direction_time_mask(self):
        v=generate_views([3.,2.,4.,-1.,-2.],input_kind='signed_timestamp',enable_timing=True)
        self.assertEqual(v.timing.same_direction_interval,(None,None,2.,None,1.))
        self.assertEqual(v.timing.run_span,(None,1.))
        self.assertEqual(v.timing.invalid_run_spans,1)
        self.assertEqual(v.timing.invalid_same_direction_intervals,1)

    def test_missing_not_zero(self):
        for kind in ['direction','signed_timestamp']:
            v=generate_views([1,-1],input_kind=kind)
            self.assertIsNone(v.timing); self.assertIsNone(v.size)
            with self.assertRaises(ValueError): v.select('timing')

    def test_sizes_not_timestamps(self):
        v=generate_views([100,200,-50,0],input_kind='signed_size')
        self.assertEqual(v.size.absolute_sizes,(100,200,50))
        self.assertEqual(v.size.run_totals,(300,50))
        self.assertIsNone(v.timing)
        with self.assertRaises(ValueError): generate_views([1],input_kind='signed_size',enable_timing=True)

    def test_export_no_bypass(self):
        v=generate_views([1]*4+[-1]*6,input_kind='direction')
        self.assertEqual(set(v.select('coarse_runs')),{'coarse_runs'})
        for names in [(),('unknown',),('exact_runs','exact_runs')]:
            with self.assertRaises(ValueError): v.select(*names)

    def test_no_suffix_read(self):
        def stream():
            yield 1.; yield -2.
            raise AssertionError('outside budget')
        v=generate_views(stream(),input_kind='signed_timestamp',budget=2,enable_timing=True)
        self.assertEqual(v.packet_direction,(1,-1))
        self.assertEqual(v.runs.end_reason,'budget')

    def test_guards(self):
        for values in [[1,0,1],[float('nan')],[float('inf')]]:
            with self.assertRaises(ValueError): generate_views(values,input_kind='signed_timestamp')
        for windows in [(0,),(-1,),(2,2),(True,),(1.5,)]:
            with self.assertRaises(ValueError): generate_views([1],input_kind='direction',window_sizes=windows)
        for b in [0,True,1.5]:
            with self.assertRaises(ValueError): generate_views([1],input_kind='direction',budget=b)

    def test_empty_and_zero_duration(self):
        v=generate_views([0,0],input_kind='signed_timestamp',enable_timing=True)
        self.assertEqual(v.packet_direction,()); self.assertEqual(v.timing.run_span,())
        v=generate_views([1.,1.,-2.],input_kind='signed_timestamp',enable_timing=True)
        self.assertEqual(v.timing.run_zero_span,(True,True))
        self.assertEqual(v.timing.same_direction_interval,(None,0.,None))

    def test_same_structure_across_modalities(self):
        a=generate_views([1,1,-1],input_kind='direction')
        for kind in ['signed_size','signed_timestamp']:
            b=generate_views([2,4,-3],input_kind=kind)
            self.assertEqual(a.runs,b.runs)
            self.assertEqual(a.direction_windows,b.direction_windows)

if __name__=='__main__': unittest.main()
