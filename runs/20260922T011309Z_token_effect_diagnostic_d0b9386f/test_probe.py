import unittest
import numpy as np
from probe import features,fit,predict_scores,metric

class ProbeTests(unittest.TestCase):
    def test_dimensions_and_zero_masks(self):
        f=features(np.array([1.,1.,-2.,0.]))
        self.assertEqual({k:len(v) for k,v in f.items()},dict(packet=5000,exact_run=5000,coarse_run=5000,windows=240,timing=20000))
        self.assertTrue(all(np.isfinite(v).all() for v in f.values()))
        t=f['timing'].reshape(2,5000,2)
        self.assertEqual(t[0,0,1],0); self.assertEqual(t[0,1,1],1)
        self.assertEqual(t[0,1,0],0)
    def test_coarse_no_exact_length_bypass(self):
        a=features(np.array([1.]*4+[-1.]*8))
        b=features(np.array([1.]*7+[-1.]*15))
        np.testing.assert_array_equal(a['coarse_run'],b['coarse_run'])
        self.assertFalse(np.array_equal(a['exact_run'],b['exact_run']))
    def test_fit_metrics(self):
        x=np.eye(102,dtype=np.float32); y=np.arange(102)
        p=predict_scores(x,fit(x,y)).argmax(1)
        self.assertEqual(metric(y,p)['accuracy'],1.)
        self.assertEqual(metric(y,p)['macro_f1'],1.)
        self.assertEqual(metric(y,(y+1)%102)['macro_f1'],0.)

if __name__=='__main__': unittest.main()
