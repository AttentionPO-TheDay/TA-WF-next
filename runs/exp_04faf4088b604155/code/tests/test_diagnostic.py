import importlib.util
import unittest
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/"run_diagnostic.py"
S=importlib.util.spec_from_file_location("diagnostic",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)

class DiagnosticTests(unittest.TestCase):
    def test_combinations(self):
        self.assertEqual(M.COMBOS.shape,(120,3)); np.testing.assert_array_equal(M.COMBOS[0],[0,1,2])
        self.assertEqual(len({tuple(x) for x in M.COMBOS}),120)
        self.assertTrue(all(sum(np.all(M.COMBOS==c,axis=1))==1 for c in M.COMBOS))
    def test_normalized(self):
        x=M.normalized(np.array([[3.,4.],[0.,0.]],np.float32)); np.testing.assert_allclose(x[0],[.6,.8]); np.testing.assert_array_equal(x[1],[0,0])
    def test_subset_predict(self):
        support=np.zeros((M.CLASSES,10,512),np.float32)
        for c in range(M.CLASSES): support[c,:,c]=1
        q=np.zeros((3,512),np.float32); q[0,0]=1;q[1,1]=1;q[2,2]=1
        pred,ten=M.predict_all_subsets(q,support)
        np.testing.assert_array_equal(pred[:,0],0); np.testing.assert_array_equal(pred[:,1],1); np.testing.assert_array_equal(ten,[0,1,2])

if __name__=="__main__": unittest.main()
