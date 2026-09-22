import unittest
import torch
from ta_wf_next.screening import (classification_metrics, classify_features, region_descriptors,
    masked_descriptors, valid_local_positions, classify_masked, LOCAL_SPECS)


class ScreeningTests(unittest.TestCase):
    def test_metrics_perfect(self):
        result=classification_metrics(__import__('numpy').arange(102),__import__('numpy').arange(102))
        self.assertEqual(result,{k:1.0 for k in ['accuracy','macro_precision','macro_recall','macro_f1']})

    def test_classifiers_do_not_accept_query_labels(self):
        rg=torch.nn.functional.normalize(torch.tensor([[1.,0.],[0.,1.]]),dim=1)
        rl=torch.nn.functional.normalize(torch.tensor([[[1.,0.]]*4,[[0.,1.]]*4]),dim=2)
        b,c,d,pairs=classify_features(rg,rl,rg,rl,torch.tensor([3,7]))
        self.assertEqual(b.tolist(),[3,7]); self.assertEqual(c.tolist(),[3,7]); self.assertEqual(d.tolist(),[3,7]); self.assertEqual(list(pairs.shape),[2,5])

    def test_region_shape(self):
        self.assertEqual(list(region_descriptors(torch.randn(2,256,18),'df').shape),[2,4,256])
        self.assertEqual(list(region_descriptors(torch.randn(2,512,157),'varcnn_direction').shape),[2,4,512])

    def test_exact_rf_mask_and_same_layer_pool(self):
        for name,spec in LOCAL_SPECS.items():
            x=torch.ones(2,1,5000); x[1,0,100:]=0
            local=torch.randn(2,spec['channels'],spec['positions'])
            mask=valid_local_positions(x,name)
            for i in range(spec['positions']):
                left,right=(4*i-8,4*i+13) if name=='df' else (4*i-17,4*i+17)
                self.assertEqual(bool(mask[0,i]),left>=0 and right<5000)
                self.assertEqual(bool(mask[1,i]),left>=0 and right<100)
            regions,pooled,ok,counts=masked_descriptors(local,x,name)
            self.assertTrue(ok.all()); self.assertTrue((counts>=4).all())
            expected=torch.nn.functional.normalize(local[1,:,mask[1]].mean(1),dim=0)
            torch.testing.assert_close(pooled[1],expected)
            changed=local.clone()
            changed[1,:,~mask[1]]=99999
            r2,p2,_,_=masked_descriptors(changed,x,name)
            torch.testing.assert_close(regions,r2); torch.testing.assert_close(pooled,p2)

    def test_zero_holes_and_insufficient_positions(self):
        for name,spec in LOCAL_SPECS.items():
            x=torch.ones(3,1,5000); x[0]=0; x[1,0,20:]=0; x[2,0,50]=0
            mask=valid_local_positions(x,name)
            for i in range(spec['positions']):
                left,right=(4*i-8,4*i+13) if name=='df' else (4*i-17,4*i+17)
                if left<=50<=right: self.assertFalse(mask[2,i])
            r,p,ok,_=masked_descriptors(torch.randn(3,spec['channels'],spec['positions']),x,name)
            self.assertEqual(ok[:2].tolist(),[False,False])
            self.assertTrue(torch.isfinite(r).all()); self.assertTrue(torch.isfinite(p).all())

    def test_same_layer_classifier_and_fallback(self):
        g=torch.eye(2); local=g[:,None,:].expand(-1,4,-1); labels=torch.tensor([3,7])
        ok=torch.tensor([True,True])
        b,c,d,e,pairs,used=classify_masked(g,local,g,ok,g,local,g,ok,labels)
        self.assertEqual(e.tolist(),[3,7]); self.assertTrue(used.all())
        # An ineligible query retains the full-reference C prediction.
        _,c,d,e,pairs,used=classify_masked(g,local,g,torch.tensor([True,False]),g,local,g,ok,labels)
        self.assertEqual(d[1],c[1]); self.assertEqual(e[1],c[1]); self.assertTrue((pairs[1]==-1).all())
        # Missing reference class disables regional classification, not class coverage.
        _,c,d,e,pairs,used=classify_masked(g,local,g,ok,g,local,g,torch.tensor([True,False]),labels)
        torch.testing.assert_close(d,c); torch.testing.assert_close(e,c); self.assertFalse(used.any())

if __name__=='__main__': unittest.main()
