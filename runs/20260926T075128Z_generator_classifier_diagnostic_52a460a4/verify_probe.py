"""Saved-artifact replay and independent ridge stationarity/metrics checks."""
from pathlib import Path
import numpy as np


def independent_metrics(y,p):
    f1=[]
    for label in range(102):
        tp=np.count_nonzero((y==label)&(p==label));fp=np.count_nonzero((y!=label)&(p==label));fn=np.count_nonzero((y==label)&(p!=label))
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return {'accuracy':float(np.mean(y==p)),'macro_f1':float(np.mean(f1))}


def verify_saved(out,true,rows,config):
    checked=0;normal_residuals=[]
    with np.load(out/'features.npz',allow_pickle=False) as features:
        for kind in config['probe_kinds']:
            xs=features['source_'+kind].astype(np.float64);xv=features['valid_'+kind].astype(np.float64)
            expected_mean=np.mean(xs,axis=0);s=np.sqrt(np.mean((xs-expected_mean)**2,axis=0));expected_scale=np.where(s<1e-6,1.,s)
            ys=np.eye(102)[true['source']];target_mean=ys.mean(0)
            for lam in [config['primary_ridge_lambda']]+config['sensitivity_ridge_lambdas']:
                tag=kind+'_lambda'+str(lam).replace('.','p')
                row=next(r for r in rows if r['kind']==kind and r['lambda']==lam)
                with np.load(out/f'probe_{tag}.npz',allow_pickle=False) as probe:
                    mean=probe['mean'];scale=probe['scale'];w=probe['weights'];bias=probe['intercept']
                assert np.allclose(mean,expected_mean,rtol=0,atol=1e-12)
                assert np.allclose(scale,expected_scale,rtol=1e-12,atol=1e-12)
                assert np.array_equal(bias,target_mean)
                z=(xs-mean)/scale
                residual=z.T@(z@w-(ys-bias))/len(xs)+lam*w
                relative=float(np.linalg.norm(residual)/(np.linalg.norm(z.T@(ys-bias)/len(xs))+1e-15))
                assert relative<1e-8,('ridge stationarity',relative)
                normal_residuals.append(relative)
                with np.load(out/f'predictions_{tag}.npz',allow_pickle=False) as saved:
                    for role,x in [('source',xs),('valid',xv)]:
                        prediction=(((x-mean)/scale)@w+bias).argmax(1)
                        assert np.array_equal(prediction,saved[role])
                        score=independent_metrics(true[role],prediction)
                        for metric,value in score.items():assert abs(value-row[role][metric])<1e-12
                        confusion=np.zeros((102,102),dtype=np.int64)
                        np.add.at(confusion,(true[role],prediction),1)
                        assert np.array_equal(confusion,np.load(out/f'confusion_{tag}_{role}.npy'))
                        checked+=1
    return {'probe_prediction_sets_checked':checked,'expected':12,'probe_fits_checked':len(normal_residuals),
            'normal_equation_max_relative_residual':max(normal_residuals),'source_only_standardization_verified':True,'errors':[]}
