"""Terminal verification and descriptive diagnosis; no fitting or method selection."""
from runtime import *
from methods import project,reference_features
from scipy.special import expit

def main():
 verify()
 # Verify original delivery before any interpretation amendments.
 delivery=RUN/'artifacts/delivery_audit.json';original=json.loads(delivery.read_text())
 for p,h in original['hashes'].items():assert sha(ROOT/p)==h,p
 dm=json.loads((RUN/'checkpoints/density.json').read_text());checks=[];clean=[]
 import torch
 src=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',weights_only=True,map_location='cpu')['source'];hashes={int(row):hashlib.sha256(x.numpy().tobytes()).hexdigest() for row,x in zip(src['rows'],src['timestamps'])}
 for seed in C['seeds']:
  for sl in C['slices']:
   key=f'{seed}_{sl}';tr=read(ROOT/C['v14']/'artifacts'/f'source_features_train_{key}.npz');ref=reference_features(seed,sl);m=dm[key];mask=tr['types']>0;x=np.concatenate([tr['X'][mask],ref['X'][ref['changed']]]);x=(x-np.array(m['mean']))/np.array(m['sd']);p=np.zeros(len(x));seen=np.zeros(len(x),int);groups=np.r_[np.array(['source:'+hashes[int(row)] for row in tr['rows'][mask]]),np.array(['reference:'+str(int(row)) for row in ref['rows'][ref['changed']]])]
   for fold in m['folds']:
    ix=np.array(fold['heldout']);comp=np.ones(len(x),bool);comp[ix]=False;assert set(groups[ix]).isdisjoint(groups[comp]);p[ix]=expit((x[ix]@np.array(fold['coef']).T).ravel()+np.array(fold['intercept'])[0]);seen[ix]+=1
   assert (seen==1).all();raw=np.exp(np.clip(np.log(np.maximum(p[:mask.sum()],1e-12))-np.log(np.maximum(1-p[:mask.sum()],1e-12)),-50,50));weights=np.clip(raw,.1,10);weights/=weights.mean();err=float(np.max(np.abs(weights-np.array(m['source_changed_weights']))));assert err<1e-12,err;checks.append(dict(seed=seed,slice=sl,weight_max_error=err,folds=5,group_disjoint=True))
   for role in ['train','valid']:
    d=read(ROOT/C['v14']/'artifacts'/f'source_features_{role}_{key}.npz');z=d['case_id']==0;clean.append(dict(seed=seed,slice=sl,role=role,rows=int(z.sum()),frozen_accuracy=float((d['base'][z]==d['labels'][z]).mean()),R=int((d['types'][z]==1).sum()),D=int((d['types'][z]==2).sum()),W=int((d['types'][z]==3).sum())))
   d=read(ROOT/C['v13']/'artifacts'/f'rows_query_{key}.npz');t=np.zeros(len(d['rows']),np.int8);changed=d['base']!=d['candidates'];t[changed&(d['base']!=d['labels'])&(d['candidates']==d['labels'])]=1;t[changed&(d['base']==d['labels'])]=2;t[changed&(d['base']!=d['labels'])&(d['candidates']!=d['labels'])]=3;assert np.array_equal(t,d['type'])
 save('density_reload_audit.json',dict(passed=True,states=checks,all_oof_weights_rebuilt=True,query_taxonomy_rebuilt=True))
 save('posthoc_clean_source_gap.json',clean)
 rows=[]
 for partition,file in [('sourcevalid','fusion_source.json'),('query','fusion_target.json')]:
  d=json.loads((RUN/'artifacts'/file).read_text())
  for sl in C['slices']:
   for arm in C['arms']:
    for kind in ['branch','fused']:
     rr=[r for r in d if r['slice']==sl and r['arm']==arm and r['kind']==kind];rows.append(dict(partition=partition,slice=sl,arm=arm,kind=kind,pauc_norm_mean=float(np.mean([r['pauc_norm'] for r in rr])),states=len(rr)))
 save('posthoc_fusion_summary.json',rows)
 save('terminal_verification.json',dict(passed=True,original_delivery_sha256=sha(delivery),original_delivery_files_verified=len(original['hashes']),density_states=6,folds_rebuilt=30,source_valid_gap_descriptive_only=True,no_refit=True,no_new_score_selection=True))
 print('original delivery, 30 domain folds, all weights and query taxonomy verified')

if __name__=='__main__':main()
