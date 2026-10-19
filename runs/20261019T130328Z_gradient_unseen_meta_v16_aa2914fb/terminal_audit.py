"""Terminal delivery/role/descriptor verification; no fitting, scoring changes or selection."""
from runtime import *
from scipy.special import logsumexp
import torch

def main():
 verify();delivery=json.loads((RUN/'artifacts/delivery_audit.json').read_text())
 for p,h in delivery['hashes'].items():assert sha(ROOT/p)==h,p
 roles=json.loads((RUN/'artifacts/roles.json').read_text());raw=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',weights_only=True,map_location='cpu');groups={}
 for role in ['IM','GU','calibration','diagnostic']:
  d=raw['source' if role=='IM' else 'valid'];rr=roles[role];ix=rr['indices'];assert d['rows'][ix].tolist()==rr['rows'] and d['labels'][ix].tolist()==rr['labels'];h=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in d['tam'][ix]];assert h==rr['groups'];assert len(set(h))==len(h);groups[role]=set(h)
 for a in groups:
  for b in groups:
   if a!=b:assert groups[a].isdisjoint(groups[b])
 cal=json.loads((ROOT/C['v12']/'artifacts/source_calibration.json').read_text());w=np.array(cal['weights']);temp=np.array(cal['temperatures']);checked=0;clean=[]
 for seed in C['seeds']:
  for sl in C['slices']:
   key=f'{seed}_{sl}';scale=cal['states'][key]['scales'];parts=[]
   for ci,case in enumerate(C['perturbations']):
    ds=[read(RUN/'artifacts'/f'validpool_{ss}_{case["name"]}.npz') for ss in C['seeds']];d=ds[C['seeds'].index(seed)];b=d['frozen_logits'].argmax(1);c=d[sl+'_logits'].argmax(1);q=np.stack([a['aux_logits'] for a in ds]).astype(np.float64)/temp[:,:,None,None];index=np.arange(len(c));qc=q[:,:,index,c].copy();qb=q[:,:,index,b].copy();lp=(w[:,:,None]*(qc-logsumexp(q,axis=-1))).sum((0,1));q[:,:,index,b]=-np.inf;q[:,:,index,c]=-np.inf;third=(w[:,:,None]*(qc-q.max(-1))).sum((0,1));odds=qc-qb;mean=(w[:,:,None]*odds).sum((0,1));sd=np.sqrt(np.maximum((w[:,:,None]*(odds-mean)**2).sum((0,1)),0));soft=mean-.25*sd;m=d[sl+'_margin'].astype(np.float64);s=(m-scale['margin']['mean'])/scale['margin']['sd']+.25*(soft-scale['weighted_soft']['mean'])/scale['weighted_soft']['sd'];parts.append(dict(rows=d['rows'],X=np.stack([m,lp,third],-1),v12_primary=s))
   full={k:np.concatenate([d[k] for d in parts]) for k in parts[0]}
   for role in ['GU','calibration','diagnostic']:
    expected=read(RUN/'artifacts'/f'features_{role}_{key}.npz');mask=np.isin(full['rows'],roles[role]['rows']);assert np.array_equal(full['X'][mask],expected['X']);assert np.array_equal(full['v12_primary'][mask],expected['v12_primary']);checked+=1
   im=read(RUN/'artifacts'/f'features_IM_{key}.npz');old=read(ROOT/C['v14']/'artifacts'/f'source_features_train_{key}.npz');mask=np.isin(old['rows'],roles['IM']['rows']);assert all(np.array_equal(v[mask],im[k]) for k,v in old.items());checked+=1
   for role in ['IM','GU']:
    d=read(RUN/'artifacts'/f'features_{role}_{key}.npz');mask=d['case_id']==0;clean.append(dict(seed=seed,slice=sl,role=role,rows=int(mask.sum()),frozen_accuracy=float((d['labels'][mask]==d['base'][mask]).mean()),R=int((d['types'][mask]==1).sum()),D=int((d['types'][mask]==2).sum()),W=int((d['types'][mask]==3).sum())))
 finite=0
 for p in (RUN/'artifacts').glob('*.npz'):
  with np.load(p) as z:
   for k in z.files:assert np.isfinite(z[k]).all(),(p,k)
  finite+=1
 save('terminal_audit.json',dict(passed=True,original_delivery_files=len(delivery['hashes']),role_contents_rebuilt=True,descriptor_states_recomputed=checked,finite_cache_files=finite,source_models_changed=False,refit=False,original_delivery_sha256=sha(RUN/'artifacts/delivery_audit.json')));save('clean_supervision_counts.json',clean)
 print(json.dumps(dict(passed=True,descriptor_states=checked,finite_npz=finite)))

if __name__=='__main__':main()
