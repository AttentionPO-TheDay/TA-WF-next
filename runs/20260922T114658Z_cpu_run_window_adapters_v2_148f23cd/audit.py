"""Post-training independent audit; no model selection or further training."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parents[1]
cfg=json.loads((R/'config.json').read_text())
for name in ('pretraining_seal.json','output_seal.json'):
 for f,h in json.loads((R/'artifacts'/name).read_text()).items():
  assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h, f
manifest=json.loads((R/'artifacts/manifest.json').read_text())
assert set(manifest['sampling'])=={'source','valid'}
assert not set(manifest['direction_hashes']['source'])&set(manifest['direction_hashes']['valid'])
data=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
stats=np.load(R/'artifacts/window_statistics.npz')
for role in cfg['roles']:
 item=manifest['sampling'][role]
 with np.load(data/item['path'],allow_pickle=False) as z:
  raw=z['X'][item['rows'],:5000]; labels=z['y'][item['rows']].astype(np.int64)
 with np.load(R/'artifacts'/f'{role}_inputs.npz') as inputs:
  np.testing.assert_array_equal(inputs['labels'],labels)
  assert np.array_equal(np.unique(labels,return_counts=True)[1],np.full(102,20 if role=='source' else 5))
  if role=='source':
   mean=inputs['raw_windows'].mean(0); std=inputs['raw_windows'].std(0); std=np.where(std<1e-6,1.,std)
   np.testing.assert_array_equal(mean,stats['mean']); np.testing.assert_array_equal(std,stats['std'])
  np.testing.assert_array_equal(inputs['windows'],(inputs['raw_windows']-stats['mean'])/stats['std'])
  for i,row in enumerate(raw):
   direction=np.sign(row).astype(np.int8)
   assert hashlib.sha256(direction.tobytes()).hexdigest()==manifest['direction_hashes'][role][i]
   nz=direction[direction!=0]; assert np.array_equal(direction[:len(nz)],nz)
   starts=np.r_[0,np.flatnonzero(nz[1:]!=nz[:-1])+1]; ends=np.r_[starts[1:],len(nz)]
   ds=np.where(nz[starts]<0,1,2)[:512]; bins=np.floor(np.log2(ends-starts)).astype(int)[:512]+1
   np.testing.assert_array_equal(inputs['direction'][i,:len(ds)],ds)
   np.testing.assert_array_equal(inputs['bucket'][i,:len(ds)],bins)
   np.testing.assert_array_equal(inputs['mask'][i],np.arange(512)<len(ds))
   assert not inputs['direction'][i,len(ds):].any() and not inputs['bucket'][i,len(ds):].any()
   ws=np.zeros(240,np.float32); offset=0
   for width in (50,250):
    for j,start in enumerate(range(0,len(nz),width)):
     w=nz[start:start+width]; ws[offset+2*j]=np.mean(w>0)
     ws[offset+2*j+1]=np.mean(w[1:]!=w[:-1]) if len(w)>1 else 0.
    offset+=2*((5000+width-1)//width)
   np.testing.assert_array_equal(inputs['raw_windows'][i],ws)
 if role=='valid': y=labels
rows=json.loads((R/'artifacts/metrics.json').read_text()); history=json.loads((R/'artifacts/history.json').read_text())
pred=np.load(R/'artifacts/predictions.npz')
expected={f'{a}_{s}' for a in cfg['arms'] for s in cfg['seeds']}
assert len(rows)==15 and {r['key'] for r in rows}==set(pred.files)==set(history)==expected
for r in rows:
 p=pred[r['key']]; assert p.shape==(510,) and ((p>=0)&(p<102)).all()
 f=[]
 for c in range(102):
  tp=int(((p==c)&(y==c)).sum()); den=int((p==c).sum()+(y==c).sum()); f.append(2*tp/den if den else 0.)
 assert abs(float((p==y).mean())-r['accuracy'])<1e-12 and abs(sum(f)/102-r['macro_f1'])<1e-12
 h=history[r['key']]; assert [v['epoch'] for v in h]==list(range(1,16))
 best=max(h,key=lambda v:v['macro_f1'])
 assert best['epoch']==r['best_epoch'] and best['macro_f1']==r['macro_f1'] and best['accuracy']==r['accuracy']
aggregate={}
for arm in cfg['arms']:
 a=np.array([[next(r for r in rows if r['key']==f'{arm}_{s}')[k] for k in ('accuracy','macro_f1')] for s in cfg['seeds']])*100
 aggregate[arm]={'values':a.tolist(),'mean':a.mean(0).tolist(),'sd':a.std(0,ddof=1).tolist()}
pairs={}
for a,b in [('fusion','run_pair'),('fusion','window_pair'),('fusion_specific','fusion_shared'),('fusion_specific','fusion'),('fusion_shared','fusion'),('fusion_specific','window_pair')]:
 delta=np.array(aggregate[a]['values'])-np.array(aggregate[b]['values'])
 pairs[a+'-'+b]={'delta_pp':delta.tolist(),'mean_pp':delta.mean(0).tolist(),'positive_f1_seeds':int((delta[:,1]>0).sum()),'screen_gate':bool((delta[:,1]>0).sum()>=2 and delta[:,1].mean()>0)}
out={'arms':aggregate,'pairs':pairs,'last_epoch_selected':sum(r['best_epoch']==15 for r in rows)}
(R/'artifacts/aggregate.json').write_text(json.dumps(out,indent=2))
(R/'artifacts/integrity.json').write_text(json.dumps({'errors':0,'metrics':15,'predictions':7650,'independent_input_reconstruction':2550,'source_only_normalization':True,'seals':True,'selection':True,'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print(json.dumps(out,indent=2))
