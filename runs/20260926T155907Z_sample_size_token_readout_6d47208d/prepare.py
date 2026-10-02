"""Source-only nested sampling; full valid X is used only for duplicate exclusion."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,time
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views
from ta_wf_next.packet_patches import packet_patch_inputs

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def digest_rows(x):return [hashlib.sha256(r.tobytes()).hexdigest() for r in x]
def pack(b):return {k:getattr(b,k) for k in ['packet','packet_mask','runs','runs_mask','windows','windows_mask','spans']}
def main():
 torch.set_num_threads(2);started=time.monotonic();c=json.loads((RUN/'config.json').read_text());assert c['status']=='draft'
 assert sha(ROOT/c['prior_prepared'])==c['prior_prepared_sha256'];assert sha(ROOT/c['prior_manifest'])==c['prior_manifest_sha256']
 prior=torch.load(ROOT/c['prior_prepared'],map_location='cpu',weights_only=False)
 mapping=json.loads((ROOT/'configs/datasets.json').read_text());base=Path(mapping['data_root'])/mapping['datasets']['proteus_temporal']['path']
 rawhash={str(base/n):sha(base/n) for n in ['train.npz','valid.npz']}
 with np.load(base/'valid.npz',allow_pickle=False) as f:
  vx=f['X'][:,:5000].astype(np.float32);assert np.isfinite(vx).all();vd=np.sign(vx).astype(np.int8);del vx
  vr=prior['valid']['rows'].numpy();vy=f['y'][vr].astype(np.int64)
 assert np.array_equal(vd[vr],prior['valid']['directions'].numpy()) and np.array_equal(vy,prior['valid']['labels'].numpy())
 valid_hash=set(digest_rows(vd));del vd
 with np.load(base/'train.npz',allow_pickle=False) as f:
  x=f['X'][:,:5000].astype(np.float32);labels=f['y'];assert np.isfinite(labels).all() and np.equal(labels,labels.astype(np.int64)).all();y=labels.astype(np.int64)
  finite=np.isfinite(x).all(1);assert finite.all(),'nonfinite raw input requires separate audit';directions=np.sign(x).astype(np.int8);del x
 assert set(y)==set(range(102))
 sr=prior['source']['rows'].numpy();assert np.array_equal(directions[sr],prior['source']['directions'].numpy()) and np.array_equal(y[sr],prior['source']['labels'].numpy())
 observed=directions!=0;suffix_bad=((~observed).cumsum(1)>0)&observed
 structural=observed.any(1)&~suffix_bad.any(1);del suffix_bad,observed
 hashes=digest_rows(directions);groups={}
 for i,h in enumerate(hashes):groups.setdefault(h,[]).append(i)
 conflict={h for h,ix in groups.items() if len(set(y[ix]))>1}
 overlap=[i for i,h in enumerate(hashes) if h in valid_hash]
 small=set(sr.tolist());seen=set();eligible=[];duplicates=[];excluded_conflict=[]
 for i in list(sr)+[i for i in range(len(y)) if i not in small]:
  h=hashes[i]
  if not structural[i] or h in valid_hash or h in conflict:
   assert i not in small,('original source not eligible',int(i))
   if h in conflict:excluded_conflict.append(int(i))
   continue
  if h in seen:
   assert i not in small,'original source duplicate';duplicates.append(int(i));continue
  seen.add(h);eligible.append(int(i))
 rng=np.random.default_rng(c['sampling_seed']);extras=[];counts={}
 for label in range(102):
  candidates=np.array([i for i in eligible if y[i]==label and i not in small],dtype=np.int64)
  counts[str(label)]={'total_eligible':len(candidates)+20,'extra_available':len(candidates)}
  assert len(candidates)>=60,('insufficient eligible source',label,len(candidates))
  extras.extend(rng.choice(candidates,size=60,replace=False).tolist())
 large=np.concatenate((sr,np.array(extras,dtype=np.int64)));assert len(large)==8160 and len(set(large))==8160
 assert np.all(np.bincount(y[large],minlength=102)==80)
 assert len({hashes[i] for i in large})==8160 and not {hashes[i] for i in large}&valid_hash
 manifest={'small_rows':sr.tolist(),'large_rows':large.tolist(),'valid_rows':vr.tolist(),'sampling_seed':c['sampling_seed'],'raw_sha256':rawhash,'counts':counts,'excluded_full_valid_overlap':overlap,'excluded_label_conflict':excluded_conflict,'excluded_invalid_structure':np.flatnonzero(~structural).tolist(),'excluded_duplicate':duplicates,'no_future_access':True}
 (RUN/'artifacts/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 # Generate only newly selected source rows; preserve existing source/valid tensors exactly.
 pieces=[]
 for start in range(0,len(extras),256):
  if time.monotonic()-started>600:raise TimeoutError('preparation budget')
  b=batch_from_views([generate_views(directions[i],input_kind='direction',budget=5000,window_sizes=(50,250)) for i in extras[start:start+256]],packet_patch=50,max_runs=128)
  packet_patch_inputs(b,torch.from_numpy(directions[extras[start:start+256]]),condition='ordered')
  pieces.append(pack(b))
 original={}
 for k,v in prior['source']['original'].items():
  original[k]={n:torch.cat([v[n]]+[p[k][n] for p in pieces]) for n in v} if isinstance(v,dict) else torch.cat([v]+[p[k] for p in pieces])
 new={'source20':prior['source'],'source80':{'original':original,'directions':torch.from_numpy(directions[large]),'labels':torch.from_numpy(y[large]),'rows':torch.from_numpy(large)},'valid':prior['valid']}
 torch.save(new,RUN/'artifacts/prepared.pt')
 audit={'complete':True,'raw_train_count':len(y),'raw_valid_count':2160,'small_count':2040,'large_count':8160,'valid_count':510,'min_eligible_per_class':min(v['total_eligible'] for v in counts.values()),'full_valid_overlap_exclusions':len(overlap),'label_conflict_exclusions':len(excluded_conflict),'duplicate_exclusions':len(duplicates),'structural_exclusions':int((~structural).sum()),'nested_small_preserved':True,'fixed_valid_preserved':True,'selected_source_valid_overlap':0,'prepared_sha256':sha(RUN/'artifacts/prepared.pt'),'manifest_sha256':sha(RUN/'artifacts/manifest.json'),'elapsed_seconds':time.monotonic()-started,'future_access':False}
 (RUN/'artifacts/input_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
