"""P0 artifact audit; original y/URL metadata only, no original X/model load."""
import csv,hashlib,io,json,time,zipfile
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];OUT=RUN/'artifacts'
CFG=json.loads((RUN/'config.json').read_text());PRE=ROOT/CFG['input_run'];P=PRE/'artifacts'
BASE=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root']);START=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def deadline():
 if time.monotonic()-START>CFG['wall_time_seconds']:raise TimeoutError('budget exceeded')
def stat(p):
 s=p.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
def meta_field(p,key):
 with zipfile.ZipFile(p) as z:return np.load(io.BytesIO(z.read(key+'.npy')),allow_pickle=False)
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert CFG['status']=='frozen'
 for p,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 old=json.loads((P/'p0_audit.json').read_text());meta=json.loads((P/'core_file_index.json').read_text());n=len(meta)
 labels=[];ids={};views=['raw10000','float32_first5000','direction_first5000']
 with np.load(P/'core_row_group_ids.npz',allow_pickle=False) as z:
  for v in views:ids[v]=[z[f'{v}_{i}'] for i in range(n)]
 for i,m in enumerate(meta):
  path=BASE/m['path'];assert stat(path)==m['source_stat'],path
  y=meta_field(path,'y');assert len(y)==m['rows'] and np.isfinite(y).all() and np.equal(y,np.trunc(y)).all();labels.append(y.astype(np.int64))
  for v in views:assert len(ids[v][i])==m['rows'] and (ids[v][i]>=0).all()
 assert sum(m['rows'] for m in meta)==old['core_npz']['rows']
 # Check equivalence partition over ALL rows; bijection proof handles arbitrary IDs.
 raw=np.concatenate(ids[views[0]]);unique=np.unique(raw);assert np.array_equal(unique,np.arange(len(unique)))
 for v in views[1:]:
  other=np.concatenate(ids[v]);pairs=np.unique(np.stack([raw,other],axis=1),axis=0)
  assert len(pairs)==len(unique)==len(np.unique(other)),v
  assert len(np.unique(pairs[:,0]))==len(pairs)==len(np.unique(pairs[:,1])),v
 print('PASS all-row partition equality',flush=True)
 count=np.bincount(raw);protected=np.zeros(len(count),bool);group_lab=np.full(len(count),-1,np.int64);conflict=np.zeros(len(count),bool)
 for i,m in enumerate(meta):
  if m['role']!='train':protected[ids[views[0]][i]]=True
  for g,y in zip(ids[views[0]][i],labels[i]):
   if group_lab[g]<0:group_lab[g]=y
   elif group_lab[g]!=y:conflict[g]=True
 assert not conflict.any(),'P0 said no conflicts'
 expected={};seen_train=set();keep=[];file_totals=[];class_counts=Counter()
 for i,m in enumerate(meta):
  before=len(keep);reasons=Counter()
  if m['role']=='train':
   for r,(g,y) in enumerate(zip(ids[views[0]][i],labels[i])):
    if protected[g]:reason='matches_original_nontrain_role'
    elif int(g) in seen_train:reason='duplicate_train_copy'
    else:
     keep.append((i,r,int(y),int(g)));class_counts[int(y)]+=1;seen_train.add(int(g));continue
    expected[(m['path'],r)]=reason;reasons[reason]+=1
  file_totals.append(dict(file=m['path'],role=m['role'],rows=m['rows'],retained_train_candidates=len(keep)-before,exclusions=dict(reasons)))
 with (P/'core_quarantine_candidates.csv').open() as f:
  candidates=list(csv.DictReader(f))
 assert len(candidates)==len(expected)==old['core_npz']['quarantine_candidate_rows']
 for row in candidates:
  key=(row['file'],int(row['row']));reason=expected.pop(key)
  assert row['original_role']=='train' and row['applied']=='False'
  assert row['reasons']=='|'.join(sorted(v+':'+reason for v in views)),row
 assert not expected
 # All duplicate CSV memberships/labels/actions agree with row arrays and role policy.
 fileidx={m['path']:i for i,m in enumerate(meta)}
 for v in views:
  checked=set();actions=Counter();dup_seen=set()
  with (P/f'core_{v}_duplicate_members.csv').open() as f:
   for row in csv.DictReader(f):
    deadline();i=fileidx[row['file']];r=int(row['row']);g=int(row['group_id']);assert (i,r) not in checked;checked.add((i,r))
    assert ids[v][i][r]==g and labels[i][r]==int(row['label']) and meta[i]['role']==row['original_role']
    assert count[ids[views[0]][i][r]]>1
    if meta[i]['role']!='train':action,reason='group_constraint_only','same_observed_input_group'
    elif protected[ids[views[0]][i][r]]:action,reason='exclude_from_future_joint_training','matches_original_nontrain_role'
    elif g in dup_seen:action,reason='exclude_from_future_joint_training','duplicate_train_copy'
    else:action,reason='canonical_train_candidate','first_deterministic_train_copy';dup_seen.add(g)
    assert row['action']==action and row['reason']==reason,row;actions[action]+=1
  assert len(checked)==int(count[count>1].sum())
  assert dict(actions)==old['core_npz']['views'][v]['action_counts']
  print('PASS CSV',v,len(checked),flush=True)
 with (OUT/'conservative_train_candidates.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['file','row','label','raw_group_id','original_role','eligibility','applied'])
  for i,r,y,g in keep:w.writerow([meta[i]['path'],r,y,g,'train','structural_candidate_permission_and_identity_pending',False])
 assert len(set(g for i,r,y,g in keep))==len(keep) and all(not protected[g] for i,r,y,g in keep)
 # Unique shared content groups support label correspondences, not official semantic mapping.
 family_groups=defaultdict(dict)
 for i,m in enumerate(meta):
  d=family_groups[m['dataset']]
  for g,y in zip(ids[views[0]][i],labels[i]):
   assert int(g) not in d or d[int(g)]==int(y);d[int(g)]=int(y)
 mapping=[]
 with (OUT/'cross_family_label_evidence.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['family_a','family_b','label_a','label_b','shared_unique_raw_inputs','status'])
  for a,b in combinations(family_groups,2):
   da,db=family_groups[a],family_groups[b];shared=da.keys()&db.keys();c=Counter((da[g],db[g]) for g in shared)
   for (ya,yb),ct in sorted(c.items()):w.writerow([a,b,ya,yb,ct,'content_supported_correspondence_not_official_site_map'])
   mapping.append(dict(a=a,b=b,shared_unique_inputs=len(shared),label_pairs=len(c),unequal_numeric_pairs=sum(ya!=yb for ya,yb in c),covered_labels_a=len(set(y for y,_ in c)),covered_labels_b=len(set(y for _,y in c))))
 print('PASS role policy; retained',len(keep),flush=True)
 # URL evidence supports subpage label semantics only.
 urls=meta_field(BASE/'BehaviorDrift/subpage.npz','url');yi=fileidx['BehaviorDrift/subpage.npz'];yl=labels[yi];assert len(urls)==len(yl)
 urlsets=defaultdict(set);hosts=defaultdict(Counter);url_labs=defaultdict(set);bad=[]
 for r,(url,y) in enumerate(zip(urls,yl)):
  host=urlsplit(str(url)).hostname or '';host=host.lower();host=host[4:] if host.startswith('www.') else host
  if not host:bad.append(r)
  hosts[int(y)][host]+=1;urlsets[int(y)].add(str(url));url_labs[str(url)].add(int(y))
 hostname_labels=defaultdict(set)
 for y,hs in hosts.items():
  for h in hs:hostname_labels[h].add(y)
 url_report=dict(rows=len(urls),unique_urls=len(url_labs),extra_exact_url_copies=len(urls)-len(url_labs),exact_url_multilabel_count=sum(len(x)>1 for x in url_labs.values()),invalid_hostname_rows=bad,labels_with_multiple_hostnames=[y for y,x in hosts.items() if len(x)>1],hostnames_with_multiple_labels={h:sorted(x) for h,x in hostname_labels.items() if len(x)>1},label_hostname_counts={str(y):dict(x) for y,x in sorted(hosts.items())},limits=['www stripping is descriptive; no public suffix normalization','subpage URL semantics cannot certify homepage label mapping or acquisition independence'])
 dump('subpage_url_identity.json',url_report)
 with (OUT/'subpage_label_hostnames.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['raw_label','hostname_www_stripped','rows','evidence_scope'])
  for y,hs in sorted(hosts.items()):
   for h,ct in sorted(hs.items()):w.writerow([y,h,ct,'BehaviorDrift/subpage only'])
 # Reconstruct HDF5 candidate rules from all duplicate member metadata.
 hm=json.loads((P/'shusterman_file_index.json').read_text());hf={m['file']:i for i,m in enumerate(hm)}
 with np.load(P/'shusterman_row_group_ids.npz',allow_pickle=False) as z:hi=[z[f'file_{i}'] for i in range(len(hm))]
 hc=np.bincount(np.concatenate(hi));hg=defaultdict(list);hseen=set()
 with (P/'shusterman_duplicate_members.csv').open() as f:
  for row in csv.DictReader(f):
   i=hf[row['file']];r=int(row['row']);g=int(row['group_id']);assert (i,r) not in hseen;hseen.add((i,r))
   assert hi[i][r]==g and 0<=r<hm[i]['rows'] and hm[i]['role']==row['original_role'];hg[g].append(row)
 assert len(hseen)==int(hc[hc>1].sum())
 hx={};zero_cross=0;nonzero_cross=0
 for g,rows in hg.items():
  assert len(rows)==hc[g];zero=all(int(r['length'])==0 for r in rows);lc=len(set(r['site'] for r in rows))>1
  main=[r for r in rows if r['component']!='static_webpages_network'];prot=any(r['original_role']=='test' for r in main);first=False
  for r in rows:
   reason=None
   if zero:reason='all_zero_unusable_trace'
   elif lc:reason='nonzero_exact_vector_conflicting_site_labels'
   elif r['component']=='static_webpages_network':reason='auxiliary_component_not_in_main_pool'
   elif r['original_role']=='train' and prot:reason='nonzero_matches_test_trace'
   elif r['original_role']=='train':
    if first:reason='duplicate_main_training_copy'
    first=True
   if reason:hx[(r['file'],int(r['row']))]=reason
  for a,b in combinations(rows,2):
   if Path(a['file']).parent==Path(b['file']).parent and a['original_role']!=b['original_role']:
    if zero:zero_cross+=1
    else:nonzero_cross+=1
 for i,m in enumerate(hm):
  assert len(hi[i])==m['rows'] and stat(BASE/m['file'])==m['source_stat']
  if m['component']=='static_webpages_network':
   for r in range(m['rows']):hx[(m['file'],r)]='auxiliary_component_not_in_main_pool'
 with (P/'shusterman_quarantine_candidates.csv').open() as f:
  hs=list(csv.DictReader(f))
 assert len(hs)==len(hx)
 for r in hs:assert hx.pop((r['file'],int(r['row'])))==r['reason'] and r['applied']=='False'
 assert not hx and nonzero_cross==0 and zero_cross==2
 for m in meta+hm:
  p=BASE/m.get('path',m.get('file'));assert stat(p)==m['source_stat']
 model=json.loads((ROOT/'runs/20261001T104959Z_gpu_multiview_generator_head_factorial_cf47d62c/manifest.json').read_text())
 result=dict(status='passed',scope='artifact and metadata structural audit; no training or X re-scan',p0_all_row_three_view_partition_equal=True,core_duplicate_csv_members_checked_per_view=int(count[count>1].sum()),p0_core_candidates_exactly_reproduced=len(candidates),p0_hdf5_candidates_exactly_reproduced=len(hs),hdf5_duplicate_members_checked=len(hseen),hdf5_same_date_nonzero_pairs=nonzero_cross,hdf5_same_date_zero_member_pairs=zero_cross,total_original_train_rows=sum(m['rows'] for m in meta if m['role']=='train'),retained_conservative_train_unique_inputs=len(keep),retained_label_counts=dict(sorted(class_counts.items())),retained_protected_overlap=0,per_file=file_totals,cross_family_content_label_evidence=mapping,url_summary={k:v for k,v in url_report.items() if k!='label_hostname_counts'},model_boundary_historical_reference=dict(source_rows=model['audits']['source']['rows'],valid_rows=model['audits']['valid']['rows'],selected_overlap=model['selected_source_valid_direction_overlap'],source_excluded_for_full_valid_overlap=model['prior_overlap_audit']['source_excluded_for_full_valid_overlap'],not_rerun=True),all_source_stats_unchanged=True,limits=['full byte matches inherited from P0, not independently rerun','original nontrain protection is conservative; condition-specific train eligibility requires explicit new protocol','numeric correspondence on shared inputs does not certify unmapped rows or official website mapping','no acquisition/session/circuit independence established'],elapsed_seconds=time.monotonic()-START)
 dump('audit.json',result)
 for p,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 integrity={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='integrity.json'};dump('integrity.json',integrity)
 print('FINISHED',len(keep),'retained candidates',len(class_counts),'labels',flush=True)
if __name__=='__main__':main()
