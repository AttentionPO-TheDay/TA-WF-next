from runtime import *
from scoring import *
import traceback,sys

def freeze():
 paths=list(RUN.glob('*.py'))+[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',ROOT/'configs/datasets.json',ROOT/'PROTOCOL.md',ROOT/'AGENTS.md',ROOT/C['source_run']/'artifacts/prepared.pt',ROOT/C['v14']/'artifacts/source_rows.json',ROOT/C['v14']/'checkpoints/rerank_models.json',ROOT/C['v12']/'artifacts/source_calibration.json',ROOT/C['v12']/'artifacts/cells.json']
 for s in C['seeds']:
  paths += [ROOT/C['source_run']/'checkpoints'/f'pure_source_wide_bn_{s}.pth',ROOT/C['v10']/'checkpoints'/f'{s}_direction_iat.pt',ROOT/C['v10']/'artifacts'/f'sourcevalid_{s}_direction_iat.npz']
  for v in C['views'][1:]:paths.append(ROOT/C['v11']/'artifacts'/f'logits_{s}_direction_iat_valid_{v}.npz')
  for sl in C['slices']:
   paths += [ROOT/C['v9']/'checkpoints'/f'{s}_{sl}_smooth_balance.pt',ROOT/C['v13']/'artifacts'/f'rows_query_{s}_{sl}.npz',ROOT/C['v14']/'artifacts'/f'source_features_train_{s}_{sl}.npz']
   for cond in ['frozen','smooth_balance']:paths.append(ROOT/C['v9']/'artifacts'/f'valid_{s}_{sl}_{cond}.npz')
 save('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))})

def prepare():
 import torch
 verify();data=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',weights_only=True,map_location='cpu');allowed=set(json.loads((ROOT/C['v14']/'artifacts/source_rows.json').read_text())['train']);roles={};entries={}
 for role,part in [('train','source'),('valid','valid')]:
  d=data[part];groups={}
  for i in range(len(d['rows'])):
   if role=='train' and int(d['rows'][i]) not in allowed:continue
   h=hashlib.sha256(d['tam'][i].numpy().tobytes()).hexdigest();y=int(d['labels'][i])
   if h in groups:assert groups[h]['label']==y
   else:groups[h]=dict(group=h,label=y,index=i,row=int(d['rows'][i]))
  entries[role]=groups
 for cls in range(102):
  vg=sorted([v for v in entries['valid'].values() if v['label']==cls],key=lambda v:hashlib.sha256(('v16|valid|'+v['group']).encode()).digest());tg=sorted([v for v in entries['train'].values() if v['label']==cls],key=lambda v:hashlib.sha256(('v16|train|'+v['group']).encode()).digest());assert len(vg)>=17 and len(tg)>=6
  for name,vs in [('IM',tg[:6]),('GU',vg[:6]),('calibration',vg[6:11]),('diagnostic',vg[11:17])]:roles.setdefault(name,[]).extend(vs)
 for a in roles:
  for b in roles:
   if a!=b:assert set(v['group'] for v in roles[a]).isdisjoint(v['group'] for v in roles[b]),(a,b)
 out={k:{'indices':[v['index'] for v in vs],'rows':[v['row'] for v in vs],'labels':[v['label'] for v in vs],'groups':[v['group'] for v in vs]} for k,vs in roles.items()};out['valid_pool']={k:sum([out[n][k] for n in ['GU','calibration','diagnostic']],[]) for k in ['indices','rows','labels','groups']};save('roles.json',out);save('role_audit.json',dict(passed=True,counts={k:len(v['rows']) for k,v in out.items()},source_valid_group_overlap=0,valid_min_unique_groups_per_class=min(sum(v['label']==cl for v in entries['valid'].values()) for cl in range(102)),qualification='gradient-unseen only; historically selected validation, not independent confirmation'))
 save('role_freeze.json',{str((RUN/'artifacts/roles.json').relative_to(ROOT)):sha(RUN/'artifacts/roles.json')})

def features():
 verify();roles=json.loads((RUN/'artifacts/roles.json').read_text());cal=json.loads((ROOT/C['v12']/'artifacts/source_calibration.json').read_text());counts=[]
 for seed in C['seeds']:
  for sl in C['slices']:
   scales=cal['states'][f'{seed}_{sl}']['scales'];parts=[]
   for ci,case in enumerate(C['perturbations']):
    ds=[read(RUN/'artifacts'/f'validpool_{ss}_{case["name"]}.npz') for ss in C['seeds']];own=ds[C['seeds'].index(seed)];assert all(np.array_equal(own['rows'],d['rows']) and np.array_equal(own['labels'],d['labels']) for d in ds);b=own['frozen_logits'].argmax(-1);cc=own[sl+'_logits'].argmax(-1);lp,third,soft=evidence(np.stack([d['aux_logits'] for d in ds]),b,cc,cal);m=own[sl+'_margin'].astype(np.float64);s=(m-scales['margin']['mean'])/scales['margin']['sd']+.25*(soft-scales['weighted_soft']['mean'])/scales['weighted_soft']['sd'];parts.append(dict(X=np.stack([m,lp,third],-1),rows=own['rows'],labels=own['labels'],base=b,candidates=cc,types=kind(own['labels'],b,cc),case_id=np.full(len(b),ci,np.int8),v9_margin=m,v12_primary=s))
   allvalid={k:np.concatenate([d[k] for d in parts]) for k in parts[0]}
   for role in ['GU','calibration','diagnostic']:
    mask=np.isin(allvalid['rows'],roles[role]['rows']);d={k:v[mask] for k,v in allvalid.items()};assert len(d['rows'])==5*len(roles[role]['rows']);np.savez_compressed(RUN/'artifacts'/f'features_{role}_{seed}_{sl}.npz',**d)
    for ci in range(5):counts.append(dict(seed=seed,slice=sl,role=role,case_id=ci,rows=int((d['case_id']==ci).sum()),**{n:int(((d['case_id']==ci)&(d['types']==k)).sum()) for n,k in [('R',1),('D',2),('W',3)]}))
   d=read(ROOT/C['v14']/'artifacts'/f'source_features_train_{seed}_{sl}.npz');mask=np.isin(d['rows'],roles['IM']['rows']);d={k:v[mask] for k,v in d.items()};assert len(d['rows'])==3060;np.savez_compressed(RUN/'artifacts'/f'features_IM_{seed}_{sl}.npz',**d)
   for ci in range(5):counts.append(dict(seed=seed,slice=sl,role='IM',case_id=ci,rows=int((d['case_id']==ci).sum()),**{n:int(((d['case_id']==ci)&(d['types']==k)).sum()) for n,k in [('R',1),('D',2),('W',3)]}))
 save('source_type_counts.json',counts);csvout('source_type_counts.csv',counts)
 paths=list((RUN/'artifacts').glob('validpool_*.npz'))+list((RUN/'artifacts').glob('features_*.npz'));save('forward_freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})

def fit():
 verify();save('metric_preflight.json',preflight());features();models={};thresholds={};feasibility=[];old=json.loads((ROOT/C['v14']/'checkpoints/rerank_models.json').read_text());total=0
 for seed in C['seeds']:
  for sl in C['slices']:
   key=f'{seed}_{sl}';models[key]={};va=read(RUN/'artifacts'/f'features_calibration_{key}.npz')
   for arm in C['arms']:
    role='IM' if arm.startswith('IM') else 'GU';d=read(RUN/'artifacts'/f'features_{role}_{key}.npz');mask=np.ones(len(d['rows']),bool)
    if arm.endswith('_aug'):mask=d['case_id']!=0
    elif arm.endswith('_clean'):mask=d['case_id']==0
    dd={k:v[mask] for k,v in d.items()};nums={k:int((dd['types']==t).sum()) for k,t in [('R',1),('D',2),('W',3)]};minimum=C['fit']['min_clean_each_type'] if arm=='GU_clean' else C['fit']['min_each_type'];ok=min(nums.values())>=minimum;feasibility.append(dict(seed=seed,slice=sl,arm=arm,feasible=ok,min_each_type=minimum,raw_label_budget=612,**nums));models[key][arm]=fitmodel(dd) if ok else None;total+=int(ok)
   scores=values(va,models[key],old[key]['confirmation_primary']);thresholds[key]={k:calibrate(va,v) for k,v in scores.items()};np.savez_compressed(RUN/'artifacts'/f'calibration_scores_{key}.npz',**scores);print(json.dumps(dict(stage='fit',key=key,models=total)),flush=True)
 (RUN/'checkpoints/models.json').write_text(json.dumps(models,indent=2,allow_nan=False)+'\n');save('source_thresholds.json',thresholds);save('feasibility.json',feasibility);csvout('feasibility.csv',feasibility)
 paths=[RUN/'checkpoints/models.json',RUN/'artifacts/source_thresholds.json']+list((RUN/'artifacts').glob('calibration_scores_*.npz'));save('fit_freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});save('fit_complete.json',dict(passed=True,models=total,query_arrays_opened=False,source_diagnostic_used_for_selection=False))

def fitverify():
 for name in ['role_freeze.json','forward_freeze.json','fit_freeze.json']:
  for p,h in json.loads((RUN/'artifacts'/name).read_text()).items():assert sha(ROOT/p)==h,p

def target(seed,sl):
 a=read(ROOT/C['v13']/'artifacts'/f'rows_query_{seed}_{sl}.npz');return dict(X=np.stack([a['v9_margin'],a['candidate_logprob'],a['third_margin']],-1),rows=a['rows'],labels=a['labels'],base=a['base'],candidates=a['candidates'],types=kind(a['labels'],a['base'],a['candidates']),v9_margin=a['v9_margin'],v12_primary=a['margin_plus_soft'])

def evaluate():
 verify();fitverify();models=json.loads((RUN/'checkpoints/models.json').read_text());old=json.loads((ROOT/C['v14']/'checkpoints/rerank_models.json').read_text());thresholds=json.loads((RUN/'artifacts/source_thresholds.json').read_text());cells=[];matches=[];policies=[];curves={}
 for part in ['diagnostic','query']:
  for seed in C['seeds']:
   for sl in C['slices']:
    key=f'{seed}_{sl}';d=target(seed,sl) if part=='query' else read(RUN/'artifacts'/f'features_diagnostic_{key}.npz');scores=values(d,models[key],old[key]['confirmation_primary']);local={}
    for score,v in scores.items():
     met,_,_=curve(d['labels'],d['base'],d['candidates'],v);audit_direct(d['labels'],d['base'],d['candidates'],v,met);local[score]=met;cells.append(dict(partition=part,seed=seed,slice=sl,score=score,**met));curves[f'{part}_{key}_{score}']=prefix(d['types'],v);policies.append(dict(partition=part,seed=seed,slice=sl,score=score,**policy_metrics(d['labels'],d['base'],d['candidates'],v,thresholds[key][score]['threshold'])))
    for baseline in ['v12_primary','IM_all']:
     if baseline not in local:continue
     b=local[baseline]
     for score in scores:
      p=curves[f'{part}_{key}_{score}']
      for mode in ['actual_damage','corrections']:
       allowed=np.flatnonzero(p[:,2]<=b['damaged']) if mode=='actual_damage' else np.flatnonzero((p[:,1]>=b['corrected'])&(p[:,2]/b['frozen_correct']<=.01));j=int(allowed[np.argmax(p[allowed,1])]) if mode=='actual_damage' else (int(allowed[0]) if len(allowed) else None);r=dict(partition=part,seed=seed,slice=sl,score=score,baseline=baseline,mode=mode,feasible=j is not None,baseline_R=b['corrected'],baseline_D=b['damaged'],baseline_W=b['other_wrong_accepted'])
       if j is not None:r.update(R=int(p[j,1]),D=int(p[j,2]),W=int(p[j,3]),R_overshoot=int(p[j,1])-b['corrected'],Recovery=float(p[j,1]/b['frozen_wrong']),NAR=float(p[j,2]/b['frozen_correct']))
       matches.append(r)
    np.savez_compressed(RUN/'artifacts'/f'{part}_scores_{key}.npz',**scores,labels=d['labels'],base=d['base'],candidates=d['candidates'],rows=d['rows'])
 for name,rr in [('cells',cells),('matched_frontiers',matches),('source_policy_target',policies)]:save(name+'.json',rr);csvout(name+'.csv',rr)
 np.savez_compressed(RUN/'artifacts/frontiers.npz',**curves);fitverify()

def audit():
 verify();fitverify();models=json.loads((RUN/'checkpoints/models.json').read_text());old=json.loads((ROOT/C['v14']/'checkpoints/rerank_models.json').read_text());thresholds=json.loads((RUN/'artifacts/source_thresholds.json').read_text());cells=json.loads((RUN/'artifacts/cells.json').read_text());policies=json.loads((RUN/'artifacts/source_policy_target.json').read_text());matches=json.loads((RUN/'artifacts/matched_frontiers.json').read_text());oldcells=json.loads((ROOT/C['v12']/'artifacts/cells.json').read_text());count=0
 for seed in C['seeds']:
  for sl in C['slices']:
   key=f'{seed}_{sl}';cal=read(RUN/'artifacts'/f'features_calibration_{key}.npz');saved=read(RUN/'artifacts'/f'calibration_scores_{key}.npz')
   for score,v in values(cal,models[key],old[key]['confirmation_primary']).items():assert np.array_equal(v,saved[score]) and calibrate(cal,v)==thresholds[key][score]
   for arm,m in models[key].items():
    if m is None:continue
    role='IM' if arm.startswith('IM') else 'GU';d=read(RUN/'artifacts'/f'features_{role}_{key}.npz');mask=np.ones(len(d['rows']),bool) if arm.endswith('_all') else (d['case_id']!=0 if arm.endswith('_aug') else d['case_id']==0);mask &= d['types']>0;u=projection(d['X'][mask],m);assert abs(float(u.mean())-m['u_mean'])<1e-12 and abs(float(u.std())-m['u_sd'])<1e-12
   for part in ['diagnostic','query']:
    d=target(seed,sl) if part=='query' else read(RUN/'artifacts'/f'features_diagnostic_{key}.npz');saved=read(RUN/'artifacts'/f'{part}_scores_{key}.npz')
    for score,v in values(d,models[key],old[key]['confirmation_primary']).items():
     assert np.array_equal(v,saved[score]);met,_,_=curve(d['labels'],d['base'],d['candidates'],v);rr=next(r for r in cells if (r['partition'],r['seed'],r['slice'],r['score'])==(part,seed,sl,score));assert all(rr[k]==vv for k,vv in met.items());pp=next(r for r in policies if (r['partition'],r['seed'],r['slice'],r['score'])==(part,seed,sl,score));direct=policy_metrics(d['labels'],d['base'],d['candidates'],v,pp['threshold']);assert all(pp[k]==vv for k,vv in direct.items());direct=policy_metrics(d['labels'],d['base'],d['candidates'],v,met['threshold']);assert (direct['R'],direct['D'],direct['W'])==(met['corrected'],met['damaged'],met['other_wrong_accepted'])
     if part=='query' and score=='v12_primary':
      expected=next(r for r in oldcells if r['partition']=='query' and r['seed']==seed and r['slice']==sl and r['score']=='margin_plus_soft');assert all(expected[k]==vv for k,vv in met.items())
     for mm in [r for r in matches if r['partition']==part and r['seed']==seed and r['slice']==sl and r['score']==score and r['feasible']]:
      p=prefix(d['types'],v);jj=np.flatnonzero((p[:,1]==mm['R'])&(p[:,2]==mm['D'])&(p[:,3]==mm['W']));assert len(jj);direct=policy_metrics(d['labels'],d['base'],d['candidates'],v,float(p[jj[0],0]));assert (direct['R'],direct['D'],direct['W'])==(mm['R'],mm['D'],mm['W'])
     count+=1
 save('reload_audit.json',dict(passed=True,metric_states=count,source_thresholds_reproduced=True,query_baseline_exact=True,models_source_normalization_reproduced=True,all_prediction_counts_recomputed=True,all_fit_hashes_unchanged=True))

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:save(f'failure_{sys.argv[1]}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
