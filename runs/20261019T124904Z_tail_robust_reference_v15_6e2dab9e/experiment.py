from runtime import *
from methods import *
from metrics import curve,audit_direct,preflight
import sys,traceback

def freeze():
 inputs=[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md']+list(RUN.glob('*.py'))
 for seed in C['seeds']:
  for sl in C['slices']:
   for role in ['train','valid']:inputs.append(ROOT/C['v14']/'artifacts'/f'source_features_{role}_{seed}_{sl}.npz')
   inputs.append(ROOT/C['v13']/'artifacts'/f'rows_query_{seed}_{sl}.npz')
   inputs.append(ROOT/C['v9']/'artifacts'/f'reference_{seed}_{sl}_smooth_balance.npz')
   for ss in C['seeds']:
    inputs.append(ROOT/C['v10']/'artifacts'/f'reference_{ss}_{sl}_direction_iat.npz')
    for view in C['views'][1:]:inputs.append(ROOT/C['v11']/'artifacts'/f'logits_{ss}_direction_iat_{sl}_reference_{view}.npz')
 for pp in [ROOT/C['source_run']/'artifacts/prepared.pt',ROOT/C['source_run']/'artifacts/data_preflight.json',ROOT/C['v14']/'checkpoints/rerank_models.json',ROOT/C['v12']/'artifacts/source_calibration.json',ROOT/C['v12']/'artifacts/cells.json',ROOT/'PROTOCOL.md',ROOT/'configs/datasets.json']:inputs.append(pp)
 save('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(inputs))});save('freeze_metadata.json',dict(files=len(set(inputs)),query_arrays_opened=False,permission='only hashes for query inputs before fit',user_authorized_ABC=True))

def source_content():
 import torch
 original=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',weights_only=True,map_location='cpu');maps={}
 for role,part in [('train','source'),('valid','valid')]:
  d=original[part];maps[role]={int(row):hashlib.sha256(x.numpy().tobytes()).hexdigest() for row,x in zip(d['rows'],d['timestamps'])}
 assert set(maps['train'].values()).isdisjoint(maps['valid'].values())
 save('source_content_audit.json',dict(passed=True,train_rows=len(maps['train']),valid_rows=len(maps['valid']),content_overlap=0,row_id_scope='file-relative; role-qualified',model_selection_roles_unchanged=True))
 return maps

def oldmodels():return json.loads((ROOT/C['v14']/'checkpoints/rerank_models.json').read_text())

def fit():
 verify();save('metric_preflight.json',preflight());content=source_content();models={};thresholds={};densities={};infos=[];lopo=[];locomodels={};fusion=[];old=oldmodels();nfit=0
 for seed in C['seeds']:
  for sl in C['slices']:
   key=f'{seed}_{sl}';tr=read(ROOT/C['v14']/'artifacts'/f'source_features_train_{key}.npz');va=read(ROOT/C['v14']/'artifacts'/f'source_features_valid_{key}.npz');tr['trace_groups']=np.array([content['train'][int(row)] for row in tr['rows']]);assert set(tr['trace_groups']).isdisjoint(content['valid'].values());ref=reference_features(seed,sl);np.savez_compressed(RUN/'artifacts'/f'reference_features_{key}.npz',**ref)
   dm,di=density(tr,ref);densities[key]=dm;infos.append(dict(seed=seed,slice=sl,**di));models[key]={}
   for arm in C['arms']:
    if 'C' in arm and not di['feasible']:models[key][arm]=None;continue
    w=dm['source_changed_weights'] if 'C' in arm else None;m=fitmodel(tr,arm,w,seed,check=True);models[key][arm]=m;nfit+=1
    if arm=='ce':
     expected=old[key]['confirmation_primary'];err=float(np.max(np.abs(project(tr['X'],m)-project(tr['X'],expected))));assert err<1e-4,err;save(f'ce_reproduction_{key}.json',dict(passed=True,max_branch_difference=err))
    print(json.dumps(dict(stage='fit',key=key,arm=arm,iterations=m['iterations'],models=nfit)),flush=True)
   ss=values(va,models[key],old[key]['confirmation_primary']);thresholds[key]={k:calibrate(va,v) for k,v in ss.items()};np.savez_compressed(RUN/'artifacts'/f'source_scores_valid_{key}.npz',**ss)
   for arm,m in models[key].items():
    if m is None:continue
    u=project(va['X'],m)
    for kind,v in [('branch',u),('fused',ss[arm])]:
     met,_,_=curve(va['labels'],va['base'],va['candidates'],v);fusion.append(dict(partition='sourcevalid',seed=seed,slice=sl,arm=arm,kind=kind,**met))
   for heldout in range(5):
    td=subset(tr,tr['case_id']!=heldout);vd=subset(va,va['case_id']==heldout)
    for arm in C['lopo']['arms']:
     m=fitmodel(td,arm,None,seed);locomodels[f'{key}_{heldout}_{arm}']=m;nfit+=1;v=vd['v12_primary']+.25*(project(vd['X'],m)-m['u_mean'])/m['u_sd'];met,_,_=curve(vd['labels'],vd['base'],vd['candidates'],v);lopo.append(dict(seed=seed,slice=sl,heldout=heldout,arm=arm,**met))
   print(json.dumps(dict(stage='state_complete',key=key,models=nfit,density=di)),flush=True)
 (RUN/'checkpoints/models.json').write_text(json.dumps(models,indent=2,allow_nan=False)+'\n');(RUN/'checkpoints/density.json').write_text(json.dumps(densities,indent=2,allow_nan=False)+'\n');(RUN/'checkpoints/lopo_models.json').write_text(json.dumps(locomodels,indent=2,allow_nan=False)+'\n')
 save('source_thresholds.json',thresholds);save('density_diagnostics.json',infos);save('lopo.json',lopo);csvout('lopo.csv',lopo);save('fusion_source.json',fusion);save('source_fit_complete.json',dict(passed=True,rerank_models=nfit,density_models=sum(i.get('models',0) for i in infos),query_opened=False,source_train_rows=2448,source_valid_rows=816));assert nfit<=168
 paths=list((RUN/'checkpoints').glob('*.json'))+[RUN/'artifacts/source_thresholds.json']+list((RUN/'artifacts').glob('source_scores_*.npz'))
 save('source_fit_freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});status('running',f'A/B/C源拟合完成：{nfit}重排模型，阈值已冻结；开始终端query评分。')

def fitverify():
 for p,h in json.loads((RUN/'artifacts/source_fit_freeze.json').read_text()).items():assert sha(ROOT/p)==h,p

def target(seed,sl):
 old=read(ROOT/C['v13']/'artifacts'/f'rows_query_{seed}_{sl}.npz');return dict(X=np.stack([old['v9_margin'],old['candidate_logprob'],old['third_margin']],-1),v9_margin=old['v9_margin'],v12_primary=old['margin_plus_soft'],labels=old['labels'],base=old['base'],candidates=old['candidates'],rows=old['rows'],types=old['type'])

def evaluate():
 verify();fitverify();models=json.loads((RUN/'checkpoints/models.json').read_text());old=oldmodels();thresholds=json.loads((RUN/'artifacts/source_thresholds.json').read_text());cells=[];matches=[];policies=[];curves={};fusion=[]
 for seed in C['seeds']:
  for sl in C['slices']:
   key=f'{seed}_{sl}';d=target(seed,sl);ref=read(RUN/'artifacts'/f'reference_features_{key}.npz');assert set(ref['rows']).isdisjoint(d['rows']);ss=values(d,models[key],old[key]['confirmation_primary']);local={}
   for score,v in ss.items():
    met,_,_=curve(d['labels'],d['base'],d['candidates'],v);audit_direct(d['labels'],d['base'],d['candidates'],v,met);cells.append(dict(seed=seed,slice=sl,score=score,**met));local[score]=met;curves[f'{key}_{score}']=prefix(d['types'],v);policies.append(dict(seed=seed,slice=sl,score=score,**policy(d['labels'],d['base'],d['candidates'],v,thresholds[key][score]['threshold'])))
   b=local['v12_primary']
   for score in ss:
    p=curves[f'{key}_{score}']
    for mode in ['v12_actual_damage','v12_corrections']:
     ix=np.flatnonzero(p[:,2]<=b['damaged']) if mode=='v12_actual_damage' else np.flatnonzero((p[:,1]>=b['corrected'])&(p[:,2]/b['frozen_correct']<=.01));j=int(ix[np.argmax(p[ix,1])]) if mode=='v12_actual_damage' else (int(ix[0]) if len(ix) else None);r=dict(seed=seed,slice=sl,score=score,mode=mode,feasible=j is not None,baseline_R=b['corrected'],baseline_D=b['damaged'],baseline_W=b['other_wrong_accepted'])
     if j is not None:r.update(R=int(p[j,1]),D=int(p[j,2]),W=int(p[j,3]),R_overshoot=int(p[j,1])-b['corrected'],Recovery=float(p[j,1]/b['frozen_wrong']),NAR=float(p[j,2]/b['frozen_correct']))
     matches.append(r)
   for arm,m in models[key].items():
    if m is None:continue
    met,_,_=curve(d['labels'],d['base'],d['candidates'],project(d['X'],m));fusion.append(dict(partition='query',seed=seed,slice=sl,arm=arm,kind='branch',**met));fusion.append(dict(partition='query',seed=seed,slice=sl,arm=arm,kind='fused',**local[arm]))
   np.savez_compressed(RUN/'artifacts'/f'query_scores_{key}.npz',**ss,labels=d['labels'],base=d['base'],candidates=d['candidates'],rows=d['rows'])
 for n,rr in [('cells',cells),('matched_frontiers',matches),('source_policy_target',policies),('fusion_target',fusion)]:save(n+'.json',rr);csvout(n+'.csv',rr)
 np.savez_compressed(RUN/'artifacts/frontiers.npz',**curves);fitverify()

def audit():
 verify();fitverify();models=json.loads((RUN/'checkpoints/models.json').read_text());old=oldmodels();thresholds=json.loads((RUN/'artifacts/source_thresholds.json').read_text());cells=json.loads((RUN/'artifacts/cells.json').read_text());policies=json.loads((RUN/'artifacts/source_policy_target.json').read_text());matched=json.loads((RUN/'artifacts/matched_frontiers.json').read_text());count=0;source=0;baseline=json.loads((ROOT/C['v12']/'artifacts/cells.json').read_text())
 for seed in C['seeds']:
  for sl in C['slices']:
   key=f'{seed}_{sl}';va=read(ROOT/C['v14']/'artifacts'/f'source_features_valid_{key}.npz');saved=read(RUN/'artifacts'/f'source_scores_valid_{key}.npz')
   for score,v in values(va,models[key],old[key]['confirmation_primary']).items():
    assert np.array_equal(v,saved[score]);assert calibrate(va,v)==thresholds[key][score];source+=1
   d=target(seed,sl);ss=values(d,models[key],old[key]['confirmation_primary']);saved=read(RUN/'artifacts'/f'query_scores_{key}.npz');ref=reference_features(seed,sl);rs=read(RUN/'artifacts'/f'reference_features_{key}.npz');assert all(np.array_equal(v,rs[k]) for k,v in ref.items())
   for score,v in ss.items():
    assert np.array_equal(v,saved[score]);met,_,_=curve(d['labels'],d['base'],d['candidates'],v);record=next(r for r in cells if (r['seed'],r['slice'],r['score'])==(seed,sl,score));assert all(record[k]==vv for k,vv in met.items());pp=next(r for r in policies if (r['seed'],r['slice'],r['score'])==(seed,sl,score));direct=policy(d['labels'],d['base'],d['candidates'],v,pp['threshold']);assert all(pp[k]==vv for k,vv in direct.items())
    actual=policy(d['labels'],d['base'],d['candidates'],v,met['threshold']);assert (actual['R'],actual['D'],actual['W'])==(met['corrected'],met['damaged'],met['other_wrong_accepted'])
    if score=='v12_primary':
     bb=next(r for r in baseline if r['partition']=='query' and r['seed']==seed and r['slice']==sl and r['score']=='margin_plus_soft');assert all(bb[k]==vv for k,vv in met.items())
    for mm in [r for r in matched if r['seed']==seed and r['slice']==sl and r['score']==score]:
     if mm['feasible']:
      p=prefix(d['types'],v);found=np.flatnonzero((p[:,1]==mm['R'])&(p[:,2]==mm['D'])&(p[:,3]==mm['W']));assert len(found);rr=policy(d['labels'],d['base'],d['candidates'],v,float(p[found[0],0]));assert (rr['R'],rr['D'],rr['W'])==(mm['R'],mm['D'],mm['W'])
    count+=1
 # Entire LOPO model reload on held-out valid, not merely final models.
 lm=json.loads((RUN/'checkpoints/lopo_models.json').read_text());lr=json.loads((RUN/'artifacts/lopo.json').read_text())
 for rr in lr:
  key=f'{rr["seed"]}_{rr["slice"]}';va=read(ROOT/C['v14']/'artifacts'/f'source_features_valid_{key}.npz');vd=subset(va,va['case_id']==rr['heldout']);m=lm[f'{key}_{rr["heldout"]}_{rr["arm"]}'];v=vd['v12_primary']+.25*(project(vd['X'],m)-m['u_mean'])/m['u_sd'];met,_,_=curve(vd['labels'],vd['base'],vd['candidates'],v);assert all(rr[k]==vv for k,vv in met.items())
 save('reload_audit.json',dict(passed=True,query_states=count,source_states=source,lopo_states=len(lr),source_thresholds_reproduced=True,query_predictions_and_counts_recomputed=True,reference_features_rebuilt=True,v12_baseline_exact=True,source_fit_hashes_unchanged=True))

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:save('failure_'+sys.argv[1]+'.json',dict(error=str(e),traceback=traceback.format_exc()));raise
