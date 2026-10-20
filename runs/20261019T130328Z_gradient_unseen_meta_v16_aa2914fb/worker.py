from inference import *

def load():
 seed=int(os.environ['SOURCE_SEED']);setup(seed);verify()
 for p,h in json.loads((RUN/'artifacts/role_freeze.json').read_text()).items():assert sha(ROOT/p)==h
 raw=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',weights_only=True,map_location='cpu')['valid'];roles=json.loads((RUN/'artifacts/roles.json').read_text());idx=np.array(roles['valid_pool']['indices']);data={k:v[idx] for k,v in raw.items()};models={'frozen':main_model(seed),**{sl:main_model(seed,sl) for sl in C['slices']}};return seed,raw,data,models,aux_model(seed)

def preflight():
 seed,raw,data,models,aux=load();assert torch.equal(tam(raw['timestamps']),raw['tam'])
 for name,m in models.items():
  z=main_predict(m,raw)
  for sl in C['slices'] if name=='frozen' else [name]:
   old=read(ROOT/C['v9']/'artifacts'/f'valid_{seed}_{sl}_{"frozen" if name=="frozen" else "smooth_balance"}.npz');assert np.array_equal(z,old['logits']);assert np.array_equal(raw['rows'].numpy(),old['rows'])
 z=aux_predict(aux,raw,'original');old=read(ROOT/C['v10']/'artifacts'/f'sourcevalid_{seed}_direction_iat.npz');assert np.array_equal(z,old['logits']);assert all(not p.requires_grad for m in list(models.values())+[aux] for p in m.parameters())
 small={k:v[:8] for k,v in data.items()}
 for ci,case in enumerate(C['perturbations']):
  d=perturb(small,case,ci,'valid');one=perturb({k:v[:1] for k,v in small.items()},case,ci,'valid');assert torch.equal(d['tam'][:1],one['tam']);assert torch.equal(d['timestamps'][:1],one['timestamps']);assert torch.isfinite(d['tam']).all()
 save(f'preflight_{seed}.json',dict(passed=True,full_valid_main_states=3,full_valid_aux_states=1,logits_bitwise_identical=True,models_locked=True))

def clean(seed,raw,data):
 ix=json.loads((RUN/'artifacts/roles.json').read_text())['valid_pool']['indices'];out={k:data[k].numpy() for k in ['rows','labels']};base=read(ROOT/C['v9']/'artifacts'/f'valid_{seed}_{C["slices"][0]}_frozen.npz');assert np.array_equal(base['rows'],raw['rows'].numpy());out['frozen_logits']=base['logits'][ix]
 for sl in C['slices']:
  old=read(ROOT/C['v9']/'artifacts'/f'valid_{seed}_{sl}_smooth_balance.npz');assert np.array_equal(old['rows'],raw['rows'].numpy());out[sl+'_logits']=old['logits'][ix];out[sl+'_margin']=old['margin_gain'][ix]
 zz=[]
 for view in C['views']:
  p=ROOT/C['v10']/'artifacts'/f'sourcevalid_{seed}_direction_iat.npz' if view=='original' else ROOT/C['v11']/'artifacts'/f'logits_{seed}_direction_iat_valid_{view}.npz';old=read(p);assert np.array_equal(old['rows'],raw['rows'].numpy());zz.append(old['logits'][ix])
 out['aux_logits']=np.stack(zz);return out

def compute(audit=False):
 start=time.monotonic();seed,raw,data,models,aux=load();snap=[{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in list(models.values())+[aux]];checks=0
 for ci,case in enumerate(C['perturbations']):
  assert time.monotonic()-start<C['budget']['worker_seconds'];dd={k:v[:256] for k,v in data.items()} if audit else data
  if ci==0:
   result=clean(seed,raw,data)
   if audit:result={k:v[:,:256] if k=='aux_logits' else v[:256] for k,v in result.items()}
  else:
   dd=perturb(dd,case,ci,'valid');result={k:dd[k].numpy() for k in ['rows','labels']};zs={k:main_predict(m,dd) for k,m in models.items()};result['frozen_logits']=zs['frozen']
   for sl in C['slices']:
    result[sl+'_logits']=zs[sl];c=zs[sl].argmax(-1);result[sl+'_margin']=margin(zs[sl],c)-margin(zs['frozen'],c)
   result['aux_logits']=np.stack([aux_predict(aux,dd,view) for view in C['views']])
  path=RUN/'artifacts'/f'validpool_{seed}_{case["name"]}.npz'
  if audit:
   old=read(path)
   for k,v in result.items():assert np.array_equal(v,old[k][:,:256] if k=='aux_logits' else old[k][:256]),(seed,case['name'],k)
  else:
   assert all(np.isfinite(v).all() for v in result.values());np.savez_compressed(path,**result)
  checks+=1;print(json.dumps(dict(seed=seed,case=case['name'],audit=audit,rows=len(dd['rows']))),flush=True)
 for m,before in zip(list(models.values())+[aux],snap):assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items())
 save(f'{"cache_audit" if audit else "forward"}_{seed}.json',dict(passed=True,states=checks,rows_per_probe=256 if audit else len(data['rows']),models_and_buffers_unchanged=True,seconds=time.monotonic()-start))

if __name__=='__main__':
 import sys,traceback
 try:{'preflight':preflight,'compute':compute,'audit':lambda:compute(True)}[sys.argv[1]]()
 except BaseException as e:save(f'failure_{sys.argv[1]}_{os.environ["SOURCE_SEED"]}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
