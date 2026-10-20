import os,time,json,traceback
import numpy as np
import torch
from runtime import RUN,ROOT,C,save,sha,verify,setup,valid,roles,old_logits
from model import ViewNet,features
from perturb import tam,transform,predict
def load(seed,arm):
    p=ROOT/C['v10_run']/'checkpoints'/f'{seed}_{arm}.pt';state=torch.load(p,weights_only=True,map_location='cpu');assert state['seed']==seed and state['arm']==arm
    m=ViewNet().cuda().eval().requires_grad_(False);m.load_state_dict(state['model']);return m
def preflight(seed):
    d=valid();small={k:v[:64] for k,v in d.items()};assert torch.equal(tam(small['timestamps']),small['tam'])
    for view in C['views']:
        a=transform(small,view);aa=transform({k:v[:1] for k,v in small.items()},view);assert torch.equal(a['timestamps'][:1],aa['timestamps']) and torch.equal(a['tam'][:1],aa['tam']);assert (a['tam'].sum((1,2))==(a['timestamps']!=0).sum(1)).all()
        if view in ['time095','time105']:assert torch.equal(features(small,'direction_order'),features(a,'direction_order'))
        if view=='thin01':
            for old,new in zip(small['timestamps'],a['timestamps']):
                orig=old[old!=0].tolist();subset=new[new!=0].tolist();pos=0
                for t in subset:
                    while orig[pos]!=t:pos+=1
                    pos+=1
                assert len(subset)<=len(orig) and subset[0]==orig[0]
        if view=='tail98':assert torch.equal((a['timestamps']!=0).sum(1),((small['timestamps']!=0).sum(1)*98+99)//100)
    for arm in C['arms']:
        m=load(seed,arm);z=predict(m,d,arm,'original');a=old_logits(seed,arm,'valid');assert np.array_equal(z,a['logits']) and np.array_equal(a['rows'],d['rows'].numpy())
        z=predict(m,small,arm,'time095');one=predict(m,{k:v[:1] for k,v in small.items()},arm,'time095');assert np.allclose(z[:1],one,atol=2e-5,rtol=2e-5);assert not any(p.requires_grad for p in m.parameters());del m
    save(f'preflight_{seed}.json',dict(passed=True,sourcevalid_original_exact=True,transform_batch_independent=True,tam_exact=True,direction_time_scale_invariant=True,models_locked=True))
def run(seed,audit):
    start=time.monotonic();n=0;identities=0
    for arm in C['arms']:
        m=load(seed,arm);before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        for role,d in roles():
            if time.monotonic()-start>C['budget']['max_cell_seconds']:raise TimeoutError('worker budget')
            if audit:
                z=predict(m,d,arm,'original');old=old_logits(seed,arm,role);assert np.array_equal(z,old['logits']) and np.array_equal(d['rows'].numpy(),old['rows']);identities+=1
            for view in C['views'][1:]:
                z=predict(m,d,arm,view);p=RUN/'artifacts'/f'logits_{seed}_{arm}_{role}_{view}.npz'
                if audit:
                    a=__import__('runtime').read(p);assert np.array_equal(a['logits'],z) and np.array_equal(a['rows'],d['rows'].numpy())
                    if role=='valid' or role.endswith('_query'):assert np.array_equal(a['labels'],d['labels'].numpy())
                else:
                    a=dict(logits=z,rows=d['rows'].numpy())
                    if role=='valid' or role.endswith('_query'):a['labels']=d['labels'].numpy()
                    np.savez_compressed(p,**a)
                n+=1
            print(json.dumps(dict(seed=seed,arm=arm,role=role,phase='audit' if audit else 'infer',augmented_states=n,original_identity_checks=identities,seconds=time.monotonic()-start)),flush=True)
            save(f'progress_{seed}.json',dict(phase='audit' if audit else 'infer',states=n,total=60,seconds=time.monotonic()-start))
        assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items());del m;torch.cuda.empty_cache()
    save(f'{"audit" if audit else "inference"}_{seed}.json',dict(passed=True,augmented_states=n,original_identity_states=identities,models_locked=True,seconds=time.monotonic()-start));assert n==60
if __name__=='__main__':
    seed=int(os.environ['VIEW_SEED']);mode=os.environ['VIEW_MODE']
    try:
        setup(seed);verify();assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
        if mode=='preflight':preflight(seed)
        else:run(seed,mode=='audit')
    except BaseException as e:save(f'failure_{mode}_{seed}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
