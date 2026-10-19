"""Full checkpoint reinference, independent metric recomputation and exact bypass proof."""
import json,os,time
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,sha,save,load_model,load_slice,metrics
from methods import tokenize,evaluate,attach,route,assert_base_locked

def main():
    setup();verify();start=time.monotonic();records=[]
    scores=json.loads((RUN/'artifacts/scores.json').read_text());assert len(scores)==240
    assert len({(r['seed'],r['slice'],r['condition']) for r in scores})==240
    for seed in C['seeds']:
        art=torch.load(RUN/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True)
        ps={k:{n:v.cuda() for n,v in p.items()} for k,p in art['fits'].items()}
        for name in C['slices']:
            _,query=load_slice(name);m=load_model(seed);qt=tokenize(m,query);qe,qz=evaluate(m,qt)
            base=qz.argmax(-1).cpu().numpy();y=query['labels'].numpy()
            with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as b:
                assert np.array_equal(b['logits'],qz.cpu().numpy());assert np.array_equal(b['labels'],y)
            for condition in C['conditions']:
                r=next(r for r in scores if (r['seed'],r['slice'],r['condition'])==(seed,name,condition))
                key=f'{seed}_{name}_{condition}';pf=RUN/'artifacts'/f'pred_{key}.npz';cf=RUN/'checkpoints'/f'{key}.pt'
                assert sha(pf)==r['prediction_sha256'] and sha(cf)==r['checkpoint_sha256']
                state=torch.load(cf,weights_only=True,map_location='cpu');assert state['config_sha256']==sha(RUN/'config.json')
                assert state['source_gmm_sha256']==sha(RUN/'artifacts'/f'source_gmm_{seed}.pt')
                m=load_model(seed);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                adapter=attach(m,condition.split('_')[-1]);adapter.load_state_dict(state['adapter'],strict=True)
                p=ps['4' if condition.startswith('gmm4') else '1'];_,keep,_,_=route(qe,qz,p)
                e,z=evaluate(m,qt,adapter)
                if '_routed_' in condition:z=torch.where(keep[:,None],z,qz)
                assert_base_locked(m,initial)
                with np.load(pf) as v:
                    assert np.array_equal(v['logits'],z.cpu().numpy()),key
                    assert np.array_equal(v['predictions'],z.argmax(-1).cpu().numpy())
                    assert np.array_equal(v['labels'],y) and np.array_equal(v['rows'],query['rows'].numpy())
                    bypass=v['bypass'];assert np.array_equal(v['logits'][bypass],qz.cpu().numpy()[bypass])
                    if '_routed_' in condition:assert np.array_equal(bypass,~keep.cpu().numpy())
                    mm=metrics(y,v['predictions'],base)
                    for metric in ['accuracy','macro_f1','nar','recovery']:assert abs(mm[metric]-r[metric])<1e-12,(key,metric)
                    assert abs(100*(mm['macro_f1']-metrics(y,base,base)['macro_f1'])-r['delta_f1_pp'])<1e-10
                records.append({'key':key,'checkpoint_reload_full_query':True,'metrics_recomputed':True,'bypass_bitwise':True,'base_locked':True})
                del m,adapter,z,e;torch.cuda.empty_cache()
            del qt,qe,qz;print(f'audit seed{seed} {name}: {len(records)}/240',flush=True)
    save('postrun_audit.json',{'passed':True,'cells':len(records),'full_query_reload':True,'seconds':time.monotonic()-start,'records':records})

if __name__=='__main__':main()
