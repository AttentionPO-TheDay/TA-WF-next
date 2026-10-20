"""Complete query checkpoint reinference and independent policy/metric checks."""
import json,time
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,save,sha,load_model,load_slice,predict,configure,metrics
from methods import tokenize,evaluate,attach,assert_base_locked
from candidate import gpu_dict,banks,acceptance_inputs,policy_logits
from tent import check_tent_scope

def main():
    setup();verify();start=time.monotonic();scores=json.loads((RUN/'artifacts/scores.json').read_text());assert len(scores)==630
    assert len({(r['seed'],r['slice'],r['condition'],r['policy']) for r in scores})==630
    records=[];backend=[]
    for seed in C['seeds']:
        old=ROOT/C['previous_run'];art=torch.load(old/'artifacts'/f'source_gmm_{seed}.pt',weights_only=True,map_location='cpu');p=gpu_dict(art['fits']['1']);bank_data=banks(art,p)
        data=torch.load(RUN/'artifacts'/f'source_calibration_{seed}.pt',weights_only=True,map_location='cpu');cal=gpu_dict(data['calibration'])
        for name in C['slices']:
            _,query=load_slice(name);m=load_model(seed);qt=tokenize(m,query);qe,qz=evaluate(m,qt);base=qz.argmax(-1).cpu().numpy();y=query['labels'].numpy();info=acceptance_inputs(qe,qz,p,bank_data,cal)
            with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as b:assert np.array_equal(b['logits'],qz.cpu().numpy())
            with np.load(old/'artifacts'/f'base_{seed}_{name}.npz') as b:
                assert np.array_equal(b['rows'],query['rows'].numpy());assert np.array_equal(b['labels'],y)
                backend.append({'seed':seed,'slice':name,'v2_prediction_differences':int((b['predictions']!=base).sum()),'v2_max_abs_logit_difference':float(np.abs(b['logits']-qz.cpu().numpy()).max())})
            for condition in C['conditions']:
                m=load_model(seed);original={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};adapter=None
                if condition=='frozen':z=qz
                else:
                    ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';state=torch.load(ck,weights_only=True,map_location='cpu');assert state['config_sha256']==sha(RUN/'config.json')
                    assert state['calibration_sha256']==sha(RUN/'artifacts'/f'source_calibration_{seed}.pt')
                    if condition.startswith('tent_bn'):
                        if condition=='tent_bn':configure(m,'tent_bn')
                        m.load_state_dict(state['model'],strict=True);z=torch.from_numpy(predict(m,query)).cuda();check_tent_scope(m,original,condition)
                    else:
                        adapter=attach(m,condition.rsplit('_',1)[1]);adapter.load_state_dict(state['adapter'],strict=True);_,z=evaluate(m,qt,adapter);assert_base_locked(m,original)
                assert torch.isfinite(z).all()
                raw=RUN/'artifacts'/f'raw_{seed}_{name}_{condition}.npz'
                with np.load(raw) as v:assert np.array_equal(v['logits'],z.cpu().numpy()),(seed,name,condition)
                rows=[r for r in scores if (r['seed'],r['slice'],r['condition'])==(seed,name,condition)]
                assert len(rows)==(3 if adapter is not None else 1)
                for r in rows:
                    policy=r['policy'];zz,keep=policy_logits(z,qz,info,cal,policy);pred=zz.argmax(-1).cpu().numpy();pp=RUN/'artifacts'/f'policy_{seed}_{name}_{condition}_{policy}.npz'
                    assert sha(pp)==r['policy_sha256'] and sha(raw)==r['raw_sha256']
                    if condition!='frozen':assert sha(RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt')==r['checkpoint_sha256']
                    with np.load(pp) as v:
                        assert np.array_equal(v['predictions'],pred) and np.array_equal(v['keep'],keep.cpu().numpy());assert np.array_equal(v['rows'],query['rows'].numpy())
                    mm=metrics(y,pred,base)
                    for metric in ['macro_f1','accuracy','nar','recovery']:assert abs(mm[metric]-r[metric])<1e-12
                    assert abs(100*(mm['macro_f1']-metrics(y,base,base)['macro_f1'])-r['delta_f1_pp'])<1e-10
                    if policy!='raw':assert torch.equal(zz[~keep],qz[~keep])
                    records.append({'seed':seed,'slice':name,'condition':condition,'policy':policy,'full_query_reload':True,'metrics_recomputed':True,'rejected_logits_bitwise':True})
                del m,adapter,z;torch.cuda.empty_cache()
            print(f'audit {seed} {name}: {len(records)}/630 states',flush=True)
            del qt,qe,qz,query,info;torch.cuda.empty_cache()
    assert all(r['v2_prediction_differences']==0 for r in backend)
    save('postrun_audit.json',{'passed':True,'states':len(records),'raw_checkpoints_reloaded':240,'frozen_inferences_repeated':30,'seconds':time.monotonic()-start,'records':records,'frozen_v2_comparison':backend})

if __name__=='__main__':main()
