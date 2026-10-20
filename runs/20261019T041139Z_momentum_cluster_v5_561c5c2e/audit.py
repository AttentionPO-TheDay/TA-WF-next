"""Independent metric reconstruction and complete saved-checkpoint query reload."""
import os,json,time
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,sha,save,load_model,load_slice
from methods import tokenize,evaluate,attach,assert_base_locked
from candidate import gpu_dict
from routing import geometry,scores


def mm(y,p,b):
    mat=np.zeros((102,102),dtype=np.int64);np.add.at(mat,(y,p),1);den=mat.sum(0)+mat.sum(1);f=np.divide(2*mat.diagonal(),den,out=np.zeros(102),where=den>0).mean();correct=b==y;wrong=~correct
    return {'macro_f1':float(f),'accuracy':float((p==y).mean()),'nar':float((p[correct]!=y[correct]).mean()),'recovery':float((p[wrong]==y[wrong]).mean())}


def main():
    setup();verify();seed=int(os.environ['CANDIDATE_SEED']);begin=time.monotonic();source=torch.load(ROOT/C['v4_run']/'artifacts'/f'source_{seed}.pt',map_location='cpu',weights_only=True);vt=source['valid_tokens'].cuda();vy=source['valid_labels'].numpy();vb=source['valid_logits'].argmax(-1).numpy()
    art=torch.load(ROOT/C['previous_run']/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True);p=gpu_dict(art['fits']['1']);rows=json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text());records=[]
    for name in C['slices']:
        ref,query=load_slice(name);m=load_model(seed);qt=tokenize(m,query);qe,qz=evaluate(m,qt);g=geometry(qe,p);y=query['labels'].numpy();bp=qz.argmax(-1).cpu().numpy()
        with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as a:assert np.array_equal(a['logits'],qz.cpu().numpy())
        for condition in C['conditions']:
            row=next(r for r in rows if r['slice']==name and r['condition']==condition);ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';state=torch.load(ck,map_location='cpu',weights_only=True);assert sha(ck)==row['checkpoint_sha256'];assert state['config_sha256']==sha(RUN/'config.json')
            memory=state['reference_state'];assert memory['steps']==400 and memory['features'].shape==(1024,128) and memory['probabilities'].shape==(1024,102);assert memory['prototype'].shape==(102,128)
            assert torch.allclose(memory['probabilities'].sum(1),torch.ones(1024),atol=1e-5)
            dynamic=condition in ['momentum_proto','momentum_joint'];assert dynamic or torch.equal(memory['prototype'],memory['source'])
            for k in ['prototype','features','probabilities','mean','second']:assert torch.isfinite(memory[k]).all()
            m=load_model(seed);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};a=attach(m,'lora');a.load_state_dict(state['adapter']);_,z=evaluate(m,qt,a);_,vz=evaluate(m,vt,a);assert_base_locked(m,initial)
            raw=RUN/'artifacts'/f'raw_{seed}_{name}_{condition}.npz';assert sha(raw)==row['raw_sha256']
            with np.load(raw) as f:assert np.array_equal(f['logits'],z.cpu().numpy()) and np.array_equal(f['rows'],query['rows'].numpy()) and np.array_equal(f['labels'],y)
            with np.load(RUN/'artifacts'/f'valid_{seed}_{name}_{condition}.npz') as f:assert np.array_equal(f['logits'],vz.cpu().numpy())
            pred=z.argmax(-1).cpu().numpy();validpred=vz.argmax(-1).cpu().numpy();met=mm(y,pred,bp);vm=mm(vy,validpred,vb)
            for k in met:assert abs(met[k]-row[k])<1e-12
            assert abs(row['delta_f1_pp']-100*(met['macro_f1']-mm(y,bp,bp)['macro_f1']))<1e-10
            for k in vm:assert abs(vm[k]-row['source_valid_'+k])<1e-12
            sc=scores(qz,z,g,source['calibration'])
            with np.load(RUN/'artifacts'/f'signal_{seed}_{name}_{condition}.npz') as f:
                for key in sc:assert np.array_equal(f[key],sc[key].cpu().numpy())
            hist=json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{condition}.json').read_text());assert len(hist)==400
            assert all(np.isfinite(h['loss']) and np.isfinite(h['gradient_norm']) and h['weight_min']>=.05-1e-7 and h['weight_max']<=1+1e-7 for h in hist)
            if not dynamic:assert all(h['prototype_drift_mean']==0 for h in hist)
            if condition not in ['momentum_contrast','momentum_joint']:assert all(h['contrast']==0 for h in hist)
            assert not row['output_bypass'];records.append({'slice':name,'condition':condition,'query_full_reload':True,'source_valid_full_reload':True,'metrics_independent':True,'reference_memory_only':True})
            print(f'audit {seed} {name} {condition} {len(records)}/10',flush=True)
        del qt,qe,qz,g,query,ref;torch.cuda.empty_cache()
    save(f'audit_{seed}.json',{'passed':True,'seed':seed,'query_checkpoints_reloaded':len(records),'source_valid_checkpoints_reloaded':len(records),'seconds':time.monotonic()-begin,'records':records})

if __name__=='__main__':main()
