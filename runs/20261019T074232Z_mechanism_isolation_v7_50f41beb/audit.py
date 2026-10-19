"""Independent confusion metrics and full54 state raw/valid saved-model reload."""
import os,json,time
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,sha,save,load_model,load_slice
from methods import tokenize,evaluate,attach,assert_base_locked
from routing import geometry,scores
from worker import inputs,scope,canonical,old_adapter
from isolation import capture,probe_summary


def mm(y,p,b):
    mat=np.zeros((102,102),dtype=np.int64);np.add.at(mat,(y,p),1);den=mat.sum(0)+mat.sum(1)
    f=np.divide(2*mat.diagonal(),den,out=np.zeros(102),where=den>0).mean();correct=b==y;wrong=~correct
    return dict(macro_f1=float(f),accuracy=float((p==y).mean()),nar=float((p[correct]!=y[correct]).mean()),recovery=float((p[wrong]==y[wrong]).mean()))


def main():
    setup();verify();seed=int(os.environ['CANDIDATE_SEED']);begin=time.monotonic();p,bb,prior,source,raw,valid=inputs(seed)
    vy=source['valid_labels'].numpy();vb=source['valid_logits'].argmax(-1).numpy();rows=json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text());records=[]
    for name in C['slices']:
        ref,query=load_slice(name);m=load_model(seed);pure=canonical(m);qt=tokenize(m,query);qe,qz=evaluate(m,qt);g=geometry(qe,p);y=query['labels'].numpy();bp=qz.argmax(-1).cpu().numpy()
        with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as f:assert np.array_equal(f['logits'],qz.cpu().numpy())
        # Reference scoring was computed from exactly the frozen v6 candidate.
        rt=tokenize(m,ref);re,rz=evaluate(m,rt);rg=geometry(re,p);a=attach(m,'lora');a.load_state_dict(old_adapter(seed,name));_,rr=evaluate(m,rt,a);ss=scores(rz,rr,rg,source['calibration'])
        with np.load(RUN/'artifacts'/f'reference_signal_{seed}_{name}.npz') as f:
            assert np.array_equal(f['candidate_logits'],rr.cpu().numpy()) and np.array_equal(f['base_logits'],rz.cpu().numpy())
            for k,v in ss.items():assert np.array_equal(f[k],v.cpu().numpy())
        for condition in C['v7']['bn_states']+C['conditions']:
            row=next(r for r in rows if r['slice']==name and r['condition']==condition);ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';state=torch.load(ck,map_location='cpu',weights_only=True)
            assert sha(ck)==row['checkpoint_sha256'] and state['config_sha256']==sha(RUN/'config.json');scope(state['model'],pure,condition)
            m=load_model(seed);m.load_state_dict(state['model']);initial=canonical(m);a=None
            if state['adapter'] is not None:a=attach(m,'lora');a.load_state_dict(state['adapter'])
            # Full raw inputs ensure generator statistics are included in reload verification.
            tq=tokenize(m,query);tv=tokenize(m,valid);_,z=evaluate(m,tq,a);_,vz=evaluate(m,tv,a);assert_base_locked(m,initial)
            rawfile=RUN/'artifacts'/f'raw_{seed}_{name}_{condition}.npz';assert sha(rawfile)==row['raw_sha256']
            with np.load(rawfile) as f:assert np.array_equal(f['logits'],z.cpu().numpy()) and np.array_equal(f['rows'],query['rows'].numpy()) and np.array_equal(f['labels'],y)
            with np.load(RUN/'artifacts'/f'valid_{seed}_{name}_{condition}.npz') as f:assert np.array_equal(f['logits'],vz.cpu().numpy()) and np.array_equal(f['labels'],vy)
            pred=z.argmax(-1).cpu().numpy();vp=vz.argmax(-1).cpu().numpy();met=mm(y,pred,bp);vm=mm(vy,vp,vb)
            for k in met:assert abs(met[k]-row[k])<1e-12
            for k in vm:assert abs(vm[k]-row['source_valid_'+k])<1e-12
            assert abs(row['delta_f1_pp']-100*(met['macro_f1']-mm(y,bp,bp)['macro_f1']))<1e-10
            ss=scores(qz,z,g,source['calibration'])
            with np.load(RUN/'artifacts'/f'signal_{seed}_{name}_{condition}.npz') as f:
                for k,v in ss.items():assert np.array_equal(f[k],v.cpu().numpy())
            if condition in C['conditions']:
                mem=state['reference_state'];assert mem['steps']==400 and mem['features'].shape==(1024,128) and mem['prototype'].shape==(102,128) and mem['probabilities'].shape==(1024,102)
                assert torch.allclose(mem['probabilities'].sum(1),torch.ones(1024),atol=1e-5)
                assert all(torch.isfinite(mem[k]).all() for k in ['prototype','features','probabilities','mean','second'])
                hist=json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{condition}.json').read_text());assert len(hist)==400
                assert all(np.isfinite(h['loss']) and np.isfinite(h['gradient_norm']) and h['weight_min']>=.05-1e-7 and h['weight_max']<=1+1e-7 for h in hist)
                if condition=='memory_mass_control':assert all(abs(h['writer_weighted_mass']-h['routed_reference_mass'])<1e-4 for h in hist)
                gp=json.loads((RUN/'artifacts'/f'gradients_{seed}_{name}_{condition}.json').read_text());assert [r['step'] for r in gp]==[1,200,400]
            # All layer summaries are recomputed from saved per-row source/reference features.
            layer=json.loads((RUN/'artifacts'/f'layers_{seed}_{name}.json').read_text())[condition]
            with np.load(RUN/'artifacts'/f'layers_{seed}_{name}_{condition}.npz') as f:sp={k.removeprefix('source_'):f[k] for k in f.files if k.startswith('source_')};rp={k.removeprefix('reference_'):f[k] for k in f.files if k.startswith('reference_')}
            with np.load(RUN/'artifacts'/f'layers_{seed}_{name}_frozen.npz') as f:br={k.removeprefix('reference_'):f[k] for k in f.files if k.startswith('reference_')}
            calc=probe_summary(sp,rp,br)
            for l in calc:
                for k,v in calc[l].items():assert abs(v-layer[l][k])<1e-10
            records.append(dict(slice=name,condition=condition,query_full_reload=True,valid_full_reload=True,metrics_independent=True,layer_metric_recomputed=True));print(f'audit {seed} {name} {condition} {len(records)}/18',flush=True)
            del m,a,z,vz,tq,tv;torch.cuda.empty_cache()
        del qt,qe,qz,g,query,ref,rt,re,rz,rr;torch.cuda.empty_cache()
    save(f'audit_{seed}.json',dict(passed=True,seed=seed,query_checkpoints_reloaded=len(records),source_valid_checkpoints_reloaded=len(records),seconds=time.monotonic()-begin,records=records))

if __name__=='__main__':main()
