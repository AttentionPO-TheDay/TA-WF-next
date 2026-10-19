"""Fixed-budget reference-only isolation; query is terminal inference only."""
import os,time,json,traceback,copy
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,save,sha,load_model,load_slice,metrics
from candidate import gpu_dict,banks,neighbors,target_distribution
from methods import tokenize,evaluate,attach,assert_base_locked
from routing import geometry,scores
from refinement import RefinedState
from momentum_legacy import objective
from isolation import calibrate,copy_affines,reliability,loss_parts,train,capture,probe_summary


def inputs(seed):
    art=torch.load(ROOT/C['previous_run']/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True)
    p=gpu_dict(art['fits']['1']);bb=banks(art,p)
    prior=torch.load(ROOT/C['baseline_run']/'artifacts'/f'source_calibration_{seed}.pt',map_location='cpu',weights_only=True)
    source=torch.load(ROOT/C['v4_run']/'artifacts'/f'source_{seed}.pt',map_location='cpu',weights_only=True)
    prepared=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True)
    raw={k:v[art['anchor_indices']] for k,v in prepared['source'].items() if k in ('timestamps','tam','rows')}
    valid={k:v for k,v in prepared['valid'].items() if k in ('timestamps','tam','rows')}
    return p,bb,prior,source,raw,valid


def canonical(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def scope(state,pure,kind):
    allowed=[]
    for k in pure:
        if not torch.equal(state[k],pure[k]):
            assert '.norm' in k and k.startswith('generator.stages.'),k
            if kind=='affine_only':assert k.endswith(('.weight','.bias')),k
            elif kind in ('stats_only','stats_lora','stats_memory_route','lora_stats_composed'):assert k.endswith(('running_mean','running_var','num_batches_tracked')),k
            elif kind in ('memory_route','memory_mass_control'):raise AssertionError(k)
            allowed.append(k)
    return allowed


def old_adapter(seed,name):return torch.load(ROOT/C['v6_run']/'checkpoints'/f'{seed}_{name}_sharp_target_only.pt',map_location='cpu',weights_only=True)['adapter']


def preflight(seed):
    p,bb,prior,source,raw,valid=inputs(seed);m=load_model(seed);vt=tokenize(m,valid);ve,vz=evaluate(m,vt)
    assert torch.equal(vt.cpu(),source['valid_tokens']) and torch.equal(vz.cpu(),source['valid_logits'])
    assert torch.equal(tokenize(m,raw).cpu(),prior['source_anchor_tokens'])
    a=attach(m,'lora');qf=target_distribution(vz,source['valid_votes'].cuda(),'soft_neighbor');ix=torch.arange(64,device='cuda');state=RefinedState(ve,vz,p,'sharp_target_only');q=state.target(qf,ix);w,_=state.weight(ix,True)
    st=prior['source_anchor_tokens'].cuda()[:32];se=prior['source_anchor_embeddings'].cuda()[:32]
    l,x,z,parts,n=loss_parts(m,a,vt[ix],q,w,st,se,p)
    old,xx,zz,_=objective(m,a,vt[ix],q,w,state.source,st,se,p,False)
    assert torch.equal(l,old) and torch.equal(x,xx) and torch.equal(z,zz)
    ga=torch.autograd.grad(l,list(a.parameters()),retain_graph=True);gb=torch.autograd.grad(old,list(a.parameters()),retain_graph=True)
    assert all(torch.equal(u,v) for u,v in zip(ga,gb))
    r,rd=reliability(vz,source['valid_votes'].cuda(),vz);writer=(w*r[ix]).sum()/w.sum()
    assert torch.allclose((w*writer).sum(),(w*r[ix]).sum(),atol=1e-5)
    before=state.features.clone();state.update(ix,x+1,z,q,torch.zeros_like(w),True)
    assert torch.equal(state.prototype,state.source) and not torch.equal(state.features,before)
    m=load_model(seed);pure=canonical(m);notes=calibrate(m,valid);changed=scope(canonical(m),pure,'stats_only');assert len(changed)>0
    save(f'preflight_{seed}.json',dict(passed=True,seed=seed,objective_gradient_exact=True,source_tokens_exact=True,source_valid_exact=True,zero_writer_prototype_unchanged_memory_updates=True,mass_control_exact=True,bn_scope=changed,reliability=rd,bn_notes=notes))


def main(seed):
    p,bb,prior,source,raw,valid=inputs(seed);cal=source['calibration'];rows=[];begin=time.monotonic()
    vy=source['valid_labels'].numpy();vb=source['valid_logits'].argmax(-1).numpy();se=prior['source_anchor_embeddings'].cuda()
    for name in C['slices']:
        reference,query=load_slice(name);base=load_model(seed);pure=canonical(base);rt=tokenize(base,reference);re,rz=evaluate(base,rt);qt=tokenize(base,query);qe,qz=evaluate(base,qt);vt=source['valid_tokens'].cuda();st=prior['source_anchor_tokens'].cuda()
        votes,_=neighbors(re,p,bb,prior['calibration']['neighbor_temperature'].cuda());r,rd=reliability(rz,votes,source['valid_logits'].cuda());save(f'reliability_{seed}_{name}.json',rd)
        with np.load(ROOT/C['baseline_run']/'artifacts'/f'base_{seed}_{name}.npz') as old:assert np.array_equal(old['logits'],qz.cpu().numpy()) and np.array_equal(old['rows'],query['rows'].numpy())
        y=query['labels'].numpy();bp=qz.argmax(-1).cpu().numpy();g=geometry(qe,p);bf=metrics(y,bp,bp)['macro_f1']
        np.savez_compressed(RUN/'artifacts'/f'base_{seed}_{name}.npz',logits=qz.cpu().numpy(),predictions=bp,labels=y,rows=query['rows'].numpy())
        tg=geometry(re,p);v6=load_model(seed);a6=attach(v6,'lora');a6.load_state_dict(old_adapter(seed,name));_,ref6=evaluate(v6,rt,a6)
        refsig=scores(rz,ref6,tg,cal)
        np.savez_compressed(RUN/'artifacts'/f'reference_signal_{seed}_{name}.npz',base_logits=rz.cpu().numpy(),candidate_logits=ref6.cpu().numpy(),rows=reference['rows'].numpy(),**{k:v.cpu().numpy() for k,v in refsig.items()})
        sp={k:v[:C['v7']['source_probe_rows']] for k,v in raw.items()};rp={k:v[:C['v7']['reference_probe_rows']] for k,v in reference.items()};ps=capture(base,sp);pr=capture(base,rp);probes={}
        def probe(label,m):
            ss=capture(m,sp);rr=capture(m,rp);probes[label]=probe_summary(ss,rr,pr)
            np.savez_compressed(RUN/'artifacts'/f'layers_{seed}_{name}_{label}.npz',**{'source_'+k:v for k,v in ss.items()},**{'reference_'+k:v for k,v in rr.items()})
        probe('frozen',base);probe('v6_lora',v6)
        tent=torch.load(ROOT/C['baseline_run']/'checkpoints'/f'{seed}_{name}_tent_bn_reference.pt',map_location='cpu',weights_only=True)['model']
        bm=load_model(seed);bnnotes=calibrate(bm,reference);stats=canonical(bm);scope(stats,pure,'stats_only');save(f'bn_calibration_{seed}_{name}.json',bnnotes)
        del bm,v6,a6
        for condition in C['v7']['bn_states']+C['conditions']:
            start=time.monotonic();torch.cuda.reset_peak_memory_stats();m=load_model(seed);hist=[];gp=[];state=None;a=None
            if condition=='affine_only':copy_affines(m,tent)
            elif condition in ('tent_both','lora_tent_composed'):m.load_state_dict(tent)
            elif condition in ('stats_only','lora_stats_composed','stats_lora','stats_memory_route'):m.load_state_dict(stats)
            initial=canonical(m);changed=scope(initial,pure,condition)
            if condition in C['conditions']:
                if condition.startswith('stats_'):
                    tr=tokenize(m,reference);ee,_=evaluate(m,tr);ts=tokenize(m,raw)
                else:tr,ee,ts=rt,re,st
                a,state,hist,gp=train(m,tr,ee,rz,votes,ts,se,p,condition,r)
            elif condition.startswith('lora_'):a=attach(m,'lora');a.load_state_dict(old_adapter(seed,name))
            assert_base_locked(m,initial)
            # Re-tokenization is mandatory when the generator BN differs.
            tq=tokenize(m,query) if changed else qt;tv=tokenize(m,valid) if changed else vt
            _,z=evaluate(m,tq,a);_,vz=evaluate(m,tv,a)
            if condition=='tent_both':
                with np.load(ROOT/C['baseline_run']/'artifacts'/f'raw_{seed}_{name}_tent_bn_reference.npz') as old:assert np.array_equal(old['logits'],z.cpu().numpy())
            ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';torch.save(dict(model=initial,adapter=None if a is None else {k:v.detach().cpu() for k,v in a.state_dict().items()},reference_state=None if state is None else state.dump(),seed=seed,slice=name,condition=condition,config_sha256=sha(RUN/'config.json')),ck)
            rawfile=RUN/'artifacts'/f'raw_{seed}_{name}_{condition}.npz';np.savez_compressed(rawfile,logits=z.cpu().numpy(),predictions=z.argmax(-1).cpu().numpy(),labels=y,rows=query['rows'].numpy())
            np.savez_compressed(RUN/'artifacts'/f'valid_{seed}_{name}_{condition}.npz',logits=vz.cpu().numpy(),predictions=vz.argmax(-1).cpu().numpy(),labels=vy,rows=source['valid_rows'].numpy())
            ss=scores(qz,z,g,cal);np.savez_compressed(RUN/'artifacts'/f'signal_{seed}_{name}_{condition}.npz',**{k:v.cpu().numpy() for k,v in ss.items()})
            met=metrics(y,z.argmax(-1).cpu().numpy(),bp);vm=metrics(vy,vz.argmax(-1).cpu().numpy(),vb)
            row=dict(seed=seed,slice=name,condition=condition,**met,delta_f1_pp=100*(met['macro_f1']-bf),steps=len(hist),parameters=4096 if a is not None else 0,seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),raw_sha256=sha(rawfile),checkpoint_sha256=sha(ck),bn_changed=changed,**{'source_valid_'+k:v for k,v in vm.items()})
            rows.append(row);save(f'scores_{seed}.json',rows)
            if hist:save(f'updates_{seed}_{name}_{condition}.json',hist);save(f'gradients_{seed}_{name}_{condition}.json',gp)
            probe(condition,m);save(f'layers_{seed}_{name}.json',probes)
            save(f'progress_{seed}.json',dict(tasks=len(rows),total=18,seconds=time.monotonic()-begin,pid=os.getpid()));print(json.dumps(row),flush=True)
            del m,a,state,z,vz,tq,tv;torch.cuda.empty_cache()
        del base,rt,re,rz,qt,qe,qz,vt,st,votes,r,g,tg,ref6;torch.cuda.empty_cache()
    assert len(rows)==18

if __name__=='__main__':
    try:
        setup();verify();seed=int(os.environ['CANDIDATE_SEED'])
        assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
        if os.environ['MOMENTUM_MODE']=='preflight':preflight(seed)
        else:main(seed)
    except BaseException as e:save(f'failure_{os.environ.get("MOMENTUM_MODE")}_{os.environ.get("CANDIDATE_SEED")}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
