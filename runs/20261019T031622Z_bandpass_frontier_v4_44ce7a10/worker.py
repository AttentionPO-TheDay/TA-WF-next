"""One seed/GPU. Source-only preflight, reused controls, then fixed routing arms."""
import json,os,time,traceback
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,load_model,load_slice,save,sha,metrics,predict,entropy
from methods import tokenize,evaluate,attach,standardize,encode,relational,assert_base_locked
from candidate import gpu_dict,banks,neighbors,target_distribution
from routing import geometry,calibration,weights,train,scores,select_threshold,objective

CONTROLS=['ungated_lora','ungated_residual','tent_reference','direct_rerank']
MAP={'ungated_lora':'soft_neighbor_lora','ungated_residual':'soft_neighbor_residual','tent_reference':'tent_bn_reference'}

def evidence_numpy(g):return {k:v.cpu().numpy() for k,v in g.items()}

def inputs(seed):
    old=ROOT/C['baseline_run'];art=torch.load(ROOT/C['previous_run']/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True);p=gpu_dict(art['fits']['1'])
    prior=torch.load(old/'artifacts'/f'source_calibration_{seed}.pt',map_location='cpu',weights_only=True)
    return old,p,banks(art,p),prior

def preflight(seed):
    old,p,bb,prior=inputs(seed);m=load_model(seed);source=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True)
    valid=source['valid'];vt=tokenize(m,valid);ve,vz=evaluate(m,vt)
    assert torch.equal(ve.cpu(),prior['valid_embeddings']) and torch.equal(vz.cpu(),prior['valid_logits'])
    vg=geometry(ve,p);cal=calibration(vg);src_t=prior['source_anchor_tokens'].cuda();src_e=prior['source_anchor_embeddings'].cuda()
    votes,_=neighbors(ve,p,bb,prior['calibration']['neighbor_temperature'].cuda());q=target_distribution(vz,votes,'soft_neighbor');a=attach(m,'lora')
    _,zero=evaluate(m,vt[:16],a);baseline_model=load_model(seed);_,same_batch_baseline=evaluate(baseline_model,vt[:16]);assert torch.equal(zero,same_batch_baseline);del baseline_model;ix=torch.arange(64,device='cuda');si=torch.arange(32,device='cuda');x,z=encode(m,vt[ix],a);xx=standardize(x,p);anchors=q@p['centers'][:,0]
    kl=torch.nn.functional.kl_div((z/2).log_softmax(-1),q[ix],reduction='batchmean')*4
    pull=((xx[:,None]-p['centers'][:,0][None]).square().mean(-1)*q[ix]).sum(-1).mean()
    rel,_=relational(xx,anchors[ix],torch.ones(64,device='cuda'),q[ix].argmax(-1));se,_=encode(m,src_t[si],a);ident=(standardize(se,p)-standardize(src_e[si],p)).square().mean()
    original=kl+.1*pull+rel+ident
    new,_=objective(m,a,vt[ix],q[ix],anchors[ix],q[ix].argmax(-1),torch.ones(64,device='cuda'),src_t[si],src_e[si],p)
    assert torch.allclose(original,new,atol=1e-6,rtol=1e-6),(original,new)
    original_grad=torch.autograd.grad(original,list(a.parameters()),retain_graph=True);new_grad=torch.autograd.grad(new,list(a.parameters()))
    assert all(torch.allclose(v,w,atol=1e-6,rtol=1e-5) for v,w in zip(original_grad,new_grad))
    checks={};h=entropy(vz)
    for name in C['conditions']:
        w,detail=weights(h,vg,cal,name);checks[name]=detail;assert not w.requires_grad
    # Verify smooth weights do not leak derivatives, and normalized-margin evidence is scale invariant.
    sc=scores(vz,vz,vg,cal);scaled=scores(3*vz,7*vz,vg,cal)
    assert torch.allclose(sc['normalized_margin_gain'],scaled['normalized_margin_gain'],atol=1e-5)
    torch.save({'valid_tokens':vt.cpu(),'valid_logits':vz.cpu(),'valid_rows':valid['rows'],'valid_labels':valid['labels'],'valid_geometry':{k:v.cpu() for k,v in vg.items()},'calibration':cal,'valid_votes':votes.cpu()},RUN/'artifacts'/f'source_{seed}.pt')
    save(f'preflight_{seed}.json',{'passed':True,'seed':seed,'source_rows':len(source['source']['rows']),'valid_rows':len(vt),'zero_init_bitwise':True,'no_route_loss_gradient_parity':True,'source_v3_bitwise_parity':True,'logit_scale_invariance':True,'calibration':cal,'weight_checks':checks,'source_sha256':sha(RUN/'artifacts'/f'source_{seed}.pt')})

def record(seed,name,condition,z,vz,base_z,valid_base,query,g,vg,cal,source,metadata):
    out=RUN/'artifacts'/f'raw_{seed}_{name}_{condition}.npz';np.savez_compressed(out,logits=z.cpu().numpy(),predictions=z.argmax(-1).cpu().numpy(),labels=query['labels'].numpy(),rows=query['rows'].numpy())
    ss=scores(base_z,z,g,cal);vs=scores(valid_base,vz,vg,cal)
    np.savez_compressed(RUN/'artifacts'/f'signal_{seed}_{name}_{condition}.npz',**{k:v.cpu().numpy() for k,v in ss.items()})
    np.savez_compressed(RUN/'artifacts'/f'valid_{seed}_{name}_{condition}.npz',logits=vz.cpu().numpy(),combined=vs['combined'].cpu().numpy())
    bp=base_z.argmax(-1);cp=z.argmax(-1);y=query['labels'].numpy();bn=bp.cpu().numpy();bf=metrics(y,bn,bn)['macro_f1'];rows=[];thresholds={}
    for policy in C['policies']:
        if policy=='raw':keep=cp!=bp;zz=z
        else:
            budget=.01 if policy=='source_nar1' else .02
            threshold=select_threshold(source['valid_labels'].cuda(),valid_base.argmax(-1),vz.argmax(-1),vs['combined'],budget);thresholds[policy]=threshold
            keep=(cp!=bp)&(ss['combined']>=threshold['threshold']) if threshold['enabled'] else torch.zeros_like(cp,dtype=torch.bool)
            zz=torch.where(keep[:,None],z,base_z);assert torch.equal(zz[~keep],base_z[~keep])
        pred=zz.argmax(-1).cpu().numpy();np.savez_compressed(RUN/'artifacts'/f'policy_{seed}_{name}_{condition}_{policy}.npz',predictions=pred,keep=keep.cpu().numpy(),rows=query['rows'].numpy())
        met=metrics(y,pred,bn);rows.append({'seed':seed,'slice':name,'condition':condition,'policy':policy,**met,'delta_f1_pp':100*(met['macro_f1']-bf),'changes':int(keep.sum()),**metadata})
    save(f'thresholds_{seed}_{name}_{condition}.json',thresholds)
    return rows

def run(seed,mode):
    old,p,bb,prior=inputs(seed);source=torch.load(RUN/'artifacts'/f'source_{seed}.pt',map_location='cpu',weights_only=True);vt=source['valid_tokens'].cuda();vbase=source['valid_logits'].cuda();vg=gpu_dict(source['valid_geometry']);cal=source['calibration']
    src_t=prior['source_anchor_tokens'].cuda();src_e=prior['source_anchor_embeddings'].cuda();begin=time.monotonic()
    conditions=CONTROLS if mode=='controls' else C['conditions']
    if mode=='shuffle':
        trigger=json.loads((RUN/'artifacts/shuffle_trigger.json').read_text());assert trigger['enabled'];conditions=['shuffle_'+trigger['arm']]
    scorepath=RUN/'artifacts'/f'{mode}_scores_{seed}.json';rs=json.loads(scorepath.read_text()) if scorepath.exists() else []
    for name in C['slices']:
        reference,query=load_slice(name);m=load_model(seed);qt=tokenize(m,query);qe,qz=evaluate(m,qt)
        with np.load(old/'artifacts'/f'base_{seed}_{name}.npz') as b:
            assert np.array_equal(b['logits'],qz.cpu().numpy()) and np.array_equal(b['rows'],query['rows'].numpy()) and np.array_equal(b['labels'],query['labels'].numpy())
        g=geometry(qe,p)
        if mode=='controls':np.savez_compressed(RUN/'artifacts'/f'geometry_{seed}_{name}.npz',**evidence_numpy(g))
        else:
            with np.load(RUN/'artifacts'/f'geometry_{seed}_{name}.npz') as v:
                for k in g:assert np.array_equal(v[k],g[k].cpu().numpy())
        rt=tokenize(m,reference);re,rz=evaluate(m,rt);rv,_=neighbors(re,p,bb,prior['calibration']['neighbor_temperature'].cuda());rg=geometry(re,p);rh=entropy(rz)
        for condition in conditions:
            if any(r['slice']==name and r['condition']==condition for r in rs):continue
            start=time.monotonic();m=load_model(seed);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};a=None;hist=[];detail={};torch.cuda.reset_peak_memory_stats()
            if mode=='controls':
                if condition=='direct_rerank':
                    with np.load(old/'artifacts'/f'evidence_{seed}_{name}.npz') as v:vv=torch.from_numpy(v['votes']).cuda()
                    z=target_distribution(qz,vv,'soft_neighbor').clamp_min(1e-12).log();vz=target_distribution(vbase,source['valid_votes'].cuda(),'soft_neighbor').clamp_min(1e-12).log()
                else:
                    oldcond=MAP[condition]
                    with np.load(old/'artifacts'/f'raw_{seed}_{name}_{oldcond}.npz') as v:z=torch.from_numpy(v['logits']).cuda();assert np.array_equal(v['rows'],query['rows'].numpy())
                    ck=old/'checkpoints'/f'{seed}_{name}_{oldcond}.pt';state=torch.load(ck,map_location='cpu',weights_only=True)
                    if condition=='tent_reference':m.load_state_dict(state['model']);vz=torch.from_numpy(predict(m,{'timestamps':torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True)['valid']['timestamps'],'tam':torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True)['valid']['tam'],'rows':source['valid_rows']})).cuda()
                    else:a=attach(m,oldcond.rsplit('_',1)[1]);a.load_state_dict(state['adapter']);_,vz=evaluate(m,vt,a)
                    detail={'reused_checkpoint_sha256':sha(ck),'reused_raw_sha256':sha(old/'artifacts'/f'raw_{seed}_{name}_{oldcond}.npz')}
                metadata={'reused_control':True,'steps':0,'training_provenance':'v3 400 steps; direct rerank zero training','detail':detail}
            else:
                w,detail=weights(rh,rg,cal,condition);save(f'routing_{seed}_{name}_{condition}.json',detail)
                np.savez_compressed(RUN/'artifacts'/f'reference_weights_{seed}_{name}_{condition}.npz',weights=w.cpu().numpy(),entropy=rh.cpu().numpy(),pmr=rg['pmr'].cpu().numpy(),maha=rg['maha'].cpu().numpy(),rows=reference['rows'].numpy())
                a,hist=train(m,rt,rz,rv,src_t,src_e,p,w,condition);assert_base_locked(m,initial);_,z=evaluate(m,qt,a);_,vz=evaluate(m,vt,a)
                ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';torch.save({'adapter':{n:v.detach().cpu() for n,v in a.state_dict().items()},'seed':seed,'slice':name,'condition':condition,'config_sha256':sha(RUN/'config.json'),'source_sha256':sha(RUN/'artifacts'/f'source_{seed}.pt')},ck)
                save(f'updates_{seed}_{name}_{condition}.json',hist);metadata={'reused_control':False,'steps':len(hist),'parameters':sum(v.numel() for v in a.parameters()),'checkpoint_sha256':sha(ck),'detail':detail}
            assert torch.isfinite(z).all() and torch.isfinite(vz).all();metadata.update(seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            rs.extend(record(seed,name,condition,z,vz,qz,vbase,query,g,vg,cal,source,metadata));save(f'{mode}_scores_{seed}.json',rs)
            save(f'progress_{mode}_{seed}.json',{'states':len(rs),'slice':name,'condition':condition,'pid':os.getpid(),'seconds':time.monotonic()-begin})
            print(json.dumps({'mode':mode,'seed':seed,'slice':name,'condition':condition,'states':len(rs),'seconds':time.monotonic()-start}),flush=True)
            del m,a,z,vz;torch.cuda.empty_cache()
        del qt,qe,qz,rt,re,rz,rv,rg,g,reference,query;torch.cuda.empty_cache()
        if time.monotonic()-begin>C['budget']['max_wall_seconds']:raise TimeoutError('worker wall')
    assert len(rs)==len(C['slices'])*len(conditions)*len(C['policies'])

if __name__=='__main__':
    try:
        setup();verify();seed=int(os.environ['CANDIDATE_SEED']);mode=os.environ['BANDPASS_MODE'];assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
        if mode=='preflight':preflight(seed)
        else:run(seed,mode)
    except BaseException as e:
        save(f'failure_{os.environ.get("BANDPASS_MODE")}_{os.environ.get("CANDIDATE_SEED")}.json',{'error':str(e),'traceback':traceback.format_exc()});raise
