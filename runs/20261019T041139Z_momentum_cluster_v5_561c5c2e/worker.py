"""Only two observed diagnostic slices; independent query terminal scoring."""
import os,time,json,traceback
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,save,sha,load_model,load_slice,metrics
from candidate import gpu_dict,banks,neighbors,target_distribution
from methods import tokenize,evaluate,attach,standardize,relational,encode,assert_base_locked
from routing import geometry,scores
from momentum import train,MomentumState,soft_contrast,objective


def inputs(seed):
    art=torch.load(ROOT/C['previous_run']/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True);p=gpu_dict(art['fits']['1']);bb=banks(art,p)
    prior=torch.load(ROOT/C['baseline_run']/'artifacts'/f'source_calibration_{seed}.pt',map_location='cpu',weights_only=True)
    source=torch.load(ROOT/C['v4_run']/'artifacts'/f'source_{seed}.pt',map_location='cpu',weights_only=True)
    return p,bb,prior,source


def preflight(seed):
    p,bb,prior,source=inputs(seed);m=load_model(seed);vt=source['valid_tokens'].cuda();ve,vz=evaluate(m,vt)
    assert torch.equal(vz.cpu(),source['valid_logits']) and torch.equal(ve.cpu(),prior['valid_embeddings'])
    initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};a=attach(m,'lora');base=load_model(seed);_,bz=evaluate(base,vt[:16]);_,az=evaluate(m,vt[:16],a);assert torch.equal(bz,az)
    ix=torch.arange(64,device='cuda');q=target_distribution(vz,source['valid_votes'].cuda(),'soft_neighbor');assert q.shape==(2160,102) and torch.isfinite(q).all() and torch.allclose(q.sum(1),torch.ones(2160,device='cuda'))
    state=MomentumState(ve,vz,p);q0=state.target(q,ix,False);w,detail=state.weight(ix,False);st=prior['source_anchor_tokens'].cuda()[:32];se=prior['source_anchor_embeddings'].cuda()[:32]
    loss,x,z,ld=objective(m,a,vt[ix],q0,w,state.source,st,se,p,False)
    # Full static implementation must match legacy loss under the exact same full q.
    temp=C['adaptation']['kl_temperature'];kl=torch.nn.functional.kl_div((z/temp).log_softmax(-1),q0,reduction='batchmean')*temp**2
    pull=((x[:,None]-state.source[None]).square().mean(-1)*q0).sum(-1).mean();rel,_=relational(x,q0@state.source,w,q0.argmax(-1));ee,_=encode(m,st,a);identity=(standardize(ee,p)-standardize(se,p)).square().mean();old=kl+.1*pull+rel+identity
    assert torch.allclose(loss,old,atol=1e-6,rtol=1e-6)
    ga=torch.autograd.grad(loss,list(a.parameters()),retain_graph=True);gb=torch.autograd.grad(old,list(a.parameters()),retain_graph=True);assert all(torch.allclose(v,b,atol=1e-6,rtol=1e-5) for v,b in zip(ga,gb))
    ww,detail=state.weight(ix,True);assert ww.min()>=.05 and ww.max()<=1 and not ww.requires_grad
    qq=state.target(q,ix,True);assert not qq.requires_grad and torch.allclose(qq.sum(1),torch.ones(64,device='cuda'))
    # Source-conditioned pair loss: finite nonzero gradients, diagonal is not a positive.
    xt=x.detach().clone().requires_grad_();cl,cd=soft_contrast(xt,qq,ww);grad=torch.autograd.grad(cl,xt)[0];assert torch.isfinite(grad).all() and grad.norm()>0
    per=torch.randperm(64,device='cuda');cl2,_=soft_contrast(xt[per],qq[per],ww[per]);assert torch.allclose(cl,cl2,atol=1e-6,rtol=1e-6)
    before=state.prototype.clone();state.update(ix,x,z,qq,ww,True);assert state.steps==1 and not torch.equal(before,state.prototype);assert not state.prototype.requires_grad and torch.isfinite(state.prototype).all()
    # Empty class support must leave the other101 prototypes unchanged.
    test=MomentumState(ve,vz,p);one=torch.zeros_like(qq);one[:,0]=1;test.update(ix,x,z,one,ww,True);assert torch.equal(test.prototype[1:],test.source[1:])
    assert_base_locked(m,initial)
    save(f'preflight_{seed}.json',{'passed':True,'seed':seed,'source_valid_parity':True,'zero_init_bitwise_same_batch':True,'full_static_loss_gradient_parity':True,'soft_targets_full102':True,'weights_positive_detached':True,'contrast_permutation_invariant':True,'contrast_nonzero_gradient':float(grad.norm()),'zero_support_prototypes_unchanged':True,'source_base_locked':True,'source_rows':18553,'source_valid_rows':2160,'weight_example':detail,'pair_example':cd})


def main(seed):
    p,bb,prior,source=inputs(seed);path=RUN/'artifacts'/f'scores_{seed}.json';rows=json.loads(path.read_text()) if path.exists() else [];begin=time.monotonic()
    vt=source['valid_tokens'].cuda();vbase=source['valid_logits'].cuda();st=prior['source_anchor_tokens'].cuda();se=prior['source_anchor_embeddings'].cuda();cal=source['calibration']
    for name in C['slices']:
        reference,query=load_slice(name);m=load_model(seed);rt=tokenize(m,reference);re,rz=evaluate(m,rt);qt=tokenize(m,query);qe,qz=evaluate(m,qt)
        with np.load(ROOT/C['baseline_run']/'artifacts'/f'base_{seed}_{name}.npz') as b:assert np.array_equal(b['logits'],qz.cpu().numpy()) and np.array_equal(b['rows'],query['rows'].numpy())
        votes,_=neighbors(re,p,bb,prior['calibration']['neighbor_temperature'].cuda());g=geometry(qe,p);y=query['labels'].numpy();bn=qz.argmax(-1).cpu().numpy();bf=metrics(y,bn,bn)['macro_f1'];vy=source['valid_labels'].numpy();vb=vbase.argmax(-1).cpu().numpy()
        np.savez_compressed(RUN/'artifacts'/f'base_{seed}_{name}.npz',logits=qz.cpu().numpy(),predictions=bn,labels=y,rows=query['rows'].numpy());np.savez_compressed(RUN/'artifacts'/f'geometry_{seed}_{name}.npz',**{k:v.cpu().numpy() for k,v in g.items()})
        for condition in C['conditions']:
            if any(r['slice']==name and r['condition']==condition for r in rows):continue
            start=time.monotonic();torch.cuda.reset_peak_memory_stats();m=load_model(seed);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            a,state,hist=train(m,rt,re,rz,votes,st,se,p,condition);assert_base_locked(m,initial);_,z=evaluate(m,qt,a);_,vz=evaluate(m,vt,a);assert torch.isfinite(z).all() and torch.isfinite(vz).all()
            ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';torch.save({'adapter':{k:v.detach().cpu() for k,v in a.state_dict().items()},'reference_state':state.dump(),'seed':seed,'slice':name,'condition':condition,'config_sha256':sha(RUN/'config.json')},ck)
            raw=RUN/'artifacts'/f'raw_{seed}_{name}_{condition}.npz';np.savez_compressed(raw,logits=z.cpu().numpy(),predictions=z.argmax(-1).cpu().numpy(),labels=y,rows=query['rows'].numpy());np.savez_compressed(RUN/'artifacts'/f'valid_{seed}_{name}_{condition}.npz',logits=vz.cpu().numpy(),predictions=vz.argmax(-1).cpu().numpy(),labels=vy,rows=source['valid_rows'].numpy())
            ss=scores(qz,z,g,cal);np.savez_compressed(RUN/'artifacts'/f'signal_{seed}_{name}_{condition}.npz',**{k:v.cpu().numpy() for k,v in ss.items()})
            met=metrics(y,z.argmax(-1).cpu().numpy(),bn);vm=metrics(vy,vz.argmax(-1).cpu().numpy(),vb)
            row={'seed':seed,'slice':name,'condition':condition,'policy':'raw',**met,'delta_f1_pp':100*(met['macro_f1']-bf),'source_valid_accuracy':vm['accuracy'],'source_valid_macro_f1':vm['macro_f1'],'source_valid_nar':vm['nar'],'source_valid_recovery':vm['recovery'],'steps':len(hist),'parameters':sum(v.numel() for v in a.parameters()),'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'raw_sha256':sha(raw),'checkpoint_sha256':sha(ck),'output_bypass':False,'reference_rows':1024,'reference_probability_rows':len(state.probabilities),'prototype_drift_mean':hist[-1]['prototype_drift_mean'],'weight_mean_final':hist[-1]['weight_mean']}
            rows.append(row);save(f'updates_{seed}_{name}_{condition}.json',hist);save(f'scores_{seed}.json',rows);save(f'progress_{seed}.json',{'tasks':len(rows),'total':10,'slice':name,'condition':condition,'seconds':time.monotonic()-begin,'pid':os.getpid()});print(json.dumps(row),flush=True)
            del m,a,z,vz,state;torch.cuda.empty_cache()
        del rt,re,rz,qt,qe,qz,reference,query,votes,g;torch.cuda.empty_cache()
    assert len(rows)==10

if __name__=='__main__':
    try:
        setup();verify();seed=int(os.environ['CANDIDATE_SEED']);assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
        if os.environ['MOMENTUM_MODE']=='preflight':preflight(seed)
        else:main(seed)
    except BaseException as e:
        save(f'failure_{os.environ.get("MOMENTUM_MODE")}_{os.environ.get("CANDIDATE_SEED")}.json',{'error':str(e),'traceback':traceback.format_exc()});raise
