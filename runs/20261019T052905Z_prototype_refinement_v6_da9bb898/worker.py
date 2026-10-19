"""Only two observed diagnostic slices; independent query terminal scoring."""
import os,time,json,traceback
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,save,sha,load_model,load_slice,metrics
from candidate import gpu_dict,banks,neighbors,target_distribution
from methods import tokenize,evaluate,attach,standardize,relational,encode,assert_base_locked
from routing import geometry,scores
from refinement import train,RefinedState
from momentum_legacy import MomentumState,objective


def inputs(seed):
    art=torch.load(ROOT/C['previous_run']/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True);p=gpu_dict(art['fits']['1']);bb=banks(art,p)
    prior=torch.load(ROOT/C['baseline_run']/'artifacts'/f'source_calibration_{seed}.pt',map_location='cpu',weights_only=True)
    source=torch.load(ROOT/C['v4_run']/'artifacts'/f'source_{seed}.pt',map_location='cpu',weights_only=True)
    return p,bb,prior,source


def preflight(seed):
    p,bb,prior,source=inputs(seed);m=load_model(seed);vt=source['valid_tokens'].cuda();ve,vz=evaluate(m,vt)
    assert torch.equal(vz.cpu(),source['valid_logits']) and torch.equal(ve.cpu(),prior['valid_embeddings'])
    initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};a=attach(m,'lora');base=load_model(seed);_,bz=evaluate(base,vt[:16]);_,az=evaluate(m,vt[:16],a);assert torch.equal(bz,az)
    ix=torch.arange(64,device='cuda');q=target_distribution(vz,source['valid_votes'].cuda(),'soft_neighbor');assert q.shape==(2160,102)
    legacy=MomentumState(ve,vz,p);test=RefinedState(ve,vz,p,'legacy_control')
    q0=legacy.target(q,ix,True);qq=test.target(q,ix);assert torch.equal(q0,qq)
    w,_=legacy.weight(ix,True);ww,_=test.weight(ix,True);assert torch.equal(w,ww)
    st=prior['source_anchor_tokens'].cuda()[:32];se=prior['source_anchor_embeddings'].cuda()[:32]
    loss,x,z,_=objective(m,a,vt[ix],qq,ww,test.loss_anchors(),st,se,p,False)
    old,_,_,_=objective(m,a,vt[ix],q0,w,legacy.anchored(),st,se,p,False)
    assert torch.equal(loss,old)
    ga=torch.autograd.grad(loss,list(a.parameters()),retain_graph=True);gb=torch.autograd.grad(old,list(a.parameters()),retain_graph=True)
    assert all(torch.equal(v,b) for v,b in zip(ga,gb))
    legacy.update(ix,x,z,q0,w,True);test.observe(ix,x,z,qq,ww)
    assert all(torch.equal(legacy.dump()[k],test.dump()[k]) if torch.is_tensor(legacy.dump()[k]) else legacy.dump()[k]==test.dump()[k] for k in legacy.dump())
    tests=[]
    for condition in C['conditions']:
        t=RefinedState(ve,vz,p,condition);qt=t.target(q,ix);wt,_=t.weight(ix,True)
        assert not qt.requires_grad and not wt.requires_grad and torch.isfinite(qt).all() and torch.allclose(qt.sum(1),torch.ones(64,device='cuda'))
        assert wt.min()>=.05 and wt.max()<=1
        if condition=='anchor_only':assert torch.equal(qt,q[ix])
        t.observe(ix,x,z,qt,wt)
        assert not t.prototype.requires_grad and torch.isfinite(t.prototype).all()
        if not t.options['prototype_update']:assert torch.equal(t.prototype,t.source)
        if t.options['loss_anchor']=='source':assert torch.equal(t.loss_anchors(),t.source)
        if t.options['radius_preservation']:
            aa=t.anchored();src=t.source;raw=.5*(src+t.prototype);ss=src-src.mean(0);center=raw.mean(0)
            # Radius is around the raw prototype center (not the corrected set mean).
            assert torch.allclose((aa-center).norm(dim=-1),ss.norm(dim=-1),atol=3e-6,rtol=1e-6)
        tests.append({'condition':condition,'target_detached':True,'prototype_scope':True,'loss_anchor_scope':True})
    u=torch.nn.functional.normalize(test.features[ix],dim=-1);v=torch.nn.functional.normalize(test.anchored(),dim=-1)
    soft=(u@v.T/.1).softmax(-1);sharp=(u@v.T/.05).softmax(-1)
    ent=lambda v:-(v*v.clamp_min(1e-12).log()).sum(-1)
    assert (ent(sharp)<=ent(soft)+1e-5).all()
    # No support leaves 101 source prototypes untouched.
    t=RefinedState(ve,vz,p,'sharp_proto');one=torch.zeros_like(qq);one[:,0]=1;t.observe(ix,x,z,one,ww);assert torch.equal(t.prototype[1:],t.source[1:])
    assert_base_locked(m,initial);assert sum(v.numel() for v in a.parameters())==4096
    save(f'preflight_{seed}.json',{'passed':True,'seed':seed,'source_valid_parity':True,'zero_init_same_no_grad_batch':True,'v5_loss_gradient_and_one_step_state_exact':True,'radius_constraint':True,'sharp_geometry_lower_entropy':True,'source_locked':True,'parameters':4096,'arms':tests})


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
            rows.append(row);save(f'updates_{seed}_{name}_{condition}.json',hist);save(f'scores_{seed}.json',rows);save(f'progress_{seed}.json',{'tasks':len(rows),'total':12,'slice':name,'condition':condition,'seconds':time.monotonic()-begin,'pid':os.getpid()});print(json.dumps(row),flush=True)
            del m,a,z,vz,state;torch.cuda.empty_cache()
        del rt,re,rz,qt,qe,qz,reference,query,votes,g;torch.cuda.empty_cache()
    assert len(rows)==12

if __name__=='__main__':
    try:
        setup();verify();seed=int(os.environ['CANDIDATE_SEED']);assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
        if os.environ['MOMENTUM_MODE']=='preflight':preflight(seed)
        else:main(seed)
    except BaseException as e:
        save(f'failure_{os.environ.get("MOMENTUM_MODE")}_{os.environ.get("CANDIDATE_SEED")}.json',{'error':str(e),'traceback':traceback.format_exc()});raise
