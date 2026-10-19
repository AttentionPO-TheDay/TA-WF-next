"""Two observed slices,24 adaptations, frozenBN and independent query inference."""
import os,time,json,traceback
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,save,sha,load_model,load_slice,metrics
from methods import tokenize,evaluate,attach,assert_base_locked
from candidate import neighbors,target_distribution
from routing import geometry
from isolation import loss_parts
from momentum_legacy import objective
from refinement import RefinedState
from progressive import train,schedule,grads
from view import probs
from shared import inputs,canonical,old_adapter,signals,kl_gate,cluster,diagnostics


def preflight(seed):
    p,bb,prior,src=inputs(seed);m=load_model(seed);initial=canonical(m);t=src['valid_tokens'].cuda();e,z=evaluate(m,t);assert torch.equal(z.cpu(),src['valid_logits']);a=attach(m,'lora');ix=torch.arange(64,device='cuda');qf=target_distribution(z,src['valid_votes'].cuda(),'soft_neighbor');state=RefinedState(e,z,p,'sharp_target_only');q=state.target(qf,ix);w,_=state.weight(ix,True);st=prior['source_anchor_tokens'].cuda()[:32];se=prior['source_anchor_embeddings'].cuda()[:32]
    legacy,x,zz,parts,n=loss_parts(m,a,t[ix],q,w,st,se,p);old,_,_,_=objective(m,a,t[ix],q,w,state.source,st,se,p,False);assert torch.equal(legacy,old) and torch.equal(grads(legacy,list(a.parameters())),grads(old,list(a.parameters())))
    ema=None;ell=0.;weights=[]
    for step in range(1,401):
        weight,ema,ell,ratio=schedule(1.,.025,step,ema,ell,True);weights.append(weight);assert 1<=weight<=50
        if step>100:assert abs(np.log(weights[-1])-np.log(weights[-2]))<=(.02 if step<=200 else .005)+1e-10
    assert weights[0]<1.001 and weights[99]<40 and weights[-1]>30
    view=torch.load(RUN/'artifacts/source_view.pt',weights_only=True);vq=view['valid_q'].cuda()[ix];vc=(1+(vq*vq.clamp_min(1e-12).log()).sum(-1)/np.log(102)).clamp(0,1);perm=torch.arange(63,-1,-1,device='cuda');sh=vc[perm]*(w*vc).sum()/(w*vc[perm]).sum().clamp_min(1e-12);assert torch.allclose((w*sh).sum(),(w*vc).sum(),atol=1e-5)
    assert torch.allclose(vq.sum(1),torch.ones(64,device='cuda'),atol=1e-6) and not vq.requires_grad;comp=view['components'];assert torch.allclose(comp@comp.T,torch.eye(32,dtype=comp.dtype),atol=1e-10);assert_base_locked(m,initial)
    save(f'preflight_{seed}.json',dict(passed=True,v6_loss_gradient_exact=True,schedule_warmup_rate_bounds=True,aux_mass_match=True,pca_orthogonal=True,source_view_enabled=view['enabled'],pure_bn_locked=True,parameters=4096))


def main(seed):
    p,bb,prior,src=inputs(seed);vt=src['valid_tokens'].cuda();ve0=prior['valid_embeddings'].cuda();vz0=src['valid_logits'].cuda();st=prior['source_anchor_tokens'].cuda();se=prior['source_anchor_embeddings'].cuda();rows=[];begin=time.monotonic();view=torch.load(RUN/'artifacts/source_view.pt',map_location='cpu',weights_only=True)
    for name in C['slices']:
        ref,query=load_slice(name);m=load_model(seed);rt=tokenize(m,ref);re0,rz0=evaluate(m,rt);qt=tokenize(m,query);qe0,qz0=evaluate(m,qt);y=query['labels'].numpy();bp=qz0.argmax(-1).cpu().numpy();vy=src['valid_labels'].numpy();vb=vz0.argmax(-1).cpu().numpy();votes,_=neighbors(re0,p,bb,prior['calibration']['neighbor_temperature'].cuda());gate,gateinfo=kl_gate(rz0,re0,p,vz0)
        save(f'kl_gate_{seed}_{name}.json',gateinfo);np.savez_compressed(RUN/'artifacts'/f'kl_gate_{seed}_{name}.npz',eligible=gate.cpu().numpy(),rows=ref['rows'].numpy())
        with np.load(ROOT/C['baseline_run']/'artifacts'/f'base_{seed}_{name}.npz') as old:assert np.array_equal(old['logits'],qz0.cpu().numpy()) and np.array_equal(old['rows'],query['rows'].numpy())
        with np.load(ROOT/C['baseline_run']/'artifacts'/f'evidence_{seed}_{name}.npz') as old:qvotes=old['votes']
        nearest=geometry(qe0,p)['nearest'].cpu().numpy();common=(bp!=y)&(bp==nearest)&(qvotes[0].argmax(-1)==bp)&(qvotes[1].argmax(-1)==bp)
        np.savez_compressed(RUN/'artifacts'/f'base_{seed}_{name}.npz',logits=qz0.cpu().numpy(),predictions=bp,labels=y,rows=query['rows'].numpy(),common_wrong=common)
        viewq=probs(ref['tam'],view).cuda();query_view=probs(query['tam'],view).argmax(-1).numpy();np.savez_compressed(RUN/'artifacts'/f'view_{seed}_{name}.npz',reference_q=viewq.cpu().numpy(),reference_rows=ref['rows'].numpy(),query_predictions=query_view,query_rows=query['rows'].numpy())
        view_metrics=dict(query_accuracy=float((query_view==y).mean()),frozen_wrong_corrected=float((query_view[bp!=y]==y[bp!=y]).mean()),common_corrected=float((query_view[common]==y[common]).mean()),common_same_wrong=float((query_view[common]==bp[common]).mean()))
        save(f'view_metrics_{seed}_{name}.json',view_metrics)
        for condition in ['frozen','v6_lora','v8_balance']+C['conditions']:
            start=time.monotonic();torch.cuda.reset_peak_memory_stats();m=load_model(seed);initial=canonical(m);a=None;state=None;hist=[];probes=[]
            if condition=='v6_lora':a=attach(m,'lora');a.load_state_dict(old_adapter(seed,name))
            elif condition=='v8_balance':
                a=attach(m,'lora');a.load_state_dict(torch.load(ROOT/C['v8_run']/'checkpoints'/f'{seed}_{name}_geometry_grad_balance.pt',map_location='cpu',weights_only=True)['adapter'])
            elif condition in C['conditions']:a,state,hist,probes=train(m,rt,re0,rz0,votes,st,se,p,viewq,view['enabled'],condition)
            assert_base_locked(m,initial);re,rz=evaluate(m,rt,a);qe,qz=evaluate(m,qt,a);ve,vz=evaluate(m,vt,a)
            if condition=='v6_lora':
                for prefix,z in [('raw',qz),('valid',vz)]:
                    with np.load(ROOT/C['v6_run']/'artifacts'/f'{prefix}_{seed}_{name}_sharp_target_only.npz') as old:assert np.array_equal(old['logits'],z.cpu().numpy())
            if condition=='v8_balance':
                for prefix,z in [('query',qz),('valid',vz),('reference',rz)]:
                    with np.load(ROOT/C['v8_run']/'artifacts'/f'{prefix}_{seed}_{name}_geometry_grad_balance.npz') as old:assert np.array_equal(old['logits'],z.cpu().numpy())
            for split,z,e,bz,be,data in [('query',qz,qe,qz0,qe0,query),('valid',vz,ve,vz0,ve0,dict(rows=src['valid_rows'],labels=src['valid_labels'])),('reference',rz,re,rz0,re0,ref)]:
                arrays=dict(logits=z.cpu().numpy(),embeddings=e.cpu().numpy(),base_logits=bz.cpu().numpy(),rows=data['rows'].numpy(),**signals(bz,z,be,e,p))
                if split!='reference':arrays['labels']=data['labels'].numpy()
                np.savez_compressed(RUN/'artifacts'/f'{split}_{seed}_{name}_{condition}.npz',**arrays)
            centers,cl=cluster(re,qe,p);np.savez_compressed(RUN/'artifacts'/f'cluster_{seed}_{name}_{condition}.npz',centers=centers,query_clusters=cl,reference_rows=ref['rows'].numpy(),query_rows=query['rows'].numpy())
            cp=qz.argmax(-1).cpu().numpy();met=metrics(y,cp,bp);vm=metrics(vy,vz.argmax(-1).cpu().numpy(),vb);dg=diagnostics(y,bp,cp,cl,geometry(qe,p)['nearest'].cpu().numpy(),common)
            ck=None
            if condition in C['conditions']:
                ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';torch.save(dict(adapter={k:v.detach().cpu() for k,v in a.state_dict().items()},reference_state=state.dump(),condition=condition,seed=seed,slice=name,config_sha256=sha(RUN/'config.json')),ck);save(f'updates_{seed}_{name}_{condition}.json',hist);save(f'gradients_{seed}_{name}_{condition}.json',probes)
            row=dict(seed=seed,slice=name,condition=condition,**met,delta_f1_pp=100*(met['macro_f1']-metrics(y,bp,bp)['macro_f1']),**{'source_valid_'+k:v for k,v in vm.items()},**dg,steps=len(hist),seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),query_sha256=sha(RUN/'artifacts'/f'query_{seed}_{name}_{condition}.npz'),checkpoint_sha256=sha(ck) if ck else None)
            rows.append(row);save(f'scores_{seed}.json',rows);save(f'progress_{seed}.json',dict(tasks=len(rows),total=14,new_adaptations=sum(v['steps']>0 for v in rows),seconds=time.monotonic()-begin));print(json.dumps(row),flush=True)
            del m,a,state,re,rz,qe,qz,ve,vz;torch.cuda.empty_cache()
        del rt,re0,rz0,qt,qe0,qz0,votes;torch.cuda.empty_cache()
    assert len(rows)==14

if __name__=='__main__':
    try:
        setup();verify();seed=int(os.environ['CANDIDATE_SEED']);assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
        if os.environ['MOMENTUM_MODE']=='preflight':preflight(seed)
        else:main(seed)
    except BaseException as e:save(f'failure_{os.environ.get("MOMENTUM_MODE")}_{os.environ.get("CANDIDATE_SEED")}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
