"""Two observed slices,18 adaptations, frozenBN and independent query inference."""
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
from decoupling import train,assemble,grads
from shared import inputs,canonical,old_adapter,signals,kl_gate,cluster,diagnostics


def preflight(seed):
    p,bb,prior,src=inputs(seed);m=load_model(seed);initial=canonical(m);t=src['valid_tokens'].cuda();e,z=evaluate(m,t);assert torch.equal(z.cpu(),src['valid_logits'])
    a=attach(m,'lora');ix=torch.arange(64,device='cuda');qf=target_distribution(z,src['valid_votes'].cuda(),'soft_neighbor');state=RefinedState(e,z,p,'sharp_target_only');q=state.target(qf,ix);w,_=state.weight(ix,True);st=prior['source_anchor_tokens'].cuda()[:32];se=prior['source_anchor_embeddings'].cuda()[:32]
    legacy,x,zz,parts,n=loss_parts(m,a,t[ix],q,w,st,se,p);old,_,_,_=objective(m,a,t[ix],q,w,state.source,st,se,p,False);assert torch.equal(legacy,old)
    gg=grads(legacy,list(a.parameters()));og=grads(old,list(a.parameters()));assert torch.equal(gg,og)
    temp=C['adaptation']['kl_temperature'];per=torch.nn.functional.kl_div((zz/temp).log_softmax(-1),q,reduction='none').sum(-1)*temp**2;ones=torch.ones_like(w);zeros=torch.zeros_like(w)
    no,_,_=assemble(parts,per,w,ones,'no_teacher_kl',list(a.parameters()),None);expected=.1*parts['pull']+parts['relation']+parts['identity'];assert torch.allclose(grads(no,list(a.parameters())),grads(expected,list(a.parameters())),atol=1e-7,rtol=1e-6)
    gated,_,_=assemble(parts,per,w,ones,'confident_teacher_kl',list(a.parameters()),None);assert torch.allclose(legacy,gated,atol=1e-6,rtol=1e-6)
    empty,_,_=assemble(parts,per,w,zeros,'confident_teacher_kl',list(a.parameters()),None);assert torch.allclose(empty,expected,atol=1e-6)
    bal,ema,d=assemble(parts,per,w,ones,'geometry_grad_balance',list(a.parameters()),None);expect=max(1.,min(50.,d['kl_gradient_norm']/(d['geometry_gradient_norm']+1e-8)));assert abs(expect-d['geometry_weight'])<1e-12
    explicit=parts['kl']+expect*(.1*parts['pull']+parts['relation'])+parts['identity'];assert torch.equal(grads(bal,list(a.parameters())),grads(explicit,list(a.parameters())))
    assert_base_locked(m,initial);assert sum(v.numel() for v in a.parameters())==4096
    assert all(all(not t.requires_grad for t in v.parameters()) and not v.training for v in m.modules() if isinstance(v,torch.nn.modules.batchnorm._BatchNorm))
    save(f'preflight_{seed}.json',dict(passed=True,source_logits_exact=True,v6_loss_gradient_exact=True,no_kl_geometry_identity_gradient=True,zero_full_gate_checks=True,balance_detached_gradient_exact=True,pure_bn_locked=True,parameters=4096))


def main(seed):
    p,bb,prior,src=inputs(seed);vt=src['valid_tokens'].cuda();ve0=prior['valid_embeddings'].cuda();vz0=src['valid_logits'].cuda();st=prior['source_anchor_tokens'].cuda();se=prior['source_anchor_embeddings'].cuda();rows=[];begin=time.monotonic()
    for name in C['slices']:
        ref,query=load_slice(name);m=load_model(seed);rt=tokenize(m,ref);re0,rz0=evaluate(m,rt);qt=tokenize(m,query);qe0,qz0=evaluate(m,qt);y=query['labels'].numpy();bp=qz0.argmax(-1).cpu().numpy();vy=src['valid_labels'].numpy();vb=vz0.argmax(-1).cpu().numpy();votes,_=neighbors(re0,p,bb,prior['calibration']['neighbor_temperature'].cuda());gate,gateinfo=kl_gate(rz0,re0,p,vz0)
        save(f'kl_gate_{seed}_{name}.json',gateinfo);np.savez_compressed(RUN/'artifacts'/f'kl_gate_{seed}_{name}.npz',eligible=gate.cpu().numpy(),rows=ref['rows'].numpy())
        with np.load(ROOT/C['baseline_run']/'artifacts'/f'base_{seed}_{name}.npz') as old:assert np.array_equal(old['logits'],qz0.cpu().numpy()) and np.array_equal(old['rows'],query['rows'].numpy())
        with np.load(ROOT/C['baseline_run']/'artifacts'/f'evidence_{seed}_{name}.npz') as old:qvotes=old['votes']
        nearest=geometry(qe0,p)['nearest'].cpu().numpy();common=(bp!=y)&(bp==nearest)&(qvotes[0].argmax(-1)==bp)&(qvotes[1].argmax(-1)==bp)
        np.savez_compressed(RUN/'artifacts'/f'base_{seed}_{name}.npz',logits=qz0.cpu().numpy(),predictions=bp,labels=y,rows=query['rows'].numpy(),common_wrong=common)
        for condition in ['frozen','v6_lora']+C['conditions']:
            start=time.monotonic();torch.cuda.reset_peak_memory_stats();m=load_model(seed);initial=canonical(m);a=None;state=None;hist=[];probes=[]
            if condition=='v6_lora':a=attach(m,'lora');a.load_state_dict(old_adapter(seed,name))
            elif condition in C['conditions']:a,state,hist,probes=train(m,rt,re0,rz0,votes,st,se,p,gate,condition)
            assert_base_locked(m,initial);re,rz=evaluate(m,rt,a);qe,qz=evaluate(m,qt,a);ve,vz=evaluate(m,vt,a)
            if condition=='v6_lora':
                for prefix,z in [('raw',qz),('valid',vz)]:
                    with np.load(ROOT/C['v6_run']/'artifacts'/f'{prefix}_{seed}_{name}_sharp_target_only.npz') as old:assert np.array_equal(old['logits'],z.cpu().numpy())
            for split,z,e,bz,be,data in [('query',qz,qe,qz0,qe0,query),('valid',vz,ve,vz0,ve0,dict(rows=src['valid_rows'],labels=src['valid_labels'])),('reference',rz,re,rz0,re0,ref)]:
                arrays=dict(logits=z.cpu().numpy(),embeddings=e.cpu().numpy(),base_logits=bz.cpu().numpy(),rows=data['rows'].numpy(),**signals(bz,z,be,e,p))
                if split!='reference':arrays['labels']=data['labels'].numpy()
                np.savez_compressed(RUN/'artifacts'/f'{split}_{seed}_{name}_{condition}.npz',**arrays)
            centers,cl=cluster(re,qe,p);np.savez_compressed(RUN/'artifacts'/f'cluster_{seed}_{name}_{condition}.npz',centers=centers,query_clusters=cl,reference_rows=ref['rows'].numpy(),query_rows=query['rows'].numpy())
            cp=qz.argmax(-1).cpu().numpy();met=metrics(y,cp,bp);vm=metrics(vy,vz.argmax(-1).cpu().numpy(),vb);dg=diagnostics(y,bp,cp,cl,geometry(qe,p)['nearest'].cpu().numpy(),common)
            ck=None
            if a is not None and condition!='v6_lora':
                ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';torch.save(dict(adapter={k:v.detach().cpu() for k,v in a.state_dict().items()},reference_state=state.dump(),condition=condition,seed=seed,slice=name,config_sha256=sha(RUN/'config.json')),ck);save(f'updates_{seed}_{name}_{condition}.json',hist);save(f'gradients_{seed}_{name}_{condition}.json',probes)
            row=dict(seed=seed,slice=name,condition=condition,**met,delta_f1_pp=100*(met['macro_f1']-metrics(y,bp,bp)['macro_f1']),**{'source_valid_'+k:v for k,v in vm.items()},**dg,steps=len(hist),seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),query_sha256=sha(RUN/'artifacts'/f'query_{seed}_{name}_{condition}.npz'),checkpoint_sha256=sha(ck) if ck else None)
            rows.append(row);save(f'scores_{seed}.json',rows);save(f'progress_{seed}.json',dict(tasks=len(rows),total=10,new_adaptations=sum(v['steps']>0 for v in rows),seconds=time.monotonic()-begin));print(json.dumps(row),flush=True)
            del m,a,state,re,rz,qe,qz,ve,vz;torch.cuda.empty_cache()
        del rt,re0,rz0,qt,qe0,qz0,votes;torch.cuda.empty_cache()
    assert len(rows)==10

if __name__=='__main__':
    try:
        setup();verify();seed=int(os.environ['CANDIDATE_SEED']);assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
        if os.environ['MOMENTUM_MODE']=='preflight':preflight(seed)
        else:main(seed)
    except BaseException as e:save(f'failure_{os.environ.get("MOMENTUM_MODE")}_{os.environ.get("CANDIDATE_SEED")}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
