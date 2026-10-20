"""One physical GPU per source seed; fixed 80 cells, no target feedback selection."""
import hashlib,json,os,time,traceback
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,source,load_model,load_slice,save,sha,metrics
from methods import tokenize,evaluate,train,route,assert_base_locked


def main():
    setup();verify();seed=int(os.environ['COMBO_SEED'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
    pre=json.loads((RUN/'artifacts'/f'source_preflight_{seed}.json').read_text());assert pre['passed']
    art=torch.load(RUN/'artifacts'/f'source_gmm_{seed}.pt',weights_only=True,map_location='cpu')
    ds=source();ix=art['anchor_indices'];m=load_model(seed)
    source_t=tokenize(m,{k:ds[k][ix] for k in ('timestamps','tam','rows')});source_e=art['source_embeddings'][ix].cuda();del ds
    pmap={k:{n:v.cuda() for n,v in p.items()} for k,p in art['fits'].items()}
    score_path=RUN/'artifacts'/f'scores_{seed}.json';scores=json.loads(score_path.read_text()) if score_path.exists() else []
    started=time.monotonic()
    for name in C['slices']:
        support,query=load_slice(name)
        m=load_model(seed);reference_t=tokenize(m,support);te,tz=evaluate(m,reference_t)
        # Query inputs only used in fixed model inference, never the optimizer.
        qt=tokenize(m,{k:query[k] for k in ('timestamps','tam','rows')});qe,qz=evaluate(m,qt)
        base=qz.argmax(-1).cpu().numpy();y=query['labels'].numpy();base_f1=metrics(y,base,base)['macro_f1']
        base_file=RUN/'artifacts'/f'base_{seed}_{name}.npz'
        if not base_file.exists():np.savez_compressed(base_file,logits=qz.cpu().numpy(),predictions=base,labels=y,rows=query['rows'].numpy())
        for condition in C['conditions']:
            key=f'{seed}_{name}_{condition}';pred_path=RUN/'artifacts'/f'pred_{key}.npz';ck=RUN/'checkpoints'/f'{key}.pt'
            done=next((r for r in scores if r['slice']==name and r['condition']==condition),None)
            if done:
                assert sha(pred_path)==done['prediction_sha256'] and sha(ck)==done['checkpoint_sha256'];continue
            begin=time.monotonic();torch.cuda.reset_peak_memory_stats();m=load_model(seed)
            initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            k='4' if condition.startswith('gmm4') else '1';p=pmap[k]
            adapter,history,refinfo=train(m,reference_t,te,tz,source_t,source_e,p,condition)
            assert_base_locked(m,initial)
            assert all(torch.isfinite(v).all() for v in adapter.state_dict().values())
            after_e,after_z=evaluate(m,qt,adapter);w,keep,d,h=route(qe,qz,p)
            routed='_routed_' in condition
            if routed:after_z=torch.where(keep[:,None],after_z,qz);after_e=torch.where(keep[:,None],after_e,qe)
            bypass=(~keep) if routed else torch.zeros_like(keep)
            assert torch.equal(after_z[bypass],qz[bypass])
            logits=after_z.cpu().numpy();pred=logits.argmax(-1);met=metrics(y,pred,base)
            with pred_path.open('wb') as f:np.savez_compressed(f,logits=logits,predictions=pred,labels=y,rows=query['rows'].numpy(),keep=keep.cpu().numpy(),weight=w.cpu().numpy(),distance=d.cpu().numpy(),entropy=h.cpu().numpy(),bypass=bypass.cpu().numpy())
            state={'adapter':{n:v.detach().cpu() for n,v in adapter.state_dict().items()},'seed':seed,'slice':name,'condition':condition,'config_sha256':sha(RUN/'config.json'),'source_gmm_sha256':pre['artifact_sha256']}
            torch.save(state,ck)
            r={'seed':seed,'slice':name,'condition':condition,**met,'delta_f1_pp':100*(met['macro_f1']-base_f1),'query_rows':len(y),'frozen_macro_f1':base_f1,'source_parameters':sum(v.numel() for v in initial.values()),'adapter_parameters':sum(v.numel() for v in adapter.parameters()),'steps':len(history),'updates':sum(r['updated'] for r in history),'seconds':time.monotonic()-begin,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'query_gate_coverage':float(keep.float().mean()),'query_bypassed':int(bypass.sum()),'query_bypass_bitwise_verified':True,'reference_routing':refinfo,'base_locked':True,'prediction_sha256':sha(pred_path),'checkpoint_sha256':sha(ck)}
            if seed==C['visualization']['seed'] and name in C['visualization']['slices'] and condition in C['visualization']['conditions']:
                selected=sorted(range(len(y)),key=lambda i:hashlib.sha256(str(int(query['rows'][i])).encode()).hexdigest())[:400]
                torch.save({'before':qe[selected].cpu(),'after':after_e[selected].cpu(),'labels':query['labels'][selected],'rows':query['rows'][selected],'keep':keep[selected].cpu(),'condition':condition},RUN/'artifacts'/f'visual_{name}_{condition}.pt')
            save(f'updates_{key}.json',history);scores.append(r);save(f'scores_{seed}.json',scores)
            save(f'progress_{seed}.json',{'completed':len(scores),'total':80,'last':key,'seconds':time.monotonic()-started,'pid':os.getpid(),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']})
            print(json.dumps(r),flush=True)
            del m,adapter,after_e,after_z;torch.cuda.empty_cache()
            if time.monotonic()-begin>C['budget']['max_cell_seconds']:raise TimeoutError('cell budget')
            if time.monotonic()-started>C['budget']['max_wall_seconds']:raise TimeoutError('worker budget')
        del qt,qe,qz,reference_t,te,tz,query,support;torch.cuda.empty_cache()
    assert len(scores)==80

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(f'failure_{os.environ.get("COMBO_SEED","unknown")}.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
