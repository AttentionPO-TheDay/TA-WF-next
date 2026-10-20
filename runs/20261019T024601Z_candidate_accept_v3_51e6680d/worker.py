"""One seed and GPU, complete fixed methods; no target-driven threshold selection."""
import hashlib,json,os,time,traceback
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,load_model,load_slice,save,sha,metrics,predict,configure
from methods import tokenize,evaluate,assert_base_locked
from candidate import gpu_dict,banks,acceptance_inputs,policy_logits,train_candidate
from tent import train_tent,reference_statistics,check_tent_scope


def main():
    setup();verify();seed=int(os.environ['CANDIDATE_SEED']);assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
    pre=json.loads((RUN/'artifacts'/f'preflight_{seed}.json').read_text());assert pre['passed']
    old=ROOT/C['previous_run'];art=torch.load(old/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True);p=gpu_dict(art['fits']['1']);bank_data=banks(art,p)
    caldata=torch.load(RUN/'artifacts'/f'source_calibration_{seed}.pt',map_location='cpu',weights_only=True);cal=gpu_dict(caldata['calibration']);source_t=caldata['source_anchor_tokens'].cuda();source_e=caldata['source_anchor_embeddings'].cuda()
    path=RUN/'artifacts'/f'scores_{seed}.json';scores=json.loads(path.read_text()) if path.exists() else [];start=time.monotonic();trained=0
    for name in C['slices']:
        reference,query=load_slice(name);m=load_model(seed);qt=tokenize(m,query);qe,qz=evaluate(m,qt);rt=tokenize(m,reference);re,rz=evaluate(m,rt)
        info=acceptance_inputs(qe,qz,p,bank_data,cal);rinfo=acceptance_inputs(re,rz,p,bank_data,cal)
        y=query['labels'].numpy();base=qz.argmax(-1).cpu().numpy();bf=metrics(y,base,base)['macro_f1']
        np.savez_compressed(RUN/'artifacts'/f'base_{seed}_{name}.npz',logits=qz.cpu().numpy(),predictions=base,labels=y,rows=query['rows'].numpy())
        # Candidate evidence depends on frozen source/reference inference only; no query aggregation.
        np.savez_compressed(RUN/'artifacts'/f'evidence_{seed}_{name}.npz',votes=info['votes'].cpu().numpy(),distance=info['distance'].cpu().numpy(),entropy=info['entropy'].cpu().numpy(),top5=info['top5'].cpu().numpy())
        for condition in C['conditions']:
            if any(r['slice']==name and r['condition']==condition and r['policy']=='raw' for r in scores):continue
            begin=time.monotonic();torch.cuda.reset_peak_memory_stats();m=load_model(seed);original={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};adapter=None;history=[];detail={};ck=None
            if condition=='frozen':e,z=qe,qz
            elif condition=='tent_bn':
                history=train_tent(m,reference);trained+=1;z=torch.from_numpy(predict(m,query)).cuda();e=None
                detail={'trainable_names':check_tent_scope(m,original,condition),'query_protocol':'batch-stat transductive'}
            elif condition=='tent_bn_reference':
                ckprev=RUN/'checkpoints'/f'{seed}_{name}_tent_bn.pt';state=torch.load(ckprev,map_location='cpu',weights_only=True);configure(m,'tent_bn');m.load_state_dict(state['model'],strict=True)
                m,notes=reference_statistics(m,reference,seed);z=torch.from_numpy(predict(m,query)).cuda();e=None
                detail={'trainable_names':check_tent_scope(m,original,condition),'query_protocol':'strict independent reference statistics','reference_bn_calibration':notes,'training_reused_from':'tent_bn'}
            else:
                adapter,history,detail=train_candidate(m,rt,re,rz,rinfo['votes'],source_t,source_e,p,condition);trained+=1
                assert_base_locked(m,original);e,z=evaluate(m,qt,adapter)
                assert all(torch.isfinite(v).all() for v in adapter.state_dict().values())
            assert torch.isfinite(z).all()
            if condition!='frozen':
                ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt'
                torch.save({'adapter':{n:v.detach().cpu() for n,v in adapter.state_dict().items()} if adapter is not None else None,'model':{n:v.detach().cpu() for n,v in m.state_dict().items()} if adapter is None else None,'seed':seed,'slice':name,'condition':condition,'config_sha256':sha(RUN/'config.json'),'calibration_sha256':pre['calibration_sha256']},ck)
            rawfile=RUN/'artifacts'/f'raw_{seed}_{name}_{condition}.npz'
            np.savez_compressed(rawfile,logits=z.cpu().numpy(),predictions=z.argmax(-1).cpu().numpy(),labels=y,rows=query['rows'].numpy())
            policies=C['policies'] if adapter is not None else ['raw']
            for policy in policies:
                zz,keep=policy_logits(z,qz,info,cal,policy);pred=zz.argmax(-1).cpu().numpy();met=metrics(y,pred,base)
                out=RUN/'artifacts'/f'policy_{seed}_{name}_{condition}_{policy}.npz'
                np.savez_compressed(out,predictions=pred,keep=keep.cpu().numpy(),rows=query['rows'].numpy())
                if policy!='raw':assert torch.equal(zz[~keep],qz[~keep])
                r={'seed':seed,'slice':name,'condition':condition,'policy':policy,**met,'delta_f1_pp':100*(met['macro_f1']-bf),'frozen_macro_f1':bf,'query_rows':len(y),'changed_predictions':int((pred!=base).sum()),'accepted_rows':int(keep.sum()),'raw_changed_predictions':int((z.argmax(-1)!=qz.argmax(-1)).sum()),'steps':len(history),'training_reused_from':detail.get('training_reused_from'),'parameters':sum(v.numel() for v in adapter.parameters()) if adapter is not None else sum(v.numel() for v in m.parameters() if v.requires_grad),'seconds':time.monotonic()-begin,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'detail':detail,'raw_sha256':sha(rawfile),'policy_sha256':sha(out),'checkpoint_sha256':sha(ck) if ck else None,'rejected_logits_bitwise_verified':True}
                scores.append(r)
            save(f'updates_{seed}_{name}_{condition}.json',history);save(f'scores_{seed}.json',scores)
            if seed==42 and name in C['visualization']['slices'] and condition in C['visualization']['conditions']:
                selected=sorted(range(len(y)),key=lambda i:hashlib.sha256(str(int(query['rows'][i])).encode()).hexdigest())[:400]
                torch.save({'before':qe[selected].cpu(),'after_raw':e[selected].cpu(),'labels':query['labels'][selected],'rows':query['rows'][selected],'keep':policy_logits(z,qz,info,cal,'evidence_accept')[1][selected].cpu()},RUN/'artifacts'/f'visual_{name}_{condition}.pt')
            save(f'progress_{seed}.json',{'score_states':len(scores),'total':210,'slice':name,'condition':condition,'training_tasks':trained,'seconds':time.monotonic()-start,'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES']})
            print(json.dumps({'seed':seed,'slice':name,'condition':condition,'score_states':len(scores),'last':scores[-1]}),flush=True)
            del m,adapter,e,z;torch.cuda.empty_cache()
            if time.monotonic()-start>C['budget']['max_wall_seconds']:raise TimeoutError('worker budget')
        del qt,qe,qz,rt,re,rz,info,rinfo,reference,query;torch.cuda.empty_cache()
    assert len(scores)==210

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(f'failure_{os.environ.get("CANDIDATE_SEED","unknown")}.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
