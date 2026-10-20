"""One seed, all frozen cells, no query-based method or checkpoint selection."""
from contextlib import nullcontext
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,sha,save,load_model,source,reference,load_slice,predict,metrics
from mechanisms import adapt,get_tokens
from token_alignment import aligned_generator

def main():
    setup();verify();seed=int(os.environ['DIAGNOSTIC_SEED'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
    assert json.loads((RUN/'artifacts/source_smoke.json').read_text())['passed']
    s=source();ref=reference(seed)
    order=sorted(range(len(s['rows'])),key=lambda i:hashlib.sha256(str(int(s['rows'][i])).encode()).hexdigest())[:1024]
    anchors={k:s[k][order] for k in ('timestamps','tam','rows')};del s
    score_path=RUN/'artifacts'/f'scores_{seed}.json';scores=json.loads(score_path.read_text()) if score_path.exists() else []
    started=time.monotonic()
    for name in C['slices']:
        support,query=load_slice(name);base=None
        for condition in C['conditions']:
            key=f'{seed}_{name}_{condition}';pred_path=RUN/'artifacts'/f'pred_{key}.npz';ck_path=RUN/'checkpoints'/f'{key}.pt'
            done=next((r for r in scores if r['slice']==name and r['condition']==condition),None)
            if done:
                assert sha(pred_path)==done['prediction_sha256'] and sha(ck_path)==done['checkpoint_sha256']
                if condition=='frozen':
                    with np.load(pred_path) as z:base=z['predictions'].copy()
                continue
            m=load_model(seed);begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
            aligner,history,trainable=adapt(m,support,condition,ref,anchors)
            with aligned_generator(m,aligner) if aligner is not None else nullcontext():z=predict(m,query)
            p=z.argmax(1);y=query['labels'].numpy()
            if condition=='frozen':base=p
            assert base is not None
            met=metrics(y,p,base);base_f1=metrics(y,base,base)['macro_f1']
            with pred_path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,logits=z,predictions=p,labels=y,rows=query['rows'].numpy())
            pred_path.with_suffix('.tmp').replace(pred_path)
            state={'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'aligner':{k:v.detach().cpu() for k,v in aligner.state_dict().items()} if aligner is not None else None,'seed':seed,'slice':name,'condition':condition,'config_sha256':sha(RUN/'config.json')}
            torch.save(state,ck_path)
            r={'seed':seed,'slice':name,'condition':condition,**met,'delta_f1_pp':100*(met['macro_f1']-base_f1),'query_rows':len(y),'trainable_names':trainable,'steps':len(history),'seconds':time.monotonic()-begin,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'prediction_sha256':sha(pred_path),'checkpoint_sha256':sha(ck_path)}
            if name=='behavior_subpage':
                from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
                r['hostname_nmi']=float(normalized_mutual_info_score(query['hosts'],p));r['hostname_ari']=float(adjusted_rand_score(query['hosts'],p))
            if seed==C['visualization']['seed'] and name in C['visualization']['slices'] and condition in ('frozen','prototype_anchor'):
                selected=[]
                for label in C['visualization']['classes']:
                    ix=np.flatnonzero(y==label)
                    selected.extend(sorted(ix,key=lambda i:hashlib.sha256(str(int(query['rows'][i])).encode()).hexdigest())[:C['visualization']['max_per_class']])
                d={k:query[k][selected] for k in ('timestamps','tam','rows')}
                with aligned_generator(m,aligner) if aligner is not None else nullcontext():features=get_tokens(m,d).mean(1).cpu()
                torch.save({'features':features,'labels':query['labels'][selected],'rows':d['rows']},RUN/'artifacts'/f'visual_{name}_{condition}.pt')
            save(f'updates_{key}.json',history);scores.append(r);save(f'scores_{seed}.json',scores)
            save(f'progress_{seed}.json',{'completed':len(scores),'total':70,'last':key,'wall_seconds':time.monotonic()-started})
            print(json.dumps(r),flush=True);del m,aligner;torch.cuda.empty_cache()
            if time.monotonic()-started>C['budget']['max_wall_seconds']:raise TimeoutError('worker budget')
    assert len(scores)==70

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(f'failure_{os.environ.get("DIAGNOSTIC_SEED","unknown")}.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
