"""Full-query reload audit plus independently recomputed routing, scores and metrics."""
import os,json,time
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,verify,load_model,load_slice,sha,save
from methods import tokenize,evaluate,attach,assert_base_locked
from candidate import gpu_dict
from routing import geometry,scores,select_threshold

def independently_metrics(y,p,b):
    correct=b==y;wrong=~correct;cm=np.zeros((102,102),dtype=np.int64);np.add.at(cm,(y,p),1);tp=cm.diagonal();den=cm.sum(0)+cm.sum(1);f1=np.divide(2*tp,den,out=np.zeros(102),where=den>0).mean()
    return {'accuracy':float((p==y).mean()),'macro_f1':float(f1),'nar':float((p[correct]!=y[correct]).mean()),'recovery':float((p[wrong]==y[wrong]).mean())}

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-80,80)))

def main():
    setup();verify();seed=int(os.environ['CANDIDATE_SEED']);begin=time.monotonic();source=torch.load(RUN/'artifacts'/f'source_{seed}.pt',weights_only=True,map_location='cpu');cal=source['calibration'];vt=source['valid_tokens'].cuda();vbase=source['valid_logits'].cuda();vg=gpu_dict(source['valid_geometry'])
    art=torch.load(ROOT/C['previous_run']/'artifacts'/f'source_gmm_{seed}.pt',weights_only=True,map_location='cpu');p=gpu_dict(art['fits']['1']);rs=[]
    for mode in ['controls','adapted','shuffle']:
        path=RUN/'artifacts'/f'{mode}_scores_{seed}.json'
        if path.exists():rs+=json.loads(path.read_text())
    states=0;reloaded=0;records=[];methods=list(dict.fromkeys(r['condition'] for r in rs))
    for name in C['slices']:
        _,query=load_slice(name);m=load_model(seed);qt=tokenize(m,query);qe,qz=evaluate(m,qt);bn=qz.argmax(-1).cpu().numpy();y=query['labels'].numpy();g=geometry(qe,p)
        with np.load(ROOT/C['baseline_run']/'artifacts'/f'base_{seed}_{name}.npz') as b:assert np.array_equal(b['logits'],qz.cpu().numpy())
        with np.load(RUN/'artifacts'/f'geometry_{seed}_{name}.npz') as a:
            for k in g:assert np.array_equal(a[k],g[k].cpu().numpy())
        for condition in methods:
            rrows=[r for r in rs if r['slice']==name and r['condition']==condition];assert len(rrows)==3
            with np.load(RUN/'artifacts'/f'raw_{seed}_{name}_{condition}.npz') as a:z=torch.from_numpy(a['logits']).cuda();cp=a['predictions'];assert np.array_equal(y,a['labels']) and np.array_equal(query['rows'].numpy(),a['rows'])
            with np.load(RUN/'artifacts'/f'valid_{seed}_{name}_{condition}.npz') as a:vz=torch.from_numpy(a['logits']).cuda()
            if not rrows[0]['reused_control']:
                ck=RUN/'checkpoints'/f'{seed}_{name}_{condition}.pt';state=torch.load(ck,map_location='cpu',weights_only=True);assert state['config_sha256']==sha(RUN/'config.json') and state['source_sha256']==sha(RUN/'artifacts'/f'source_{seed}.pt')
                m=load_model(seed);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};a=attach(m,'lora');a.load_state_dict(state['adapter']);_,zz=evaluate(m,qt,a);_,vv=evaluate(m,vt,a)
                assert torch.equal(zz,z) and torch.equal(vv,vz);assert_base_locked(m,initial);assert sha(ck)==rrows[0]['checkpoint_sha256'];reloaded+=1
                with np.load(RUN/'artifacts'/f'reference_weights_{seed}_{name}_{condition}.npz') as w:
                    hh=w['entropy'];pmr=w['pmr'];maha=w['maha'];low,high=np.quantile(hh,[.2,.9]);s=max(.1*(high-low),.02);ww=((hh>low)&(hh<=high)).astype(np.float32) if condition=='hard_band' else sigmoid((hh-low)/s)*sigmoid((high-hh)/s)
                    if condition.endswith('soft_geometry'):ww*=sigmoid((pmr-cal['pmr25'])/.05)*np.exp(-maha/cal['distance_scale'])*(maha<=cal['distance_cutoff'])
                    assert np.allclose(ww,w['weights'],rtol=1e-5,atol=1e-6)
                hist=json.loads((RUN/'artifacts'/f'updates_{seed}_{name}_{condition}.json').read_text());assert len(hist)==400 and all(np.isfinite(h['loss']) and np.isfinite(h['gradient_norm']) for h in hist)
            ss=scores(qz,z,g,cal);vs=scores(vbase,vz,vg,cal)
            with np.load(RUN/'artifacts'/f'signal_{seed}_{name}_{condition}.npz') as a:
                for key in ss:assert np.array_equal(a[key],ss[key].cpu().numpy())
            # Independent NumPy normalized margins and geometric score.
            raw=z.cpu().numpy().astype(np.float64);base=qz.cpu().numpy().astype(np.float64);idx=np.arange(len(cp));d=g['cosine'].cpu().numpy();dc=d[idx,cp];other=d.copy();other[idx,cp]=np.inf;other=other.min(1);gm=(other-dc)/(other+1e-8)
            def margin(zz):
                zz=zz/np.sqrt(np.mean(zz**2,axis=1,keepdims=True)+1e-12);v=zz[idx,cp].copy();zz[idx,cp]=-np.inf;return v-zz.max(1)
            gain=margin(raw)-margin(base);maha=g['maha'].cpu().numpy();combined=sigmoid(gm/.05)*sigmoid(gain/.05)*np.exp(-maha/cal['distance_scale']);combined[maha>cal['distance_cutoff']]=-1
            assert np.allclose(combined,ss['combined'].cpu().numpy(),atol=2e-5,rtol=1e-4)
            thresholds=json.loads((RUN/'artifacts'/f'thresholds_{seed}_{name}_{condition}.json').read_text())
            for r in rrows:
                policy=r['policy']
                if policy=='raw':keep=cp!=bn;pred=cp
                else:
                    th=select_threshold(source['valid_labels'].cuda(),vbase.argmax(-1),vz.argmax(-1),vs['combined'],.01 if policy=='source_nar1' else .02);assert th==thresholds[policy]
                    keep=(cp!=bn)&(ss['combined'].cpu().numpy()>=th['threshold']) if th['enabled'] else np.zeros(len(cp),dtype=bool);pred=np.where(keep,cp,bn)
                    assert np.array_equal(pred[~keep],bn[~keep])
                with np.load(RUN/'artifacts'/f'policy_{seed}_{name}_{condition}_{policy}.npz') as a:assert np.array_equal(pred,a['predictions']) and np.array_equal(keep,a['keep'])
                mm=independently_metrics(y,pred,bn)
                for k in mm:assert abs(mm[k]-r[k])<1e-12,(seed,name,condition,policy,k,mm[k],r[k])
                assert abs(r['delta_f1_pp']-100*(mm['macro_f1']-independently_metrics(y,bn,bn)['macro_f1']))<1e-10
                states+=1
            records.append({'slice':name,'condition':condition,'full_query_reloaded':not rrows[0]['reused_control'],'scores_and_metrics_independent':True})
        print(f'audit seed={seed} slice={name} states={states}',flush=True);del qt,qe,qz,g;torch.cuda.empty_cache()
    save(f'audit_{seed}.json',{'passed':True,'seed':seed,'states':states,'new_checkpoints_full_query_reloaded':reloaded,'seconds':time.monotonic()-begin,'records':records})

if __name__=='__main__':main()
