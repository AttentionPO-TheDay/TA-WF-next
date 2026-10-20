"""Source-only fit and mechanical proof, executed independently on three GPUs."""
import json,os,time,warnings
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from runtime import RUN,ROOT,C,setup,load_model,source,save,sha,entropy
from methods import tokenize,evaluate,encode,attach,standardize,distances,route,assert_base_locked

def main():
    setup();seed=int(os.environ['COMBO_SEED']);d=source();m=load_model(seed)
    initial={k:v.cpu().clone() for k,v in m.state_dict().items()}
    all_e=[];all_z=[];start=time.monotonic()
    with torch.no_grad():
        for i in range(0,len(d['rows']),128):
            sub={k:d[k][i:i+128] for k in ('timestamps','tam','rows')}
            t=tokenize(m,sub);e,z=encode(m,t);all_e.append(e.cpu());all_z.append(z.cpu())
    e=torch.cat(all_e);z=torch.cat(all_z);labels=d['labels'].numpy()
    mean=e.mean(0);sd=e.std(0,unbiased=False).clamp_min(.001);x=((e-mean)/sd).numpy()
    fits={};fit_diagnostics=[]
    for k in [1,4]:
        centers=[];covariances=[];priors=[]
        for c in range(102):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                g=GaussianMixture(n_components=k,covariance_type='diag',reg_covar=.001,n_init=3,max_iter=200,random_state=20261019).fit(x[labels==c])
            assert g.converged_,(seed,k,c)
            centers.append(g.means_);covariances.append(g.covariances_);priors.append(g.weights_)
            fit_diagnostics.append({'k':k,'class':c,'n':int((labels==c).sum()),'converged':bool(g.converged_),'iterations':int(g.n_iter_),'warnings':[str(w.message) for w in caught]})
        p={'mean':mean.cuda(),'sd':sd.cuda(),'centers':torch.tensor(np.stack(centers),dtype=torch.float32,device='cuda'),'covariances':torch.tensor(np.stack(covariances),dtype=torch.float32,device='cuda'),'priors':torch.tensor(np.stack(priors),dtype=torch.float32,device='cuda')}
        wmins=[];dmins=[]
        for i in range(0,len(x),128):
            w,ds=distances(torch.tensor(x[i:i+128],device='cuda'),p);wmins.append(w.flatten(1).min(1).values);dmins.append(ds.flatten(1).min(1).values)
        ws=torch.cat(wmins);ds=torch.cat(dmins)
        p.update(temperature=ws.median().clamp_min(.001),distance_scale=ds.median().clamp_min(.001),distance_cutoff=torch.quantile(ds,.99),entropy_cutoff=torch.quantile(entropy(z.cuda()),.95))
        assert all(torch.isfinite(v).all() for v in p.values())
        fits[str(k)]={n:v.cpu() for n,v in p.items()}
    # Exact reconstruction plus zero-initialization and useful Q/V gradient checks, only pure-source rows.
    t=tokenize(m,{k:d[k][:16] for k in ('timestamps','tam','rows')});base_e,base_z=evaluate(m,t)
    with torch.no_grad():native=m(d['timestamps'][:16].cuda(),d['tam'][:16].cuda())
    assert torch.equal(native,base_z),(native-base_z).abs().max().item()
    checks={}
    for kind in ['residual','lora']:
        test=load_model(seed);adapter=attach(test,kind)
        with torch.no_grad():_,out_z=encode(test,t,adapter)
        assert torch.equal(out_z,base_z),(kind,float((out_z-base_z).abs().max()))
        out_e,out_z=encode(test,t,adapter)
        p={n:v.cuda() for n,v in fits['4'].items()};xx=standardize(out_e,p);teacher_labels=base_z.argmax(-1)
        loss=(xx-p['centers'][teacher_labels,0]).square().mean();loss.backward()
        g=sum(float(q.grad.square().sum()) for q in adapter.parameters() if q.grad is not None)
        assert g>0 and np.isfinite(g),(kind,g)
        assert all(q.grad is None for n,q in test.named_parameters() if '.parametrizations.in_proj_weight.0.' not in n)
        assert_base_locked(test,initial)
        count=sum(q.numel() for q in adapter.parameters())
        checks[kind]={'parameters':count,'fraction_base':count/sum(v.numel() for v in m.parameters()),'zero_init_bitwise_parity':True,'gradient_square_sum':g,'base_locked':True}
        if kind=='lora':
            for layer in test.head:
                prm=layer.self_attn.parametrizations.in_proj_weight
                a=prm[0]
                assert a.bq.grad is not None and a.bq.grad.abs().sum()>0
                assert a.bv.grad is not None and a.bv.grad.abs().sum()>0
                with torch.no_grad():a.bq.add_(a.bq.grad,alpha=-.001);a.bv.add_(a.bv.grad,alpha=-.001)
                delta=prm[0](prm.original)-prm.original
                assert torch.equal(delta[128:256],torch.zeros_like(delta[128:256]))
                assert delta[:128].abs().sum()>0 and delta[256:].abs().sum()>0
            with torch.no_grad():changed=encode(test,t,adapter)[1]
            assert (changed-base_z).abs().max()>0
            checks[kind]['nonzero_update_changes_logits']=True;checks[kind]['K_exactly_unchanged']=True
        del test,adapter
    order=sorted(range(len(d['rows'])),key=lambda i:__import__('hashlib').sha256(str(int(d['rows'][i])).encode()).hexdigest())[:1024]
    torch.save({'fits':fits,'source_embeddings':e,'source_logits':z,'source_rows':d['rows'],'source_labels':d['labels'],'anchor_indices':torch.tensor(order),'seed':seed,'feature_space':C['adaptation']['feature_space']},RUN/'artifacts'/f'source_gmm_{seed}.pt')
    save(f'source_preflight_{seed}.json',{'passed':True,'seed':seed,'source_rows':len(e),'checks':checks,'fits':fit_diagnostics,'seconds':time.monotonic()-start,'artifact_sha256':sha(RUN/'artifacts'/f'source_gmm_{seed}.pt')})
    print(json.dumps({'seed':seed,'passed':True,'checks':checks}),flush=True)

if __name__=='__main__':main()
