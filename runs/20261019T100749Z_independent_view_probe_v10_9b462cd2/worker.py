import os,time,math,json,traceback
import numpy as np
import torch
import torch.nn.functional as F
from runtime import RUN,ROOT,C,sha,save,verify,setup,source,target,metrics
from model import ViewNet,DeterministicPool,features,predict

def preflight(seed):
    for length in [8,20,37]:
        probe=torch.randn(2,4,length,requires_grad=True);probe2=probe.detach().clone().requires_grad_();u=DeterministicPool()(probe);v=F.adaptive_avg_pool1d(probe2,8);u.square().sum().backward();v.square().sum().backward();assert torch.allclose(u,v,atol=1e-6) and torch.allclose(probe.grad,probe2.grad,atol=1e-6)
    d=source();s={k:v[:8] for k,v in d['source'].items()};a=features(s,'direction_order');b=features(s,'direction_iat');t=features(s,'tam_cnn')
    assert a.shape==(8,2,5000) and (a[:,1]==0).all() and torch.equal(a[:,0],b[:,0])
    assert (b[:,1]>=0).all() and (b[:,1]<=1).all() and t.shape==(8,2,1800)
    toy={'timestamps':torch.tensor([[1.,-1.01,1.0,0.,2.]])};ft=features(toy,'direction_iat');assert ft[0,1,0]==0 and ft[0,1,2]==0 and ft[0,1,3]==0 and ft[0,1,4]==0
    m=ViewNet().cuda();m.train();l=F.cross_entropy(m(a.cuda()),s['labels'].cuda());l.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval();one=m(a[:1].cuda());many=m(a.cuda())[:1];assert torch.allclose(one,many,atol=2e-5,rtol=2e-5)
    save(f'preflight_{seed}.json',dict(passed=True,parameters=sum(p.numel() for p in m.parameters()),batch_independent_inference=True,source_rows=len(d['source']['rows']),valid_rows=len(d['valid']['rows']),query_loaded=False))

def train(seed):
    d=source();y=d['source']['labels'].cuda();vy=d['valid']['labels'].numpy();rows=[]
    for arm in C['arms']:
        start=time.monotonic();setup(seed);m=ViewNet().cuda();initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};initial_digest=__import__('hashlib').sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        x=features(d['source'],arm).cuda();v=features(d['valid'],arm).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);gen=torch.Generator().manual_seed(seed+20261019);hist=[];best=-1;clip=0;best_epoch=None
        for epoch in range(1,31):
            if time.monotonic()-start>C['budget']['max_cell_seconds']:raise TimeoutError(f'{seed} {arm} cell budget')
            lr=.00005+(.001-.00005)*.5*(1+math.cos(math.pi*(epoch-1)/29))
            for pg in opt.param_groups:pg['lr']=lr
            m.train();loss_sum=0.;ix=torch.randperm(len(x),generator=gen);perm_sha=__import__('hashlib').sha256(ix.numpy().tobytes()).hexdigest()
            for inds in ix.split(128):
                inds=inds.cuda();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x[inds]),y[inds]);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);clip+=int(norm>5);opt.step();loss_sum+=float(loss.detach())*len(inds)
            row=dict(epoch=epoch,lr=lr,train_loss=loss_sum/len(x),permutation_sha256=perm_sha,seconds=time.monotonic()-start)
            if epoch in C['training']['evaluation_epochs']:
                z=predict(m,v);met=metrics(vy,z.argmax(-1).numpy(),z.argmax(-1).numpy());row.update(valid_macro_f1=met['macro_f1'],valid_accuracy=met['accuracy'])
                if met['macro_f1']>best:
                    best=met['macro_f1'];best_epoch=epoch
                    torch.save(dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},seed=seed,arm=arm,epoch=epoch,valid_macro_f1=best,source_rows=d['source']['rows'],valid_rows=d['valid']['rows'],config_sha256=sha(RUN/'config.json'),initial_sha256=initial_digest),RUN/'checkpoints'/f'{seed}_{arm}.pt')
            hist.append(row);save(f'history_{seed}_{arm}.json',hist);print(json.dumps(dict(seed=seed,arm=arm,**row)),flush=True)
        ck=RUN/'checkpoints'/f'{seed}_{arm}.pt';state=torch.load(ck,weights_only=True,map_location='cpu');m.load_state_dict(state['model']);z=predict(m,v);assert abs(metrics(vy,z.argmax(-1).numpy(),z.argmax(-1).numpy())['macro_f1']-best)<1e-12
        trials=[]
        for temp in C['training']['temperature_grid']:trials.append(dict(temperature=temp,nll=float(F.cross_entropy(z/temp,torch.from_numpy(vy)))))
        temp=min(trials,key=lambda r:r['nll'])['temperature'];train_z=predict(m,x)
        np.savez_compressed(RUN/'artifacts'/f'sourcevalid_{seed}_{arm}.npz',logits=z.numpy(),labels=vy,rows=d['valid']['rows'].numpy())
        # Store train labels/rows and final selected-model predictions for capacity/generalization auditing.
        np.savez_compressed(RUN/'artifacts'/f'sourcetrain_{seed}_{arm}.npz',predictions=train_z.argmax(-1).numpy(),labels=d['source']['labels'].numpy(),rows=d['source']['rows'].numpy())
        row=dict(seed=seed,arm=arm,best_epoch=best_epoch,valid_macro_f1=best,valid_accuracy=float((z.argmax(-1).numpy()==vy).mean()),source_train_accuracy=float((train_z.argmax(-1).cpu()==y.cpu()).float().mean()),temperature=temp,temperature_trials=trials,checkpoint_sha256=sha(ck),initial_sha256=initial_digest,clip_count=clip,seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters()),query_seen=False)
        rows.append(row);save(f'train_summary_{seed}.json',rows);print(json.dumps(row),flush=True);del m,x,v,opt,train_z;torch.cuda.empty_cache()
    assert len(rows)==3 and len({r['initial_sha256'] for r in rows})==1

def evaluate(seed,audit=False):
    assert (RUN/'artifacts/all_source_training_complete.json').is_file()
    d=source();summary=json.loads((RUN/'artifacts'/f'train_summary_{seed}.json').read_text());checks=0
    for row in summary:
        arm=row['arm'];ck=RUN/'checkpoints'/f'{seed}_{arm}.pt';assert sha(ck)==row['checkpoint_sha256'];state=torch.load(ck,weights_only=True,map_location='cpu');assert state['config_sha256']==sha(RUN/'config.json') and torch.equal(state['source_rows'],d['source']['rows']) and torch.equal(state['valid_rows'],d['valid']['rows'])
        m=ViewNet().cuda().eval();m.load_state_dict(state['model']);before=sha(ck)
        vz=predict(m,features(d['valid'],arm)).numpy()
        with np.load(RUN/'artifacts'/f'sourcevalid_{seed}_{arm}.npz') as a:assert np.array_equal(vz,a['logits'])
        # Re-evaluate one source-training probe to audit correct reload, not only saved predictions.
        tz=predict(m,features({k:v[:256] for k,v in d['source'].items()},arm)).argmax(-1).numpy()
        with np.load(RUN/'artifacts'/f'sourcetrain_{seed}_{arm}.npz') as a:assert np.array_equal(tz,a['predictions'][:256])
        for name in C['slices']:
            dd=target(name)
            for role in ['reference','query']:
                dat=dd[role];z=predict(m,features(dat,arm)).numpy();p=RUN/'artifacts'/f'{role}_{seed}_{name}_{arm}.npz'
                if audit:
                    with np.load(p) as a:
                        assert np.array_equal(a['rows'],dat['rows'].numpy()) and np.array_equal(a['logits'],z)
                        if role=='query':assert np.array_equal(a['labels'],dat['labels'].numpy())
                else:
                    arrays=dict(logits=z,rows=dat['rows'].numpy())
                    if role=='query':arrays['labels']=dat['labels'].numpy()
                    np.savez_compressed(p,**arrays)
                checks+=1
            del dd
        assert sha(ck)==before
        print(json.dumps(dict(seed=seed,arm=arm,phase='audit' if audit else 'evaluate',states_checked=checks)),flush=True);del m;torch.cuda.empty_cache()
    save(f'{"reload_audit" if audit else "evaluation_complete"}_{seed}.json',dict(passed=True,models_reloaded=3,sourcevalid_reloaded=3,source_probe_reloaded=3,target_states=checks,query_fit=False))

if __name__=='__main__':
    seed=int(os.environ['VIEW_SEED']);mode=os.environ['VIEW_MODE']
    try:
        setup(seed);verify();assert os.environ['CUDA_VISIBLE_DEVICES']==str(C['budget']['seed_gpu_map'][str(seed)])
        if mode=='preflight':preflight(seed)
        elif mode=='train':train(seed)
        else:evaluate(seed,audit=mode=='audit')
    except BaseException as e:save(f'failure_{mode}_{seed}.json',dict(error=str(e),traceback=traceback.format_exc()));raise
