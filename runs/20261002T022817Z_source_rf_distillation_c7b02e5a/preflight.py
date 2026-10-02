import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import RUN,ROOT,sha,save,deterministic
from model import MultiViewModel,span_mask,distillation_loss
from rf_native import RFNative
from worker import predict
from base_model import parameter_counts

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    c=json.loads((RUN/'config.json').read_text());deterministic('cuda',2);old=ROOT/c['historical_run']
    for path,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/path)==h,path
    assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');assert set(raw)=={'source','valid'}
    data={role:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()}
    references={};checks=0
    rf=ROOT/c['distillation']['teacher_run']
    native=torch.load(rf/'artifacts/native_prepared.pt',weights_only=False,map_location='cpu')
    for role in ['source','valid']:
        assert torch.equal(raw[role]['rows'],native[role]['rows'])
        assert torch.equal(raw[role]['labels'],native[role]['labels'])
        assert torch.equal(raw[role]['tam'],native[role]['tam'][:,0])
    teacher_audit=[]
    for seed in c['seeds']:
        ckpath=rf/'checkpoints'/f'rf_{seed}_best.pt';lp=rf/'artifacts'/f'rf_{seed}'/'logits_best.npz'
        for q in [ckpath,lp,rf/'artifacts/native_prepared.pt']:
            references[str(q.relative_to(ROOT))]=sha(q)
        teacher=RFNative(102).cuda().eval();teacher.load_state_dict(torch.load(ckpath,weights_only=True,map_location='cuda')['state_dict'])
        for param in teacher.parameters():param.requires_grad_(False)
        with np.load(lp) as f: saved=f['source'].copy()
        assert saved.shape==(15300,102) and np.isfinite(saved).all()
        replay=[]
        with torch.inference_mode():
            for i in range(0,15300,64):replay.append(teacher(data['source']['tam'][i:i+64,None]).cpu().numpy())
        replay=np.concatenate(replay);error=float(np.max(np.abs(replay-saved)))
        assert error<1e-4 and np.array_equal(replay.argmax(1),saved.argmax(1)),error
        torch.save({'source_logits':torch.from_numpy(saved),'source_rows':raw['source']['rows'],'source_labels':raw['source']['labels']},RUN/'artifacts'/f'teacher_source_{seed}.pt')
        teacher_audit.append({'seed':seed,'rows':15300,'max_logit_error':error,'prediction_replay':True})
        del teacher
    # KL direction, temperature scaling, teacher gradient isolation, identical-target zero.
    z=torch.tensor([[1.,2.,-1.],[0.,-2.,3.]],requires_grad=True);t=torch.tensor([[2.,1.,0.],[-1.,0.,2.]],requires_grad=True);T=2.
    actual=distillation_loss(z,t,T);prob=(t.detach()/T).softmax(1)
    manual=(prob*(prob.log()-(z/T).log_softmax(1))).sum()/len(z)*T*T
    assert torch.allclose(actual,manual,atol=1e-6)
    actual.backward();assert t.grad is None and z.grad is not None and torch.isfinite(z.grad).all()
    assert abs(float(distillation_loss(z.detach(),z.detach(),T)))<1e-6

    for seed in c['seeds']:
        d=RUN/'artifacts'/f'ce_only_{seed}';rep=json.loads((d/'report.json').read_text())
        torch.manual_seed(seed);m=MultiViewModel('ce_only').cuda()
        state=torch.load(d/'initial_state.pt',weights_only=True,map_location='cpu');assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
        ckpath=ROOT/'runs'/rep['origin_run']/'checkpoints'/f"{rep['origin_task']}_best.pt"
        m.load_state_dict(torch.load(ckpath,weights_only=True,map_location='cuda')['state_dict'])
        with np.load(d/'predictions_best.npz') as f:
            for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),f[role]);checks+=1
        references[str(ckpath.relative_to(ROOT))]=sha(ckpath);del m
    models=[];losses=[];params={}
    for kind in c['conditions']:
        torch.manual_seed(c['seeds'][0]);m=MultiViewModel(kind).cuda()
        ref=torch.load(RUN/'artifacts'/f"ce_only_{c['seeds'][0]}"/'initial_state.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(m.state_dict()[k].cpu(),v) for k,v in ref.items())
        e=data['source'];tam=span_mask(e['tam'][:64],torch.Generator().manual_seed(c['seeds'][0]+50000))
        target=torch.load(RUN/'artifacts'/f"teacher_source_{c['seeds'][0]}.pt",weights_only=True)['source_logits'][:64].cuda()
        logits=m(e['timestamps'][:64],tam)
        loss=F.cross_entropy(logits,e['labels'][:64])+c['distillation']['weights'][kind]*distillation_loss(logits,target,2.);assert torch.isfinite(loss)
        models.append(m);losses.append(loss);params[kind]=parameter_counts(m)
    for m,loss in zip(models,losses):
        loss.backward()
        for name,p in m.generator.named_parameters():
            if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
            else:assert p.grad is None,name
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();total=torch.cuda.mem_get_info()[1]
    data_bytes=sum(v.numel()*v.element_size() for e in data.values() for v in e.values())
    estimate=peak+2*data_bytes+3*1024**3+3*256*1024**2+4*1024**3;assert estimate<total
    params['ce_only']={'total':350006,'trainable':322726,'generator_total':3024,'generator_trainable':1472}
    references[str((old/'artifacts/prepared.pt').relative_to(ROOT))]=sha(old/'artifacts/prepared.pt')
    save(RUN/'artifacts/preflight.json',{'passed':True,'teacher_replay':teacher_audit,'kl_semantics_passed':True,'baseline_reload_checks':checks,'candidate_count':3,'source_examples_per_candidate':64,'optimizer_updates':0,'parameter_counts':params,'aggregate_gpu_bytes_with_reserve':estimate,'gpu_total_bytes':total,'historical_reference_hashes':references,'future_access':False})
    print('passed; baseline reload',checks,'GPU estimate GiB',estimate/1024**3)

if __name__=='__main__':main()
