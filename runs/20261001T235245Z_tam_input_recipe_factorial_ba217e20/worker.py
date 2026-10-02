"""Bounded CPU/GPU worker; valid labels only select predeclared checkpoints."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,f1_score
from model import MultiViewModel,span_mask,learning_rate,make_optimizer,FACTORS
from common import RUN,ROOT,config,verify_freeze,deterministic,save,savet,sha,metric

def predict(model,e,batch_size):
    model.eval();out=[]
    with torch.inference_mode():
        for i in range(0,len(e['labels']),batch_size):
            out.append(model(e['timestamps'][i:i+batch_size],e['tam'][i:i+batch_size]).cpu().numpy())
    return np.concatenate(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    c=config();assert a.seed in c['seeds'] and a.kind in c['conditions'];verify_freeze()
    assert (a.device=='cpu')==(a.kind in c['cpu_conditions'])
    if a.device=='cuda':assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    else:assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    deterministic(a.device,c['cpu_threads'] if a.device=='cpu' else c['gpu_threads'])
    torch.manual_seed(a.seed)
    if a.device=='cuda':torch.cuda.manual_seed_all(a.seed)
    raw=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
    data={role:{k:v.to(a.device) for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()};del raw
    key=f'{a.kind}_{a.seed}';out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
    m=MultiViewModel(a.kind).to(a.device)
    initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};savet(out/'initial_state.pt',initial)
    opt=make_optimizer(m,a.kind)
    n=len(data['source']['labels']);bs=c['batch_size'];steps=c['optimizer_steps'];pieces=[];count=0;cycle=0
    while count<steps*bs:
        pieces.append(torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000+cycle)));count+=n;cycle+=1
    indices=torch.cat(pieces)[:steps*bs].reshape(steps,bs);savet(out/'index_stream.pt',indices)
    aug_rng=torch.Generator().manual_seed(a.seed+50000)
    labels={role:e['labels'].cpu().numpy() for role,e in data.items()}
    budget=c['cpu_job_seconds'] if a.device=='cpu' else c['gpu_job_seconds']
    start=time.monotonic();history=[];best=-1.;beststep=0;loss_sum=ce_sum=aux_sum=0.;prev=0
    print(json.dumps({'task':key,'device':a.device,'parameters':sum(p.numel() for p in m.parameters()),'steps':steps,'budget_seconds':budget}),flush=True)
    for step in range(1,steps+1):
        if time.monotonic()-start>budget:raise TimeoutError(key+' job budget exceeded')
        lr=learning_rate(a.kind,step,steps)
        for group in opt.param_groups:group['lr']=lr
        m.train();idx=indices[step-1].to(a.device);e=data['source'];tam=e['tam'][idx]
        tam=span_mask(tam,aug_rng)
        opt.zero_grad(set_to_none=True)
        logits,features=m(e['timestamps'][idx],tam,return_features=True)
        y=e['labels'][idx];ce=F.cross_entropy(logits,y);aux=ce.new_zeros(())
        loss=ce+.05*aux;assert torch.isfinite(loss);loss.backward()
        if step==1:
            grads={k:float(p.grad.norm()) if p.grad is not None else None for k,p in m.named_parameters()}
            save(out/'first_gradients.json',grads)
            for k,p in m.generator.named_parameters():
                if p.requires_grad:assert grads['generator.'+k] is not None and np.isfinite(grads['generator.'+k]) and grads['generator.'+k]>0,k
                else:assert grads['generator.'+k] is None,k
        opt.step();loss_sum+=float(loss.detach());ce_sum+=float(ce.detach());aux_sum+=float(aux.detach())
        if step%100==0:
            save(out/'progress.json',{'task':key,'device':a.device,'step':step,'total_steps':steps,'best_step':beststep,'elapsed_seconds':time.monotonic()-start})
        if step in c['eval_steps']:
            pred={role:predict(m,e,c['eval_batch_size']) for role,e in data.items()}
            scores={role:metric(labels[role],z.argmax(1)) for role,z in pred.items()}
            row={'step':step,'lr':lr,'train_loss':loss_sum/(step-prev),'train_ce':ce_sum/(step-prev),'train_aux':aux_sum/(step-prev),'elapsed_seconds':time.monotonic()-start,**scores}
            loss_sum=ce_sum=aux_sum=0.;prev=step
            if scores['valid']['accuracy']>best:
                best=scores['valid']['accuracy'];beststep=step
                savet(RUN/'checkpoints'/f'{key}_best.pt',{'state_dict':m.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            history.append(row);save(out/'history.json',history)
            latest={'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'cpu_rng':torch.get_rng_state(),'augmentation_rng':aug_rng.get_state(),'history':history,'config_sha256':sha(RUN/'config.json')}
            if a.device=='cuda':latest['cuda_rng']=torch.cuda.get_rng_state()
            savet(RUN/'checkpoints'/f'{key}_latest.pt',latest);print(json.dumps({'task':key,**row}),flush=True)
    last={role:predict(m,e,c['eval_batch_size']) for role,e in data.items()}
    np.savez_compressed(out/'predictions_last.npz',**{role:z.argmax(1) for role,z in last.items()})
    ck=torch.load(RUN/'checkpoints'/f'{key}_best.pt',weights_only=True,map_location=a.device)
    m.load_state_dict(ck['state_dict'])
    pred={role:predict(m,e,c['eval_batch_size']) for role,e in data.items()}
    np.savez_compressed(out/'predictions_best.npz',**{role:z.argmax(1) for role,z in pred.items()});np.savez_compressed(out/'logits_best.npz',**pred)
    del m,opt
    if a.device=='cuda':torch.cuda.empty_cache()
    reload=MultiViewModel(a.kind).to(a.device);reload.load_state_dict(ck['state_dict']);verified=[]
    for role,e in data.items():
        pr=predict(reload,e,c['eval_batch_size']).argmax(1);assert np.array_equal(pr,pred[role].argmax(1))
        sc=metric(labels[role],pr)
        assert abs(sc['accuracy']-accuracy_score(labels[role],pr))<1e-12
        assert abs(sc['macro_f1']-f1_score(labels[role],pr,labels=np.arange(102),average='macro',zero_division=0))<1e-12
        verified.append(role)
    deltas={k:float((p.detach().cpu()-initial[k]).square().sum()) for k,p in reload.named_parameters()}
    frozen=[]
    for k,p in reload.named_parameters():
        if not p.requires_grad:assert deltas[k]==0,k;frozen.append(k)
        elif k.startswith('generator.') or k=='readout_query':assert deltas[k]>0,k
    report={'task':key,'kind':a.kind,'seed':a.seed,'device':a.device,'best_step':beststep,'steps_completed':steps,'validation_opportunities':len(history),'parameters_total':sum(p.numel() for p in reload.parameters()),'parameters_trainable':sum(p.numel() for p in reload.parameters() if p.requires_grad),'generator_parameter_delta_l2':sum(v for k,v in deltas.items() if k.startswith('generator.'))**.5,'parameter_deltas_squared':deltas,'inactive_parameters_unchanged':True,'elapsed_seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,'verified_roles':verified,'input_scale':FACTORS[a.kind][0],'recipe':FACTORS[a.kind][1],'best':{role:metric(labels[role],z.argmax(1)) for role,z in pred.items()},'last':{role:metric(labels[role],z.argmax(1)) for role,z in last.items()}}
    save(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
