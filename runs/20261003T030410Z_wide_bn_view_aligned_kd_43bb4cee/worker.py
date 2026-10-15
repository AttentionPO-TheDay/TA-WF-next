"""GPU worker for view-aligned probability distillation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, json, time
import numpy as np
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask, learning_rate, make_optimizer
from common import RUN, config, verify_freeze, deterministic, save, savet, sha, metric

SEEDS = [21729, 23407, 22026]

def predict(model, e, bs):
    model.eval(); before={k:v.detach().clone() for k,v in model.named_buffers()}; out=[]
    with torch.inference_mode():
        for i in range(0,len(e['labels']),bs): out.append(model(e['timestamps'][i:i+bs],e['tam'][i:i+bs]).cpu().numpy())
    after=dict(model.named_buffers()); assert all(torch.equal(before[k],after[k]) for k in before)
    return np.concatenate(out)

def load_teachers(device):
    teachers=[]
    for seed in SEEDS:
        m=MultiViewModel('ce_only').to(device)
        ck=torch.load(RUN/'artifacts'/f'teacher_checkpoint_{seed}.pt',weights_only=True,map_location=device)
        m.load_state_dict(ck['state_dict']); m.eval(); m.requires_grad_(False); teachers.append(m)
    return teachers

def teacher_probs(teachers, timestamps, tam, T):
    vals=[]
    with torch.inference_mode():
        for m in teachers: vals.append(F.softmax(m(timestamps,tam)/T,dim=1))
    probs=torch.stack(vals,0).mean(0)
    confidence=probs.max(1).values
    votes=torch.stack([v.argmax(1) for v in vals],0)
    disagreement=(votes != votes[0:1]).any(0)
    return probs, confidence, disagreement

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--kind',required=True,choices=['ce_only','matched_kd','clean_kd','gated_matched_kd']); ap.add_argument('--seed',required=True,type=int,choices=SEEDS); ap.add_argument('--device',required=True,choices=['cuda']); a=ap.parse_args()
    c=config(); verify_freeze(); assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    deterministic('cuda',2); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); device=torch.device('cuda')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
    data={role:{k:v.to(device) for k,v in e.items() if k in ('timestamps','tam','labels')} for role,e in raw.items()}; labels={r:e['labels'].cpu().numpy() for r,e in data.items()}; del raw
    teachers=None if a.kind=='ce_only' else load_teachers(device)
    T=c['distillation']['temperature']
    clean_target=None
    if a.kind=='clean_kd':
        paths=[]
        for s in SEEDS:
            with np.load(RUN/'artifacts'/f'teacher_logits_{s}.npz') as z: paths.append(z['source'].astype(np.float32,copy=False))
        # The frozen target is the arithmetic mean of teacher probabilities,
        # with temperature applied per teacher before averaging.  Averaging
        # logits first would define a different distillation target.
        probs=np.stack([np.exp(x/T - np.max(x/T, axis=1, keepdims=True)) /
                        np.exp(x/T - np.max(x/T, axis=1, keepdims=True)).sum(axis=1, keepdims=True)
                        for x in paths], axis=0).mean(axis=0)
        clean_target=torch.from_numpy(probs.astype(np.float32, copy=False)).to(device)
    # Teacher construction is intentionally before this point for frozen
    # loading, but it consumes the global RNG while creating model modules.
    # Reset so every condition uses the exact same student initialization.
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out=RUN/'artifacts'/f'{a.kind}_{a.seed}'; out.mkdir(exist_ok=False)
    model=MultiViewModel('ce_only').to(device); initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; savet(out/'initial_state.pt',initial)
    ref=torch.load(RUN/'artifacts'/f'teacher_initial_{a.seed}.pt',weights_only=True,map_location='cpu'); assert set(ref)==set(initial) and all(torch.equal(ref[k],initial[k]) for k in initial)
    opt=make_optimizer(model); n=len(data['source']['labels']); bs=c['batch_size']; steps=c['optimizer_steps']; pieces=[]; count=0; cycle=0
    while count<steps*bs: pieces.append(torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000+cycle))); count+=n; cycle+=1
    indices=torch.cat(pieces)[:steps*bs].reshape(steps,bs); savet(out/'index_stream.pt',indices); aug_rng=torch.Generator().manual_seed(a.seed+50000)
    lam=c['distillation']['weight']; start=time.monotonic(); history=[]; best=-1.; beststep=0; sums={'loss':0.,'ce':0.,'kd':0.}; prev=0; gate_total=0; gate_steps=0
    print(json.dumps({'task':f'{a.kind}_{a.seed}','device':'cuda','steps':steps}),flush=True)
    for step in range(1,steps+1):
        if time.monotonic()-start>c['gpu_job_seconds']: raise TimeoutError(f'{a.kind}_{a.seed}')
        for g in opt.param_groups: g['lr']=learning_rate(step,steps)
        model.train(); idx=indices[step-1].cuda(); e=data['source']; clean=e['tam'][idx]; tam=clean if a.kind=='clean_kd' else span_mask(clean,aug_rng); y=e['labels'][idx]
        opt.zero_grad(set_to_none=True); logits=model(e['timestamps'][idx],tam); ce=F.cross_entropy(logits,y); kd=ce.new_zeros(()); gate_count=0
        if a.kind=='clean_kd': probs=clean_target[idx]
        elif a.kind in ('matched_kd','gated_matched_kd'): probs,conf,disagreement=teacher_probs(teachers,e['timestamps'][idx],tam,T)
        if a.kind!='ce_only':
            if a.kind=='gated_matched_kd':
                gate=(disagreement & (conf>=c['distillation']['confidence_threshold'])); gate_count=int(gate.sum()); gate_total+=gate_count; gate_steps+=len(gate)
                if gate_count: kd=F.kl_div(F.log_softmax(logits[gate]/T,dim=1),probs[gate],reduction='batchmean')*(T*T)
            else: kd=F.kl_div(F.log_softmax(logits/T,dim=1),probs,reduction='batchmean')*(T*T)
        loss=ce+lam*kd; assert torch.isfinite(loss) and torch.isfinite(ce) and torch.isfinite(kd); loss.backward()
        if step==1:
            grads={k:(float(p.grad.norm()) if p.grad is not None else None) for k,p in model.named_parameters()}; save(out/'first_gradients.json',grads); active=[v for k,v in grads.items() if (k.startswith('generator.stages') or k=='generator.to_features.weight') and k.endswith('weight')]; assert active and all(v is not None and np.isfinite(v) and v>0 for v in active)
        opt.step(); sums['loss']+=float(loss.detach()); sums['ce']+=float(ce.detach()); sums['kd']+=float(kd.detach())
        if step%100==0: save(out/'progress.json',{'task':f'{a.kind}_{a.seed}','step':step,'total_steps':steps,'best_step':beststep,'elapsed_seconds':time.monotonic()-start})
        if step in c['eval_steps']:
            pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()}; scores={r:metric(labels[r],z.argmax(1)) for r,z in pred.items()}; row={'step':step,'lr':learning_rate(step,steps),'train_loss':sums['loss']/(step-prev),'train_ce':sums['ce']/(step-prev),'train_kd':sums['kd']/(step-prev),'elapsed_seconds':time.monotonic()-start,**scores}; sums={'loss':0.,'ce':0.,'kd':0.}; prev=step; history.append(row)
            if scores['valid']['accuracy']>best: best=scores['valid']['accuracy']; beststep=step; savet(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',{'state_dict':model.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            save(out/'history.json',history); print(json.dumps({'task':f'{a.kind}_{a.seed}',**row}),flush=True)
    last={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()}; np.savez_compressed(out/'predictions_last.npz',**{r:z.argmax(1) for r,z in last.items()}); ck=torch.load(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',weights_only=True,map_location=device); model.load_state_dict(ck['state_dict']); pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()}; np.savez_compressed(out/'predictions_best.npz',**{r:z.argmax(1) for r,z in pred.items()}); np.savez_compressed(out/'logits_best.npz',**pred)
    reload=MultiViewModel('ce_only').cuda(); reload.load_state_dict(ck['state_dict']);
    for r,e in data.items(): assert np.array_equal(predict(reload,e,c['eval_batch_size']).argmax(1),pred[r].argmax(1))
    report={'task':f'{a.kind}_{a.seed}','kind':a.kind,'seed':a.seed,'device':'cuda','best_step':beststep,'steps_completed':steps,'validation_opportunities':len(history),'parameters_total':sum(p.numel() for p in reload.parameters()),'parameters_trainable':sum(p.numel() for p in reload.parameters() if p.requires_grad),'teacher_target':'none' if a.kind=='ce_only' else ('clean_probability_mean' if a.kind=='clean_kd' else 'matched_probability_mean'),'temperature':T,'distillation_weight':0.0 if a.kind=='ce_only' else lam,'gate_coverage':None if a.kind!='gated_matched_kd' else gate_total/max(gate_steps,1),'verified_roles':['source','valid'],'future_access':False,'best':{r:metric(labels[r],z.argmax(1)) for r,z in pred.items()},'last':{r:metric(labels[r],z.argmax(1)) for r,z in last.items()},'elapsed_seconds':time.monotonic()-start}
    save(out/'report.json',report); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
