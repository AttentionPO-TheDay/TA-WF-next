"""GPU worker for controlled two-pass CE and SAM."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, json, time
import numpy as np
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask, learning_rate
from common import RUN, config, verify_freeze, deterministic, save, savet, sha, metric

SEEDS=[21729,23407,22026]

def predict(model,e,bs):
    model.eval();before={k:v.detach().clone() for k,v in model.named_buffers()};out=[]
    with torch.inference_mode():
        for i in range(0,len(e['labels']),bs):out.append(model(e['timestamps'][i:i+bs],e['tam'][i:i+bs]).cpu().numpy())
    after=dict(model.named_buffers());assert all(torch.equal(before[k],after[k]) for k in before);return np.concatenate(out)

def set_bn_mode(model, training):
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.training=training

def grad_norm(model):
    vals=[p.grad.detach().float().norm(2) for p in model.parameters() if p.requires_grad and p.grad is not None]
    return torch.stack(vals).norm(2) if vals else torch.zeros((),device=next(model.parameters()).device)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True,choices=['two_pass_control','sam_ce']);ap.add_argument('--seed',required=True,type=int,choices=SEEDS);ap.add_argument('--device',required=True,choices=['cuda']);a=ap.parse_args()
    c=config();verify_freeze();assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0';deterministic('cuda',2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device('cuda')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');data={r:{k:v.to(device) for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()};labels={r:e['labels'].cpu().numpy() for r,e in data.items()}
    out=RUN/'artifacts'/f'{a.kind}_{a.seed}';out.mkdir(exist_ok=False);model=MultiViewModel('ce_only').to(device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};savet(out/'initial_state.pt',initial);ref=torch.load(RUN/'artifacts'/f'ce_only_{a.seed}'/'initial_state.pt',weights_only=True);assert all(torch.equal(initial[k],ref[k]) for k in initial)
    n=len(data['source']['labels']);bs=c['batch_size'];steps=c['optimizer_steps'];pieces=[];count=0;cycle=0
    while count<steps*bs:pieces.append(torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000+cycle)));count+=n;cycle+=1
    indices=torch.cat(pieces)[:steps*bs].reshape(steps,bs);savet(out/'index_stream.pt',indices);assert torch.equal(indices,torch.load(RUN/'artifacts'/f'ce_only_{a.seed}'/'index_stream.pt',weights_only=True));rng=torch.Generator().manual_seed(a.seed+50000)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8);rho=c['sam']['rho'];start=time.monotonic();history=[];best=-1.;beststep=0;sums={'loss':0.,'first_loss':0.,'second_loss':0.,'grad_norm':0.,'perturb_norm':0.};prev=0;buffer_restore_checks=0;restore_checks=0
    print(json.dumps({'task':f'{a.kind}_{a.seed}','device':'cuda','steps':steps}),flush=True)
    for step in range(1,steps+1):
        if time.monotonic()-start>c['gpu_job_seconds']:raise TimeoutError(a.kind)
        for g in opt.param_groups:g['lr']=learning_rate(step,steps)
        idx=indices[step-1].cuda();e=data['source'];tam=span_mask(e['tam'][idx],rng);y=e['labels'][idx]
        model.train();set_bn_mode(model,True);opt.zero_grad(set_to_none=True);first=model(e['timestamps'][idx],tam);first_loss=F.cross_entropy(first,y);first_loss.backward();
        gnorm=grad_norm(model);assert torch.isfinite(gnorm) and gnorm>0
        first_buffers={k:v.detach().clone() for k,v in model.named_buffers()};orig=[(p,p.detach().clone()) for p in model.parameters() if p.requires_grad];eps=[]
        if a.kind=='sam_ce':
            with torch.no_grad():
                for p,old in orig:
                    d=rho*p.grad.detach()/gnorm;p.add_(d);eps.append(d.detach().clone())
        set_bn_mode(model,False);opt.zero_grad(set_to_none=True);second=model(e['timestamps'][idx],tam);second_loss=F.cross_entropy(second,y);second_loss.backward();assert torch.isfinite(second_loss)
        after_second={k:v.detach().clone() for k,v in model.named_buffers()};assert all(torch.equal(first_buffers[k],after_second[k]) for k in first_buffers);buffer_restore_checks+=1
        if a.kind=='sam_ce':
            with torch.no_grad():
                for p,old in orig:p.copy_(old)
            assert all(torch.equal(p,old) for p,old in orig);restore_checks+=1
        opt.step();sums['loss']+=float(second_loss.detach());sums['first_loss']+=float(first_loss.detach());sums['second_loss']+=float(second_loss.detach());sums['grad_norm']+=float(gnorm.detach());sums['perturb_norm']+=float(torch.stack([d.norm() for d in eps]).norm().detach() if eps else 0.)
        if step%100==0:save(out/'progress.json',{'task':f'{a.kind}_{a.seed}','step':step,'total_steps':steps,'best_step':beststep,'elapsed_seconds':time.monotonic()-start})
        if step in c['eval_steps']:
            pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()};scores={r:metric(labels[r],z.argmax(1)) for r,z in pred.items()};row={'step':step,'lr':learning_rate(step,steps),'train_loss':sums['loss']/(step-prev),'first_loss':sums['first_loss']/(step-prev),'second_loss':sums['second_loss']/(step-prev),'grad_norm':sums['grad_norm']/(step-prev),'perturb_norm':sums['perturb_norm']/(step-prev),'elapsed_seconds':time.monotonic()-start,**scores};history.append(row);sums={'loss':0.,'first_loss':0.,'second_loss':0.,'grad_norm':0.,'perturb_norm':0.};prev=step
            if scores['valid']['accuracy']>best:best=scores['valid']['accuracy'];beststep=step;savet(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',{'state_dict':model.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            save(out/'history.json',history);print(json.dumps({'task':f'{a.kind}_{a.seed}',**row}),flush=True)
    last={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()};np.savez_compressed(out/'predictions_last.npz',**{r:z.argmax(1) for r,z in last.items()});ck=torch.load(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',weights_only=True,map_location=device);model.load_state_dict(ck['state_dict']);pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()};np.savez_compressed(out/'predictions_best.npz',**{r:z.argmax(1) for r,z in pred.items()});np.savez_compressed(out/'logits_best.npz',**pred)
    model.eval();ve=data['valid'];fixed=span_mask(ve['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed']));clean_pred=pred['valid'];masked=predict(model,{**ve,'tam':fixed},c['eval_batch_size']);diag={'clean_accuracy':metric(labels['valid'],clean_pred.argmax(1))['accuracy'],'masked_accuracy':metric(labels['valid'],masked.argmax(1))['accuracy'],'prediction_agreement':float((clean_pred.argmax(1)==masked.argmax(1)).mean())};save(out/'fixed_view_diagnostic.json',diag)
    reload=MultiViewModel('ce_only').cuda();reload.load_state_dict(ck['state_dict']);
    for r,e in data.items():assert np.array_equal(predict(reload,e,c['eval_batch_size']).argmax(1),pred[r].argmax(1))
    report={'task':f'{a.kind}_{a.seed}','kind':a.kind,'seed':a.seed,'device':'cuda','best_step':beststep,'steps_completed':steps,'validation_opportunities':len(history),'parameters_total':sum(p.numel() for p in reload.parameters()),'parameters_trainable':sum(p.numel() for p in reload.parameters() if p.requires_grad),'sam_used':a.kind=='sam_ce','rho':rho,'bn_buffer_restore_checks':buffer_restore_checks,'parameter_restore_checks':restore_checks,'verified_roles':['source','valid'],'future_access':False,'best':{r:metric(labels[r],z.argmax(1)) for r,z in pred.items()},'last':{r:metric(labels[r],z.argmax(1)) for r,z in last.items()},'fixed_view_diagnostic':diag,'elapsed_seconds':time.monotonic()-start};save(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
