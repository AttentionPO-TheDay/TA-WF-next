"""GPU worker for peer-loss attribution with a single network."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, json, time
import numpy as np
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask, symmetric_peer_loss, learning_rate
from common import RUN, config, verify_freeze, deterministic, save, savet, sha, metric

SEEDS=[21729,23407,22026]

def predict(model,e,bs):
    model.eval(); before={k:v.detach().clone() for k,v in model.named_buffers()}; out=[]
    with torch.inference_mode():
        for i in range(0,len(e['labels']),bs): out.append(model(e['timestamps'][i:i+bs],e['tam'][i:i+bs]).cpu().numpy())
    after=dict(model.named_buffers()); assert all(torch.equal(before[k],after[k]) for k in before); return np.concatenate(out)

def js_divergence(la,lb):
    lpa,lpb=F.log_softmax(la,1),F.log_softmax(lb,1)
    lpm=torch.logaddexp(lpa,lpb)-np.log(2.0)
    return float((.5*((lpa.exp()*(lpa-lpm)).sum(1)+(lpb.exp()*(lpb-lpm)).sum(1))).mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--kind',required=True,choices=['primary_ce_only_joint','primary_ce_peer']); ap.add_argument('--seed',required=True,type=int,choices=SEEDS); ap.add_argument('--device',required=True,choices=['cuda']); a=ap.parse_args()
    c=config(); verify_freeze(); assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'; deterministic('cuda',2); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); device=torch.device('cuda')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu'); data={r:{k:v.to(device) for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()}; labels={r:e['labels'].cpu().numpy() for r,e in data.items()}
    out=RUN/'artifacts'/f'{a.kind}_{a.seed}'; out.mkdir(exist_ok=False); model=MultiViewModel('ce_only').to(device); initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; savet(out/'initial_state.pt',initial); ref=torch.load(RUN/'artifacts'/f'ce_only_{a.seed}'/'initial_state.pt',weights_only=True); assert all(torch.equal(initial[k],ref[k]) for k in initial)
    n=len(data['source']['labels']); bs=c['batch_size']; steps=c['optimizer_steps']; pieces=[];count=0;cycle=0
    while count<steps*bs: pieces.append(torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000+cycle)));count+=n;cycle+=1
    indices=torch.cat(pieces)[:steps*bs].reshape(steps,bs); savet(out/'index_stream.pt',indices); assert torch.equal(indices,torch.load(RUN/'artifacts'/f'ce_only_{a.seed}'/'index_stream.pt',weights_only=True))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8); rng_a=torch.Generator().manual_seed(a.seed+50000); rng_b=torch.Generator().manual_seed(a.seed+60000); T=c['peer_consistency']['temperature']; ramp=c['peer_consistency']['ramp_steps']; maxw=c['peer_consistency']['max_weight']; start=time.monotonic(); history=[]; best=-1.;beststep=0;sums={'loss':0.,'ce':0.,'peer':0.,'weight':0.};prev=0
    print(json.dumps({'task':f'{a.kind}_{a.seed}','device':'cuda','steps':steps}),flush=True)
    for step in range(1,steps+1):
        if time.monotonic()-start>c['gpu_job_seconds']: raise TimeoutError(a.kind)
        for g in opt.param_groups:g['lr']=learning_rate(step,steps)
        model.train();idx=indices[step-1].cuda();e=data['source'];clean=e['tam'][idx];ta=span_mask(clean,rng_a);tb=span_mask(clean,rng_b);x=torch.cat([e['timestamps'][idx],e['timestamps'][idx]]);v=torch.cat([ta,tb]);y=e['labels'][idx];opt.zero_grad(set_to_none=True);both=model(x,v);la,lb=both[:bs],both[bs:];ce=F.cross_entropy(la,y);peer=ce.new_zeros(());w=0.
        if a.kind=='primary_ce_peer': peer=symmetric_peer_loss(la,lb,T);w=maxw*min(max(step-ramp,0)/ramp,1.0)
        loss=ce+w*peer; assert torch.isfinite(loss) and torch.isfinite(ce) and torch.isfinite(peer);loss.backward()
        if step==1:
            grads={k:(float(p.grad.norm()) if p.grad is not None else None) for k,p in model.named_parameters()};save(out/'first_gradients.json',grads);active=[v for k,v in grads.items() if (k.startswith('generator.stages') or k=='generator.to_features.weight') and k.endswith('weight')];assert active and all(v is not None and np.isfinite(v) and v>0 for v in active)
        opt.step();sums['loss']+=float(loss.detach());sums['ce']+=float(ce.detach());sums['peer']+=float(peer.detach());sums['weight']+=w
        if step%100==0:save(out/'progress.json',{'task':f'{a.kind}_{a.seed}','step':step,'total_steps':steps,'best_step':beststep,'elapsed_seconds':time.monotonic()-start})
        if step in c['eval_steps']:
            pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()};scores={r:metric(labels[r],z.argmax(1)) for r,z in pred.items()};row={'step':step,'lr':learning_rate(step,steps),'train_loss':sums['loss']/(step-prev),'train_ce':sums['ce']/(step-prev),'train_peer':sums['peer']/(step-prev),'mean_peer_weight':sums['weight']/(step-prev),'elapsed_seconds':time.monotonic()-start,**scores};history.append(row);sums={'loss':0.,'ce':0.,'peer':0.,'weight':0.};prev=step
            if scores['valid']['accuracy']>best:best=scores['valid']['accuracy'];beststep=step;savet(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',{'state_dict':model.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            save(out/'history.json',history);print(json.dumps({'task':f'{a.kind}_{a.seed}',**row}),flush=True)
    last={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()};np.savez_compressed(out/'predictions_last.npz',**{r:z.argmax(1) for r,z in last.items()});ck=torch.load(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',weights_only=True,map_location=device);model.load_state_dict(ck['state_dict']);pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()};np.savez_compressed(out/'predictions_best.npz',**{r:z.argmax(1) for r,z in pred.items()});np.savez_compressed(out/'logits_best.npz',**pred)
    model.eval();ve=data['valid'];fixed=span_mask(ve['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed']))
    cl=torch.from_numpy(pred['valid']);ml=torch.from_numpy(predict(model,{**ve,'tam':fixed},c['eval_batch_size']))
    diag={'clean_accuracy':metric(labels['valid'],cl.argmax(1).cpu().numpy())['accuracy'],'masked_accuracy':metric(labels['valid'],ml.argmax(1).cpu().numpy())['accuracy'],'prediction_agreement':float((cl.argmax(1)==ml.argmax(1)).float().mean()),'js_divergence':js_divergence(cl,ml)};save(out/'fixed_view_diagnostic.json',diag)
    reload=MultiViewModel('ce_only').cuda();reload.load_state_dict(ck['state_dict']);
    for r,e in data.items():assert np.array_equal(predict(reload,e,c['eval_batch_size']).argmax(1),pred[r].argmax(1))
    report={'task':f'{a.kind}_{a.seed}','kind':a.kind,'seed':a.seed,'device':'cuda','best_step':beststep,'steps_completed':steps,'validation_opportunities':len(history),'parameters_total':sum(p.numel() for p in reload.parameters()),'parameters_trainable':sum(p.numel() for p in reload.parameters() if p.requires_grad),'single_network':True,'peer_consistency':a.kind=='primary_ce_peer','primary_view_ce_only':True,'temperature':T,'max_peer_weight':maxw if a.kind=='primary_ce_peer' else 0.,'ramp_steps':ramp if a.kind=='primary_ce_peer' else 0.,'verified_roles':['source','valid'],'future_access':False,'best':{r:metric(labels[r],z.argmax(1)) for r,z in pred.items()},'last':{r:metric(labels[r],z.argmax(1)) for r,z in last.items()},'fixed_view_diagnostic':diag,'elapsed_seconds':time.monotonic()-start};save(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
