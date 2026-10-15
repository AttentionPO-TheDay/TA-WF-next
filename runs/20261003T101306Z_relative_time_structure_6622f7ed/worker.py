import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, time, json
import numpy as np
import torch
import torch.nn.functional as F
from model import MultiViewModel, span_mask, learning_rate
from common import RUN, config, verify_freeze, deterministic, save, savet, sha, metric
SEEDS=[21729,23407,22026]

def predict(model,e,bs):
    model.eval(); before={k:v.detach().clone() for k,v in model.named_buffers()}; out=[]
    with torch.inference_mode():
        for i in range(0,len(e['labels']),bs): out.append(model(e['timestamps'][i:i+bs],e['tam'][i:i+bs]).cpu().numpy())
    after=dict(model.named_buffers()); assert all(torch.equal(before[k],after[k]) for k in before)
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--kind',required=True,choices=['ce_control','relative_time']); ap.add_argument('--seed',required=True,type=int,choices=SEEDS); ap.add_argument('--device',required=True,choices=['cuda']); a=ap.parse_args()
    c=config(); verify_freeze(); assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'; deterministic('cuda',2); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); device=torch.device('cuda')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu'); data={r:{k:v.to(device) for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()}; labels={r:e['labels'].cpu().numpy() for r,e in data.items()}
    out=RUN/'artifacts'/f'{a.kind}_{a.seed}'; out.mkdir(exist_ok=False); model=MultiViewModel(a.kind).to(device); initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; savet(out/'initial_state.pt',initial)
    n=len(data['source']['labels']); bs=c['batch_size']; steps=c['optimizer_steps']; pieces=[]; count=0; cycle=0
    while count<steps*bs: pieces.append(torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000+cycle))); count+=n; cycle+=1
    indices=torch.cat(pieces)[:steps*bs].reshape(steps,bs); savet(out/'index_stream.pt',indices); rng=torch.Generator().manual_seed(a.seed+50000)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8); start=time.monotonic(); history=[]; best=-1.; beststep=0; sums={'loss':0.}; prev=0
    print(json.dumps({'task':f'{a.kind}_{a.seed}','device':'cuda','steps':steps}),flush=True)
    for step in range(1,steps+1):
        if time.monotonic()-start>c['gpu_job_seconds']: raise TimeoutError(a.kind)
        for g in opt.param_groups:g['lr']=learning_rate(step,steps)
        idx=indices[step-1].cuda(); e=data['source']; tam=span_mask(e['tam'][idx],rng); y=e['labels'][idx]; model.train(); opt.zero_grad(set_to_none=True); logits=model(e['timestamps'][idx],tam); loss=F.cross_entropy(logits,y); loss.backward(); opt.step(); sums['loss']+=float(loss.detach())
        if step%100==0: save(out/'progress.json',{'task':f'{a.kind}_{a.seed}','step':step,'total_steps':steps,'best_step':beststep,'elapsed_seconds':time.monotonic()-start})
        if step in c['eval_steps']:
            pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()}; scores={r:metric(labels[r],z.argmax(1)) for r,z in pred.items()}; row={'step':step,'lr':learning_rate(step,steps),'train_loss':sums['loss']/(step-prev),'elapsed_seconds':time.monotonic()-start,**scores}; history.append(row); sums={'loss':0.}; prev=step
            if scores['valid']['accuracy']>best: best=scores['valid']['accuracy']; beststep=step; savet(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',{'state_dict':model.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            save(out/'history.json',history); print(json.dumps({'task':f'{a.kind}_{a.seed}',**row}),flush=True)
    ck=torch.load(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',weights_only=True,map_location=device); model.load_state_dict(ck['state_dict']); pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()}; np.savez_compressed(out/'logits_best.npz',**pred)
    ve=data['valid']; fixed=span_mask(ve['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed'])); masked=predict(model,{**ve,'tam':fixed},c['eval_batch_size']); diag={'clean_accuracy':metric(labels['valid'],pred['valid'].argmax(1))['accuracy'],'masked_accuracy':metric(labels['valid'],masked.argmax(1))['accuracy'],'prediction_agreement':float((pred['valid'].argmax(1)==masked.argmax(1)).mean()),'relative_feature_mean':float(model.relative_time_features(ve['timestamps']).mean().detach()),'relative_feature_std':float(model.relative_time_features(ve['timestamps']).std().detach())}
    reload=MultiViewModel(a.kind).cuda(); reload.load_state_dict(ck['state_dict']);
    for r,e in data.items(): assert np.array_equal(predict(reload,e,c['eval_batch_size']).argmax(1),pred[r].argmax(1))
    report={'task':f'{a.kind}_{a.seed}','kind':a.kind,'seed':a.seed,'device':'cuda','best_step':beststep,'steps_completed':steps,'validation_opportunities':len(history),'parameters_total':sum(p.numel() for p in reload.parameters()),'parameters_trainable':sum(p.numel() for p in reload.parameters() if p.requires_grad),'verified_roles':['source','valid'],'future_access':False,'best':{r:metric(labels[r],z.argmax(1)) for r,z in pred.items()},'last':None,'fixed_view_diagnostic':diag,'elapsed_seconds':time.monotonic()-start}; save(out/'report.json',report); print(json.dumps(report),flush=True)
if __name__=='__main__': main()
