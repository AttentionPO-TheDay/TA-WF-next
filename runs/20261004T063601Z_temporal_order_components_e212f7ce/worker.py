import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,time
import numpy as np
import torch
import torch.nn.functional as F
from model import MultiViewModel,span_mask,learning_rate
from common import RUN,config,verify_freeze,deterministic,save,savet,sha,metric
SEEDS=[21729,23407,22026]
TARGETS={'hard_scalar':(True,1.0),'soft_scalar':(True,.5),'hard_shape':(False,1.0),'soft_shape':(False,.5)}

def apply_order_component(tam, generator, kind, probability=.75, span_bins=24, raw_per_bin=15):
    if kind not in TARGETS: raise ValueError(kind)
    if tam.ndim != 3 or tam.shape[1:] != (2, 1800): raise ValueError('expected B x 2 x 1800 TAM')
    b=tam.shape[0]; alpha_shape,alpha=TARGETS[kind]
    active=torch.rand(b,generator=generator)<probability
    starts=torch.randint(0,120-span_bins+1,(b,),generator=generator)
    x=tam.reshape(b,2,120,raw_per_bin)
    bins=torch.arange(120)
    inside=(bins[None,:]>=starts[:,None])&(bins[None,:]<starts[:,None]+span_bins)
    m=inside.to(tam.device)[:,None,:,None]
    if alpha_shape:
        denom=m.sum((2,3),keepdim=True).clamp_min(1) * raw_per_bin
        template=(x*m).sum((2,3),keepdim=True)/denom
    else:
        denom=m.sum(2,keepdim=True).clamp_min(1)
        template=(x*m).sum(2,keepdim=True)/denom
    template=template.expand_as(x)
    mixed=(1.0-alpha)*x+alpha*template
    out=torch.where(m,mixed,x).reshape_as(tam)
    return torch.where(active.to(tam.device)[:,None,None],out,tam)

def predict(model,e,bs):
    model.eval();before={k:v.detach().clone() for k,v in model.named_buffers()};out=[]
    with torch.inference_mode():
        for i in range(0,len(e['labels']),bs):out.append(model(e['timestamps'][i:i+bs],e['tam'][i:i+bs]).cpu().numpy())
    after=dict(model.named_buffers());assert all(torch.equal(before[k],after[k]) for k in before);return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True,choices=['ce_control']+list(TARGETS));ap.add_argument('--seed',required=True,type=int,choices=SEEDS);ap.add_argument('--device',required=True,choices=['cuda']);a=ap.parse_args();c=config();verify_freeze();assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0';deterministic('cuda',2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device('cuda')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');data={r:{k:v.to(device) for k,v in e.items() if k in ('timestamps','tam','labels')} for r,e in raw.items()};labels={r:e['labels'].cpu().numpy() for r,e in data.items()};out=RUN/'artifacts'/f'{a.kind}_{a.seed}';out.mkdir(exist_ok=False);model=MultiViewModel('ce_only').to(device);savet(out/'initial_state.pt',{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    n=len(data['source']['labels']);bs=c['batch_size'];steps=c['optimizer_steps'];pieces=[];count=0;cycle=0
    while count<steps*bs:pieces.append(torch.randperm(n,generator=torch.Generator().manual_seed(a.seed+9000+cycle)));count+=n;cycle+=1
    indices=torch.cat(pieces)[:steps*bs].reshape(steps,bs);savet(out/'index_stream.pt',indices);span_rng=torch.Generator().manual_seed(a.seed+50000);component_rng=torch.Generator().manual_seed(a.seed+70000);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8);start=time.monotonic();history=[];best=-1.;beststep=0;sums={'loss':0.};prev=0
    print(json.dumps({'task':f'{a.kind}_{a.seed}','device':'cuda:0','steps':steps}),flush=True)
    for step in range(1,steps+1):
        if time.monotonic()-start>c['gpu_job_seconds']:raise TimeoutError(a.kind)
        for g in opt.param_groups:g['lr']=learning_rate(step,steps)
        idx=indices[step-1].cuda();e=data['source'];tam=span_mask(e['tam'][idx],span_rng)
        if a.kind!='ce_control':tam=apply_order_component(tam,component_rng,a.kind)
        y=e['labels'][idx];model.train();opt.zero_grad(set_to_none=True);logits=model(e['timestamps'][idx],tam);loss=F.cross_entropy(logits,y);loss.backward();opt.step();sums['loss']+=float(loss.detach())
        if step%100==0:save(out/'progress.json',{'task':f'{a.kind}_{a.seed}','step':step,'total_steps':steps,'best_step':beststep,'elapsed_seconds':time.monotonic()-start})
        if step in c['eval_steps']:
            pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()};scores={r:metric(labels[r],z.argmax(1)) for r,z in pred.items()};row={'step':step,'lr':learning_rate(step,steps),'train_loss':sums['loss']/(step-prev),'elapsed_seconds':time.monotonic()-start,**scores};history.append(row);sums={'loss':0.};prev=step
            if scores['valid']['accuracy']>best:best=scores['valid']['accuracy'];beststep=step;savet(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',{'state_dict':model.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            save(out/'history.json',history);print(json.dumps({'task':f'{a.kind}_{a.seed}',**row}),flush=True)
    ck=torch.load(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',weights_only=True,map_location=device);model.load_state_dict(ck['state_dict']);pred={r:predict(model,e,c['eval_batch_size']) for r,e in data.items()};np.savez_compressed(out/'logits_best.npz',**pred);ve=data['valid'];fixed=span_mask(ve['tam'],torch.Generator().manual_seed(c['diagnostics']['valid_fixed_mask_seed']));masked=predict(model,{**ve,'tam':fixed},c['eval_batch_size']);diag={'clean_accuracy':metric(labels['valid'],pred['valid'].argmax(1))['accuracy'],'masked_accuracy':metric(labels['valid'],masked.argmax(1))['accuracy'],'prediction_agreement':float((pred['valid'].argmax(1)==masked.argmax(1)).mean())};reload=MultiViewModel('ce_only').cuda();reload.load_state_dict(ck['state_dict'])
    for r,e in data.items():assert np.array_equal(predict(reload,e,c['eval_batch_size']).argmax(1),pred[r].argmax(1))
    report={'task':f'{a.kind}_{a.seed}','kind':a.kind,'seed':a.seed,'device':'cuda:0','best_step':beststep,'steps_completed':steps,'validation_opportunities':len(history),'parameters_total':sum(p.numel() for p in reload.parameters()),'parameters_trainable':sum(p.numel() for p in reload.parameters() if p.requires_grad),'verified_roles':['source','valid'],'future_access':False,'best':{r:metric(labels[r],z.argmax(1)) for r,z in pred.items()},'fixed_view_diagnostic':diag,'augmentation':({'name':a.kind,'probability':.75,'span_bins':24,'raw_per_bin':15,'alpha':(TARGETS[a.kind][1] if a.kind!='ce_control' else 0.0),'template':('scalar_interval_mean' if a.kind.endswith('scalar') else 'per_offset_interval_mean'),'source_only':True,'channel_total_preserved':True} if a.kind!='ce_control' else {'name':'none','existing_span_mask':True}),'elapsed_seconds':time.monotonic()-start};save(out/'report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
