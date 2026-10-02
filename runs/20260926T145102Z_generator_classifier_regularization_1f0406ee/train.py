"""Bounded CPU joint training; restart logic adapted from the prior CPU run."""
from __future__ import annotations
import argparse
import math
import random
import time
from common import *
from torch.nn import functional as F


def train(name,seed):
    config=config_read();c=condition_config(config,name);resource=config['resource_proposal']
    assert seed in config['seeds']
    config_sha=sha(RUN/'config.json')
    torch.set_num_threads(resource['threads_per_worker']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    key=f'{name}_{seed}';report_path=RUN/'artifacts'/f'metrics_{key}.json'
    if report_path.exists():
        old=json.loads(report_path.read_text())
        if old['status']=='completed' and old['config_sha256']==config_sha:
            print('already completed',key,flush=True);return
        raise RuntimeError('stopped/partial result requires audit; no extra budget')
    data=load_data(config);source=make_batch(data['source']);valid=make_batch(data['valid'])
    y=data['source']['labels'];vy=data['valid']['labels'].numpy()
    diag_idx=torch.tensor([torch.where(y==label)[0][0] for label in range(32)])
    diagnostic=source.subset(diag_idx)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=make_model(config,name);initial_sha=state_hash(model);generator_initial_sha=state_hash(model.generator)
    counts=parameter_counts(model)
    initial_generator={k:v.detach().clone() for k,v in model.generator.state_dict().items()}
    model.eval()
    with torch.inference_mode():initial_tokens=diagnostic.tokens(model).clone()
    optimizer=make_optimizer(model,config,name)
    latest_path=RUN/'checkpoints'/f'{key}_latest.pt';best_path=RUN/'checkpoints'/f'{key}_best.pt'
    first_epoch=1;history=[];best_f1=-1.;best_epoch=None;elapsed_base=0.
    if latest_path.exists():
        saved=torch.load(latest_path,map_location='cpu',weights_only=False)
        assert saved['config_sha256']==config_sha and saved['initial_state_sha256']==initial_sha
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        first_epoch=saved['epoch']+1;history=saved['history'];elapsed_base=saved['elapsed_seconds']
        best_f1=saved['best_f1'];best_epoch=saved['best_epoch']
    started=time.monotonic();elapsed=lambda:elapsed_base+time.monotonic()-started
    print(json.dumps({'key':key,'pid':os.getpid(),'device':'cpu','threads':torch.get_num_threads(),
                     'parameters':counts,'initial_state_sha256':initial_sha,'generator_initial_sha256':generator_initial_sha,
                     'start_epoch':first_epoch}),flush=True)
    status='completed'
    for epoch in range(first_epoch,config['epochs']+1):
        if elapsed()>=resource['time_limit_seconds_per_seed']:status='stopped_budget';break
        epoch_start=time.monotonic();model.train();loss_total=0.;ce_total=0.;anchor_total=0.;correct=0;grad_record=None
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+1000+epoch))
        for start in range(0,len(y),config['batch_size']):
            idx=order[start:start+config['batch_size']];optimizer.zero_grad(set_to_none=True)
            logits=source.subset(idx).logits(model);ce=F.cross_entropy(logits,y[idx])
            penalty=anchor_penalty(model,initial_generator,c['anchor_lambda']);loss=ce+penalty
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('nonfinite gradient')
            if start==0:
                grad_record={n:float(p.grad.norm()) if p.grad is not None else 0. for n,p in model.generator.named_parameters()}
                if epoch==1:
                    if c['generator_trainable']:
                        for n in ('conv1.weight','conv2.weight','projection.weight'):assert grad_record[n]>0,n+' missing feedback'
                        if c['pool']=='masked_segment_attention':assert grad_record['score.weight']>0,'score missing feedback'
                    else:assert all(p.grad is None for p in model.generator.parameters())
            optimizer.step();loss_total+=float(loss.detach())*len(idx);ce_total+=float(ce.detach())*len(idx);anchor_total+=float(penalty.detach())*len(idx);correct+=int((logits.argmax(1)==y[idx]).sum())
        record={'epoch':epoch,'train_loss':loss_total/len(y),'train_ce':ce_total/len(y),'train_anchor_penalty':anchor_total/len(y),'online_train_accuracy':correct/len(y),'generator_first_batch_grad_norms':grad_record}
        if epoch%config['eval_every']==0:
            sp=predict(model,source,config['batch_size']);vp=predict(model,valid,config['batch_size'])
            with torch.inference_mode():
                token_change=float((diagnostic.tokens(model)-initial_tokens).square().mean().sqrt())
                parameter_change=math.sqrt(sum(float((v-initial_generator[k]).square().sum()) for k,v in model.generator.state_dict().items()))
            if not c['generator_trainable']:
                assert state_hash(model.generator)==generator_initial_sha and token_change==0.,'frozen generator changed'
            else:
                assert parameter_change>0 and token_change>0,'generator did not update'
            record.update({'source':metric(y.numpy(),sp),'valid':metric(vy,vp),'valid_predicted_classes':int(len(np.unique(vp))),
                           'generator_token_delta_rms':token_change,'generator_parameter_delta_l2':parameter_change,
                           'pool_score_weight_l2':float(model.generator.score.weight.detach().norm()),
                           'anchor_penalty_at_eval':float(anchor_penalty(model,initial_generator,c['anchor_lambda']).detach())})
            if record['valid']['macro_f1']>best_f1:
                best_f1=record['valid']['macro_f1'];best_epoch=epoch
                atomic_torch(best_path,{'state_dict':model.state_dict(),'epoch':epoch,'condition':name,'seed':seed,
                                       'config_sha256':config_sha,'initial_state_sha256':initial_sha})
        record['epoch_seconds']=time.monotonic()-epoch_start;record['elapsed_seconds']=elapsed();history.append(record)
        atomic_torch(latest_path,{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                     'epoch':epoch,'history':history,'best_f1':best_f1,'best_epoch':best_epoch,'elapsed_seconds':elapsed(),
                     'initial_state_sha256':initial_sha,'config_sha256':config_sha})
        atomic_json(RUN/'artifacts'/f'history_{key}.json',{'history':history,'best_epoch':best_epoch,'config_sha256':config_sha})
        atomic_json(RUN/'artifacts'/f'progress_{key}.json',{'epoch':epoch,'elapsed_seconds':elapsed(),'best_epoch':best_epoch,
                    'best_valid_macro_f1':best_f1,'status':'running','pid':os.getpid()})
        if epoch%config['eval_every']==0:print(json.dumps({'key':key,**record}),flush=True)
    if not history or history[-1]['epoch']<config['epochs']:status='stopped_budget'
    report={'condition':name,'seed':seed,'status':status,'epochs_completed':history[-1]['epoch'] if history else 0,
            'best_epoch':best_epoch,'parameters':counts,'elapsed_seconds':elapsed(),'config_sha256':config_sha,
            'initial_state_sha256':initial_sha,'generator_initial_sha256':generator_initial_sha,'device':'cpu','threads':torch.get_num_threads()}
    if best_epoch is not None:
        best=torch.load(best_path,map_location='cpu',weights_only=True);model.load_state_dict(best['state_dict'])
        sp=predict(model,source,config['batch_size']);vp=predict(model,valid,config['batch_size'])
        report.update({'source':metric(y.numpy(),sp),'valid':metric(vy,vp),'valid_predicted_classes':int(len(np.unique(vp)))})
        np.savez_compressed(RUN/'artifacts'/f'predictions_{key}.npz',source=sp,valid=vp)
    atomic_json(report_path,report);atomic_json(RUN/'artifacts'/f'progress_{key}.json',report)
    print(json.dumps(report),flush=True)
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True);parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args();train(args.condition,args.seed)
