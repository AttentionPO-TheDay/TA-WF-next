"""Source-only factorial; adapted from project's audited extended-training run."""
import argparse,time
from common import *
from torch.nn import functional as F

def train(name,seed):
    c=config_read();cond=condition_config(c,name);assert seed in c['seeds'] and not cond['reuse']
    torch.set_num_threads(c['resource']['threads_per_worker']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();key=f'{name}_{seed}';out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
    data=load_data(c);entry=data[f"source{cond['per_class']}"]
    entries={'source':entry,'common_source':data['source80'],'valid':data['valid']}
    batches={k:make_batch(v) for k,v in entries.items()};y=entry['labels']
    indices=index_stream(len(y),seed,c);atomic_torch(out/'index_stream.pt',indices)
    counts=torch.bincount(indices.flatten(),minlength=len(y));atomic_torch(out/'sample_visits.pt',counts)
    torch.manual_seed(seed);model=make_model(c,name);initial=state_hash(model);ginitial=state_hash(model.generator);shared=shared_state_hash(model)
    initial_generator={k:v.detach().clone() for k,v in model.generator.state_dict().items()}
    added={k:v.detach().clone() for k,v in model.state_dict().items() if '.local_encoders.' in k and '.layers.1.' in k}
    diagidx=torch.tensor([torch.where(data['source80']['labels']==label)[0][0] for label in range(32)])
    diagnostic=batches['common_source'].subset(diagidx);model.eval()
    with torch.inference_mode():initial_tokens=diagnostic.tokens(model).clone()
    optimizer=make_optimizer(model,c,name);history=[];best=-1;best_block=None
    print(json.dumps({'condition':name,'seed':seed,'pid':os.getpid(),'initial_state_sha256':initial,'shared_initial_sha256':shared,'samples':len(y),'steps':len(indices),'parameters':parameter_counts(model)}),flush=True)
    for block in range(1,c['blocks']+1):
        if time.monotonic()-started>c['resource']['job_seconds']:raise TimeoutError('job budget')
        model.train();total=0.;penalties=0.;objectives=0.;correct=0;grad={};extra_grad={}
        for step in range((block-1)*c['steps_per_block'],block*c['steps_per_block']):
            idx=indices[step]
            for group in optimizer.param_groups:group['lr']=lr_for_step(c,step+1)
            optimizer.zero_grad(set_to_none=True)
            logits=batches['source'].subset(idx).logits(model);ce=F.cross_entropy(logits,y[idx]);penalty=anchor_penalty(model,initial_generator,cond['generator_anchor_lambda']);loss=ce+penalty;assert torch.isfinite(loss)
            loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if step==0:
                grad={k:float(p.grad.norm()) if p.grad is not None else 0. for k,p in model.generator.named_parameters()}
                assert all(grad[k]>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight'])
                extra_grad={k:float(p.grad.norm()) if p.grad is not None else 0. for k,p in model.named_parameters() if k in added}
                assert all(v>0 for v in extra_grad.values())
            optimizer.step();total+=float(ce.detach());penalties+=float(penalty.detach());objectives+=float(loss.detach());correct+=int((logits.argmax(1)==y[idx]).sum())
        row={'block':block,'step':block*c['steps_per_block'],'learning_rate':optimizer.param_groups[0]['lr'],'train_ce':total/c['steps_per_block'],'train_anchor_penalty':penalties/c['steps_per_block'],'train_total_loss':objectives/c['steps_per_block'],'online_train_accuracy':correct/(c['steps_per_block']*c['batch_size'])}
        if grad:row.update(first_generator_grad_norms=grad,first_added_layer_grad_norms=extra_grad)
        if row['step'] in c['eval_steps']:
            for role in entries:
                if role=='common_source' and cond['per_class']==80:row[role]=dict(row['source']);continue
                p=predict(model,batches[role],c['batch_size']);row[role]=metric(entries[role]['labels'].numpy(),p)
            with torch.inference_mode():row['generator_token_delta_rms']=float((diagnostic.tokens(model)-initial_tokens).square().mean().sqrt())
            row['generator_parameter_delta_l2']=sum(float((v-initial_generator[k]).square().sum()) for k,v in model.generator.state_dict().items())**.5
            row['generator_anchor_penalty']=float(anchor_penalty(model,initial_generator,cond['generator_anchor_lambda']).detach())
            row['added_layer_parameter_delta_l2']=sum(float((v-added[k]).square().sum()) for k,v in model.state_dict().items() if k in added)**.5
            assert row['generator_parameter_delta_l2']>0 and row['generator_token_delta_rms']>0
            if added:assert row['added_layer_parameter_delta_l2']>0
            if row['valid']['macro_f1']>best:
                best=row['valid']['macro_f1'];best_block=block
                atomic_torch(RUN/'checkpoints'/f'{key}_best.pt',{'state_dict':model.state_dict(),'block':block,'step':row['step'],'config_sha256':sha(RUN/'config.json'),'initial_state_sha256':initial})
        row['elapsed_seconds']=time.monotonic()-started;history.append(row)
        atomic_torch(RUN/'checkpoints'/f'{key}_latest.pt',{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'block':block,'step':row['step'],'next_index_stream_row':row['step'],'history':history,'config_sha256':sha(RUN/'config.json'),'elapsed_seconds':row['elapsed_seconds']})
        atomic_json(out/'history.json',history);atomic_json(out/'progress.json',{'status':'running','block':block,'step':row['step'],'best_block':best_block,'elapsed_seconds':row['elapsed_seconds']})
        if 'valid' in row:print(json.dumps(row),flush=True)
    last_preds={role:predict(model,batches[role],c['batch_size']) for role in entries};np.savez_compressed(out/'predictions_last.npz',**last_preds)
    last_scores={role:metric(entries[role]['labels'].numpy(),last_preds[role]) for role in entries}
    model.load_state_dict(torch.load(RUN/'checkpoints'/f'{key}_best.pt',map_location='cpu',weights_only=True)['state_dict'])
    preds={role:predict(model,batches[role],c['batch_size']) for role in entries};np.savez_compressed(out/'predictions.npz',**preds)
    report={'condition':name,'seed':seed,'status':'completed','historical_reuse':False,'generator_anchor_lambda':cond['generator_anchor_lambda'],'classifier_dropout':cond['dropout'],'steps_completed':c['optimizer_steps'],'blocks_completed':c['blocks'],'last':last_scores,'best_block':best_block,'initial_state_sha256':initial,'shared_initial_sha256':shared,'generator_initial_sha256':ginitial,'generator_changed':state_hash(model.generator)!=ginitial,'config_sha256':sha(RUN/'config.json'),'source_count':len(y),'sample_visits_min':int(counts.min()),'sample_visits_max':int(counts.max()),'samples_presented':int(counts.sum()),'elapsed_seconds':time.monotonic()-started,'parameters':parameter_counts(model)}
    for role in entries:report[role]=metric(entries[role]['labels'].numpy(),preds[role])
    atomic_json(out/'report.json',report);atomic_json(out/'progress.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();train(a.condition,a.seed)
