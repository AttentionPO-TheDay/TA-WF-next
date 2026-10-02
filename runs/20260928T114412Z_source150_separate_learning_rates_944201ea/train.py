import argparse,time,os,json
from common import *
from torch.nn import functional as F

def train(name,seed):
    c=config_read();assert seed in c['seeds'];cond=condition_config(c,name)
    torch.set_num_threads(c['resource']['threads_per_worker']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();key=f'{name}_{seed}';out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
    data=load_data(c);entry=data['source150'];entries={'source':entry,'valid':data['valid']};y=entry['labels']
    batches={k:make_batch(v) for k,v in entries.items()};indices=index_stream(len(y),seed,c);atomic_torch(out/'index_stream.pt',indices)
    counts=torch.bincount(indices.flatten(),minlength=len(y));atomic_torch(out/'sample_visits.pt',counts)
    torch.manual_seed(seed);model=make_model(c,name);initial=state_hash(model);ginitial=state_hash(model.generator)
    initial_generator={k:v.detach().clone() for k,v in model.generator.state_dict().items()}
    with torch.inference_mode():
        diagidx=torch.tensor([torch.where(y==label)[0][0] for label in range(32)]);diagnostic=batches['source'].subset(diagidx);initial_tokens=diagnostic.tokens(model).clone()
    optimizer=make_optimizer(model,c,name);history=[];best=-1.;best_block=None
    print(json.dumps({'condition':name,'seed':seed,'pid':os.getpid(),'initial_state_sha256':initial,'samples':len(y),'steps':len(indices),'parameters':parameter_counts(model),'group_lrs':group_lrs(c,name,1)}),flush=True)
    for block in range(1,c['blocks']+1):
        if time.monotonic()-started>c['resource']['job_seconds']: raise TimeoutError('job budget')
        model.train();total=0.;correct=0;first_grad={}
        for step in range((block-1)*c['steps_per_block'],block*c['steps_per_block']):
            lrs=group_lrs(c,name,step+1)
            for group in optimizer.param_groups: group['lr']=lrs[group['role']]
            idx=indices[step];optimizer.zero_grad(set_to_none=True)
            logits=batches['source'].subset(idx).logits(model);loss=F.cross_entropy(logits,y[idx]);assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if step==0:
                first_grad={'generator':{k:float(p.grad.norm()) for k,p in model.generator.named_parameters() if p.grad is not None},'classifier_norm':float(torch.sqrt(sum((p.grad**2).sum() for p in model.classifier.parameters() if p.grad is not None)))}
                assert all(first_grad['generator'][k]>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight']) and first_grad['classifier_norm']>0
            optimizer.step();total+=float(loss.detach());correct+=int((logits.argmax(1)==y[idx]).sum())
        row={'block':block,'step':block*c['steps_per_block'],'learning_rates':group_lrs(c,name,block*c['steps_per_block']),'train_ce':total/c['steps_per_block'],'online_train_accuracy':correct/(c['steps_per_block']*c['batch_size'])}
        if first_grad:row['first_gradient_norms']=first_grad
        if row['step'] in c['eval_steps']:
            for role in entries:
                p=predict(model,batches[role],c['batch_size']);row[role]=metric(entries[role]['labels'].numpy(),p)
            with torch.inference_mode(): row['generator_token_delta_rms']=float((diagnostic.tokens(model)-initial_tokens).square().mean().sqrt())
            row['generator_parameter_delta_l2']=sum(float((v-initial_generator[k]).square().sum()) for k,v in model.generator.state_dict().items())**.5
            assert row['generator_parameter_delta_l2']>0 and row['generator_token_delta_rms']>0
            if row['valid']['macro_f1']>best: 
                best=row['valid']['macro_f1'];best_block=block
                atomic_torch(RUN/'checkpoints'/f'{key}_best.pt',{'state_dict':model.state_dict(),'block':block,'step':row['step'],'config_sha256':sha(RUN/'config.json'),'initial_state_sha256':initial})
        row['elapsed_seconds']=time.monotonic()-started;history.append(row);atomic_json(out/'history.json',history)
        atomic_torch(RUN/'checkpoints'/f'{key}_latest.pt',{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'block':block,'step':row['step'],'next_index_stream_row':row['step'],'history':history,'config_sha256':sha(RUN/'config.json'),'elapsed_seconds':row['elapsed_seconds']})
        atomic_json(out/'progress.json',{'status':'running','block':block,'step':row['step'],'best_block':best_block,'elapsed_seconds':row['elapsed_seconds']})
        if 'valid' in row: print(json.dumps(row),flush=True)
    last_preds={role:predict(model,batches[role],c['batch_size']) for role in entries};np.savez_compressed(out/'predictions_last.npz',**last_preds);last_scores={role:metric(entries[role]['labels'].numpy(),last_preds[role]) for role in entries}
    state=torch.load(RUN/'checkpoints'/f'{key}_best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['state_dict']);preds={role:predict(model,batches[role],c['batch_size']) for role in entries};np.savez_compressed(out/'predictions.npz',**preds)
    report={'condition':name,'seed':seed,'status':'completed','historical_reuse':False,'steps_completed':c['optimizer_steps'],'blocks_completed':c['blocks'],'last':last_scores,'best_block':best_block,'initial_state_sha256':initial,'shared_initial_sha256':initial,'generator_initial_sha256':ginitial,'generator_changed':state_hash(model.generator)!=ginitial,'config_sha256':sha(RUN/'config.json'),'source_count':len(y),'sample_visits_min':int(counts.min()),'sample_visits_max':int(counts.max()),'samples_presented':int(counts.sum()),'elapsed_seconds':time.monotonic()-started,'parameters':parameter_counts(model),'condition_multipliers':cond}
    for role in entries: report[role]=metric(entries[role]['labels'].numpy(),preds[role])
    atomic_json(out/'report.json',report);atomic_json(out/'progress.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();train(a.condition,a.seed)
