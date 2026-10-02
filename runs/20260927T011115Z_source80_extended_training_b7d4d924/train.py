"""Fixed-step supervised training, with nested sample counts and paired readouts."""
import argparse,time
from common import *
from torch.nn import functional as F

def train(name,seed):
    c=config_read();cond=condition_config(c,name);assert seed in c['seeds']
    torch.set_num_threads(c['resource']['threads_per_worker']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();key=f'{name}_{seed}';out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
    data=load_data(c);entry=data[f"source{cond['per_class']}"]
    entries={'source':entry,'common_source':data['source20'],'valid':data['valid']}
    batches={k:make_batch(v) for k,v in entries.items()};y=entry['labels']
    indices=index_stream(len(y),seed,c);atomic_torch(out/'index_stream.pt',indices)
    counts=torch.bincount(indices.flatten(),minlength=len(y));atomic_torch(out/'sample_visits.pt',counts)
    torch.manual_seed(seed);model=make_model(c,name);initial=state_hash(model);ginitial=state_hash(model.generator)
    initial_generator={k:v.detach().clone() for k,v in model.generator.state_dict().items()}
    diagidx=torch.tensor([torch.where(data['source20']['labels']==label)[0][0] for label in range(32)])
    diagnostic=batches['common_source'].subset(diagidx);model.eval()
    with torch.inference_mode():initial_tokens=diagnostic.tokens(model).clone()
    optimizer=make_optimizer(model,c,name)
    prior=ROOT/c['prior_run'];anchor_path=prior/'checkpoints'/f'{key}_latest.pt'
    old=torch.load(anchor_path,map_location='cpu',weights_only=False)
    old_report=json.loads((prior/'artifacts'/key/'report.json').read_text())
    assert old_report['initial_state_sha256']==initial and old_report['generator_initial_sha256']==ginitial
    assert old['step']==old['next_index_stream_row']==c['resume_step'] and old['block']==c['resume_block']
    old_indices=torch.load(prior/'artifacts'/key/'index_stream.pt',map_location='cpu',weights_only=True)
    assert torch.equal(indices[:c['resume_step']],old_indices)
    model.load_state_dict(old['state_dict']);optimizer.load_state_dict(old['optimizer'])
    assert all(int(v['step'])==c['resume_step'] for v in optimizer.state.values())
    resumed_state=state_hash(model);replay={}
    for role,e in entries.items():
        rp=predict(model,batches[role]);replay[role]=metric(e['labels'].numpy(),rp)
        for k,v in replay[role].items():assert abs(v-old['history'][-1][role][k])<1e-12,'resume metric mismatch'
    atomic_json(out/'resume_audit.json',{'complete':True,'step':c['resume_step'],'anchor_sha256':sha(anchor_path),'state_sha256':resumed_state,'optimizer_step':c['resume_step'],'metrics':replay,'index_prefix_exact':True})
    history=list(old['history']);best=-1;best_block=None
    torch.set_rng_state(old['torch_rng'])
    print(json.dumps({'condition':name,'seed':seed,'pid':os.getpid(),'initial_state_sha256':initial,'samples':len(y),'steps':len(indices),'sample_visits_min':int(counts.min()),'sample_visits_max':int(counts.max()),'parameters':parameter_counts(model)}),flush=True)
    for block in range(c['resume_block']+1,c['blocks']+1):
        if time.monotonic()-started>c['resource']['job_seconds']:raise TimeoutError('job budget')
        model.train();total=0.;correct=0;grad={}
        for step in range((block-1)*c['steps_per_block'],block*c['steps_per_block']):
            idx=indices[step]
            for group in optimizer.param_groups:group['lr']=lr_for_step(c,step+1)
            optimizer.zero_grad(set_to_none=True)
            logits=batches['source'].subset(idx).logits(model);loss=F.cross_entropy(logits,y[idx]);assert torch.isfinite(loss)
            loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if step==c['resume_step']:
                grad={k:float(p.grad.norm()) if p.grad is not None else 0. for k,p in model.generator.named_parameters()}
                assert all(grad[k]>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight'])
            optimizer.step();total+=float(loss.detach());correct+=int((logits.argmax(1)==y[idx]).sum())
        row={'block':block,'step':block*c['steps_per_block'],'learning_rate':optimizer.param_groups[0]['lr'],'train_ce':total/c['steps_per_block'],'online_train_accuracy':correct/(c['steps_per_block']*c['batch_size'])}
        if grad:row['first_generator_grad_norms']=grad
        if (block-c['resume_block'])%c['eval_every_blocks']==0:
            for role in entries:
                if role=='common_source' and cond['per_class']==20:row[role]=row['source'];continue
                p=predict(model,batches[role],c['batch_size']);row[role]=metric(entries[role]['labels'].numpy(),p)
            with torch.inference_mode():row['generator_token_delta_rms']=float((diagnostic.tokens(model)-initial_tokens).square().mean().sqrt())
            row['generator_parameter_delta_l2']=sum(float((v-initial_generator[k]).square().sum()) for k,v in model.generator.state_dict().items())**.5
            assert row['generator_parameter_delta_l2']>0 and row['generator_token_delta_rms']>0
            if row['valid']['macro_f1']>best:
                best=row['valid']['macro_f1'];best_block=block
                atomic_torch(RUN/'checkpoints'/f'{key}_best.pt',{'state_dict':model.state_dict(),'block':block,'step':row['step'],'config_sha256':sha(RUN/'config.json'),'initial_state_sha256':initial})
        row['elapsed_seconds']=time.monotonic()-started;history.append(row)
        atomic_torch(RUN/'checkpoints'/f'{key}_latest.pt',{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'block':block,'step':row['step'],'next_index_stream_row':row['step'],'history':history,'config_sha256':sha(RUN/'config.json'),'elapsed_seconds':row['elapsed_seconds']})
        atomic_json(out/'history.json',history);atomic_json(out/'progress.json',{'status':'running','block':block,'step':row['step'],'best_block':best_block,'elapsed_seconds':row['elapsed_seconds']})
        if 'valid' in row:print(json.dumps(row),flush=True)
    assert state_hash(model)!=resumed_state
    last_preds={role:predict(model,batches[role],c['batch_size']) for role in entries}
    np.savez_compressed(out/'predictions_last.npz',**last_preds)
    last_scores={role:metric(entries[role]['labels'].numpy(),last_preds[role]) for role in entries}
    model.load_state_dict(torch.load(RUN/'checkpoints'/f'{key}_best.pt',map_location='cpu',weights_only=True)['state_dict'])
    preds={role:predict(model,batches[role],c['batch_size']) for role in entries}
    np.savez_compressed(out/'predictions.npz',**preds)
    report={'condition':name,'seed':seed,'status':'completed','steps_completed':c['optimizer_steps'],'new_steps_completed':c['optimizer_steps']-c['resume_step'],'blocks_completed':c['blocks'],'last':last_scores,'resume_metrics':replay,'historical_best':{role:old_report[role] for role in entries},'resume_state_sha256':resumed_state,'best_block':best_block,'initial_state_sha256':initial,'generator_initial_sha256':ginitial,'generator_changed':state_hash(model.generator)!=ginitial,'config_sha256':sha(RUN/'config.json'),'source_count':len(y),'sample_visits_min':int(counts.min()),'sample_visits_max':int(counts.max()),'samples_presented':int(counts.sum()),'elapsed_seconds':time.monotonic()-started,'parameters':parameter_counts(model)}
    for role in entries:report[role]=metric(entries[role]['labels'].numpy(),preds[role])
    atomic_json(out/'report.json',report);atomic_json(out/'progress.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();train(a.condition,a.seed)
