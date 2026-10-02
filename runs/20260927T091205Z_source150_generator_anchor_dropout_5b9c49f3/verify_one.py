"""Independent checkpoint/prediction, update, schedule and selection verification."""
import argparse
from common import *

def independent_metric(y,p):
    scores=[]
    for label in range(102):
        den=np.count_nonzero(y==label)+np.count_nonzero(p==label)
        scores.append(2*np.count_nonzero((y==label)&(p==label))/den if den else 0.)
    return {'accuracy':float(np.mean(y==p)),'macro_f1':float(np.mean(scores))}

def main(name,seed):
    c=config_read();cond=condition_config(c,name);key=f'{name}_{seed}';out=RUN/'artifacts'/key
    assert not cond['reuse'];torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    report=json.loads((out/'report.json').read_text());assert report['status']=='completed' and report['steps_completed']==c['optimizer_steps']
    assert report['config_sha256']==sha(RUN/'config.json')
    data=load_data(c);entries={'source':data[f"source{cond['per_class']}"],'common_source':data['source80'],'valid':data['valid']};n=len(entries['source']['labels'])
    ix=torch.load(out/'index_stream.pt',map_location='cpu',weights_only=True)
    assert torch.equal(ix,index_stream(n,seed,c)) and ix.shape==(c['optimizer_steps'],c['batch_size'])
    counts=torch.bincount(ix.flatten(),minlength=n);assert torch.equal(counts,torch.load(out/'sample_visits.pt',weights_only=True))
    assert int(counts.sum())==819200==report['samples_presented'] and int(counts.max()-counts.min())<=1
    torch.manual_seed(seed);model=make_model(c,name);assert state_hash(model)==report['initial_state_sha256'] and shared_state_hash(model)==report['shared_initial_sha256']
    assert report['generator_anchor_lambda']==cond['generator_anchor_lambda'] and report['classifier_dropout']==cond['dropout']
    for module in model.classifier.modules():
        if isinstance(module,torch.nn.Dropout):assert module.p==cond['dropout']
        if isinstance(module,torch.nn.MultiheadAttention):assert module.dropout==cond['dropout']
    assert state_hash(model.generator)==report['generator_initial_sha256'];initial={k:v.detach().clone() for k,v in model.generator.state_dict().items()}
    added={k:v.detach().clone() for k,v in model.state_dict().items() if '.local_encoders.' in k and '.layers.1.' in k}
    history=json.loads((out/'history.json').read_text());assert [h['block'] for h in history]==list(range(1,401)) and [h['step'] for h in history]==list(range(32,12801,32))
    scored=[h for h in history if 'valid' in h];assert [h['step'] for h in scored]==c['eval_steps']
    for h in history:
        assert h['learning_rate']==lr_for_step(c,h['step'])
        assert h['train_anchor_penalty']>=0
        assert abs(h['train_total_loss']-h['train_ce']-h['train_anchor_penalty'])<1e-5
        if cond['generator_anchor_lambda']==0:assert h['train_anchor_penalty']==0
    assert all(history[0]['first_generator_grad_norms'][k]>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight'])
    if added:assert set(history[0]['first_added_layer_grad_norms'])==set(added) and all(v>0 for v in history[0]['first_added_layer_grad_norms'].values())
    chosen=max(scored,key=lambda h:h['valid']['macro_f1']);assert chosen['block']==report['best_block']
    checked=0;paths=[]
    for kind,ref,predfile,scores in [('best',chosen,'predictions.npz',report),('latest',history[-1],'predictions_last.npz',report['last'])]:
        path=RUN/'checkpoints'/f'{key}_{kind}.pt';state=torch.load(path,map_location='cpu',weights_only=kind=='best');paths.append(path)
        assert state['step']==ref['step'] and state['config_sha256']==sha(RUN/'config.json')
        model.load_state_dict(state['state_dict']);assert state_hash(model)!=report['initial_state_sha256']
        delta=sum(float((v.detach()-initial[k]).double().square().sum()) for k,v in model.generator.state_dict().items())**.5
        assert delta>0 and abs(delta-ref['generator_parameter_delta_l2'])<1e-5
        expected_penalty=cond['generator_anchor_lambda']/2*delta**2
        assert abs(expected_penalty-ref['generator_anchor_penalty'])<1e-6
        delta_added=sum(float((v.detach()-added[k]).double().square().sum()) for k,v in model.state_dict().items() if k in added)**.5
        assert abs(delta_added-ref['added_layer_parameter_delta_l2'])<1e-5
        if added:assert all(not torch.equal(v,added[k]) for k,v in model.state_dict().items() if k in added)
        with np.load(out/predfile,allow_pickle=False) as saved:
            for role,e in entries.items():
                p=predict(model,make_batch(e));assert np.array_equal(p,saved[role]),(kind,role)
                for k,v in independent_metric(e['labels'].numpy(),p).items():assert abs(v-scores[role][k])<1e-12 and abs(v-ref[role][k])<1e-12
                checked+=1
        paths.append(out/predfile)
        if kind=='latest':
            assert state['step']==state['next_index_stream_row']==12800 and state['history']==history
            assert all(int(v['step'])==12800 for v in state['optimizer']['state'].values())
            assert len(state['optimizer']['state'])==len(list(model.parameters()))
            for group in state['optimizer']['param_groups']:assert group['lr']==.0001 and group['weight_decay']==.0001
    paths+=[out/x for x in ['report.json','history.json','index_stream.pt','sample_visits.pt']]
    atomic_json(out/'verification.json',{'complete':True,'prediction_sets_checked':checked,'historical_reuse':False,'lr_verified':True,'selection_verified':True,'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'errors':[]})
    print('VERIFIED',key,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();main(a.condition,a.seed)
