"""Independent best/latest replay and bounded continuation verification."""
import argparse
from common import *

def independent_metric(y,p):
    scores=[]
    for label in range(102):
        den=np.count_nonzero(y==label)+np.count_nonzero(p==label)
        scores.append(2*np.count_nonzero((y==label)&(p==label))/den if den else 0.)
    return {'accuracy':float(np.mean(y==p)),'macro_f1':float(np.mean(scores))}

def main(name,seed):
    c=config_read();cond=condition_config(c,name);key=f'{name}_{seed}';out=RUN/'artifacts'/key;prior=ROOT/c['prior_run']
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    report=json.loads((out/'report.json').read_text());assert report['status']=='completed' and report['steps_completed']==12800 and report['new_steps_completed']==9600
    assert report['config_sha256']==sha(RUN/'config.json')
    data=load_data(c);entries={'source':data['source80'],'common_source':data['source20'],'valid':data['valid']}
    ix=torch.load(out/'index_stream.pt',map_location='cpu',weights_only=True)
    assert torch.equal(ix,index_stream(8160,seed,c)) and ix.shape==(12800,64)
    assert torch.equal(ix[:3200],torch.load(prior/'artifacts'/key/'index_stream.pt',weights_only=True))
    counts=torch.bincount(ix.flatten(),minlength=8160);assert torch.equal(counts,torch.load(out/'sample_visits.pt',weights_only=True))
    assert int(counts.sum())==819200==report['samples_presented'] and int(counts.max()-counts.min())<=1
    torch.manual_seed(seed);model=make_model(c,name);assert state_hash(model)==report['initial_state_sha256']
    assert state_hash(model.generator)==report['generator_initial_sha256'];initial={k:v.detach().clone() for k,v in model.generator.state_dict().items()}
    original=torch.load(prior/'checkpoints'/f'{key}_latest.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(original['state_dict']);assert state_hash(model)==report['resume_state_sha256']
    audit=json.loads((out/'resume_audit.json').read_text());assert audit['complete'] and audit['optimizer_step']==3200
    history=json.loads((out/'history.json').read_text());assert history[:100]==original['history']
    assert [h['block'] for h in history]==list(range(1,401)) and [h['step'] for h in history]==list(range(32,12801,32))
    new=history[100:];scored=[h for h in new if 'valid' in h]
    assert [h['step'] for h in scored]==list(range(3680,12801,480))
    for h in new:
        expected=.001 if h['step']<=6400 else .0003 if h['step']<=9600 else .0001
        assert h['learning_rate']==expected
    assert all(new[0]['first_generator_grad_norms'][k]>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight'])
    chosen=max(scored,key=lambda h:h['valid']['macro_f1']);assert chosen['block']==report['best_block']
    checked=0;paths=[]
    for kind,ref,predfile,scores in [('best',chosen,'predictions.npz',report),('latest',history[-1],'predictions_last.npz',report['last'])]:
        path=RUN/'checkpoints'/f'{key}_{kind}.pt';state=torch.load(path,map_location='cpu',weights_only=kind=='best');paths.append(path)
        assert state['step']==ref['step'] and state['config_sha256']==sha(RUN/'config.json')
        model.load_state_dict(state['state_dict']);assert state_hash(model)!=report['resume_state_sha256']
        delta=sum(float((v.detach()-initial[k]).double().square().sum()) for k,v in model.generator.state_dict().items())**.5
        assert abs(delta-ref['generator_parameter_delta_l2'])<1e-5
        with np.load(out/predfile,allow_pickle=False) as saved:
            for role,e in entries.items():
                p=predict(model,make_batch(e));assert np.array_equal(p,saved[role]),(kind,role)
                for k,v in independent_metric(e['labels'].numpy(),p).items():assert abs(v-scores[role][k])<1e-12 and abs(v-ref[role][k])<1e-12
                checked+=1
        paths.append(out/predfile)
        if kind=='latest':
            assert state['step']==state['next_index_stream_row']==12800
            assert all(int(v['step'])==12800 for v in state['optimizer']['state'].values())
            for group in state['optimizer']['param_groups']:assert group['lr']==.0001 and group['weight_decay']==.0001
    paths+=[out/x for x in ['report.json','history.json','resume_audit.json','index_stream.pt','sample_visits.pt']]
    atomic_json(out/'verification.json',{'complete':True,'prediction_sets_checked':checked,'resume_verified':True,'lr_verified':True,'selection_verified':True,'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'errors':[]})
    print('VERIFIED',key,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();main(a.condition,a.seed)
