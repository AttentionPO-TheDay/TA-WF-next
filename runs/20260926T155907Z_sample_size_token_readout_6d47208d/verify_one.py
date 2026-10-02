"""Independent checkpoint replay, sampler, selection and metric verification."""
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
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    report=json.loads((out/'report.json').read_text());assert report['status']=='completed' and report['steps_completed']==3200
    assert report['config_sha256']==sha(RUN/'config.json')
    data=load_data(c);entries={'source':data[f"source{cond['per_class']}"],'common_source':data['source20'],'valid':data['valid']}
    ix=torch.load(out/'index_stream.pt',map_location='cpu',weights_only=True)
    assert torch.equal(ix,index_stream(len(entries['source']['labels']),seed,c));assert ix.shape==(3200,64)
    counts=torch.bincount(ix.flatten(),minlength=len(entries['source']['labels']));saved_counts=torch.load(out/'sample_visits.pt',weights_only=True)
    assert torch.equal(counts,saved_counts) and int(counts.max()-counts.min())<=1
    assert int(counts.sum())==204800==report['samples_presented'];assert int(counts.min())==report['sample_visits_min']
    torch.manual_seed(seed);model=make_model(c,name);assert state_hash(model)==report['initial_state_sha256']
    gi=state_hash(model.generator);assert gi==report['generator_initial_sha256'];init={k:v.detach().clone() for k,v in model.generator.state_dict().items()}
    assert parameter_counts(model)==report['parameters']
    history=json.loads((out/'history.json').read_text());assert [h['block'] for h in history]==list(range(1,101))
    assert [h['step'] for h in history]==list(range(32,3201,32))
    scored=[h for h in history if 'valid' in h];assert [h['step'] for h in scored]==list(range(160,3201,160))
    chosen=max(scored,key=lambda h:h['valid']['macro_f1']);assert chosen['block']==report['best_block']
    best_path=RUN/'checkpoints'/f'{key}_best.pt';best=torch.load(best_path,map_location='cpu',weights_only=True)
    assert best['block']==chosen['block'] and best['step']==chosen['step'] and best['config_sha256']==sha(RUN/'config.json')
    model.load_state_dict(best['state_dict']);assert state_hash(model.generator)!=gi
    delta=sum(float((v.detach()-init[k]).double().square().sum()) for k,v in model.generator.state_dict().items())**.5
    assert abs(delta-chosen['generator_parameter_delta_l2'])<1e-5
    assert all(history[0]['first_generator_grad_norms'][k]>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight'])
    checked=0
    with np.load(out/'predictions.npz',allow_pickle=False) as saved:
        assert set(saved.files)==set(entries)
        for role,e in entries.items():
            p=predict(model,make_batch(e));assert np.array_equal(p,saved[role])
            for k,v in independent_metric(e['labels'].numpy(),p).items():
                assert abs(v-report[role][k])<1e-12 and abs(v-chosen[role][k])<1e-12
            checked+=1
    latest=torch.load(RUN/'checkpoints'/f'{key}_latest.pt',map_location='cpu',weights_only=False)
    assert latest['step']==latest['next_index_stream_row']==3200 and latest['block']==100
    assert all(int(v['step'])==3200 for v in latest['optimizer']['state'].values())
    opt=make_optimizer(model,c,name).state_dict()['param_groups'];actual=latest['optimizer']['param_groups'];assert len(opt)==len(actual)
    for a,b in zip(opt,actual):
        for f in ['lr','weight_decay','params']:assert a[f]==b[f]
    paths=[out/'report.json',out/'predictions.npz',out/'history.json',out/'index_stream.pt',out/'sample_visits.pt',best_path,RUN/'checkpoints'/f'{key}_latest.pt']
    atomic_json(out/'verification.json',{'complete':True,'prediction_sets_checked':checked,'sampling_stream_verified':True,'selection_verified':True,'generator_feedback_verified':True,'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'errors':[]})
    print('VERIFIED',key,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();main(a.condition,a.seed)
