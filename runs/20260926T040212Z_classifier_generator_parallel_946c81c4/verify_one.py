"""Independent metrics and checkpoint replay for one completed CPU job."""
from __future__ import annotations
import argparse
import traceback
from common import *


def independent_metric(y,p):
    scores=[]
    for label in range(102):
        tp=np.count_nonzero((y==label)&(p==label));den=np.count_nonzero(y==label)+np.count_nonzero(p==label)
        scores.append(2*tp/den if den else 0.)
    return {'accuracy':float(np.mean(y==p)),'macro_f1':float(np.mean(scores))}


def verify(name,seed):
    key=f'{name}_{seed}';checked=0;errors=[];digests={};generator_check={}
    try:
        config=config_read();config_sha=sha(RUN/'config.json');c=condition_config(config,name)
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        data=load_data(config)
        paths={'report':RUN/'artifacts'/f'metrics_{key}.json','history':RUN/'artifacts'/f'history_{key}.json',
               'best':RUN/'checkpoints'/f'{key}_best.pt','latest':RUN/'checkpoints'/f'{key}_latest.pt',
               'predictions':RUN/'artifacts'/f'predictions_{key}.npz'}
        report=json.loads(paths['report'].read_text());assert report['config_sha256']==config_sha
        assert report['status']=='completed' and report['epochs_completed']==100,'training incomplete'
        assert report['device']=='cpu' and report['threads']==config['resource_proposal']['threads_per_worker']
        h=json.loads(paths['history'].read_text())['history']
        assert [row['epoch'] for row in h]==list(range(1,101))
        scored=[row for row in h if 'valid' in row];assert [row['epoch'] for row in scored]==list(range(5,101,5))
        chosen=max(scored,key=lambda row:row['valid']['macro_f1'])
        assert chosen['epoch']==report['best_epoch']
        state=torch.load(paths['best'],map_location='cpu',weights_only=True)
        assert state['config_sha256']==config_sha and state['epoch']==chosen['epoch']
        torch.manual_seed(seed);model=make_model(config,name)
        assert state_hash(model)==report['initial_state_sha256']==state['initial_state_sha256']
        assert parameter_counts(model)==report['parameters']
        init_generator=state_hash(model.generator)
        assert init_generator==report['generator_initial_sha256']
        y=data['source']['labels'];idx=torch.tensor([torch.where(y==label)[0][0] for label in range(32)])
        diagnostic=make_batch(data['source']).subset(idx)
        model.eval()
        with torch.inference_mode():initial_tokens=diagnostic.tokens(model).clone()
        model.load_state_dict(state['state_dict'])
        with torch.inference_mode():token_change=float((diagnostic.tokens(model)-initial_tokens).square().mean().sqrt())
        assert abs(token_change-chosen['generator_token_delta_rms'])<1e-7
        generator_changed=state_hash(model.generator)!=init_generator
        assert generator_changed==c['generator_trainable']
        if not c['generator_trainable']:
            assert token_change==0 and all(row['generator_parameter_delta_l2']==0 for row in scored)
            assert all(all(v==0 for v in row['generator_first_batch_grad_norms'].values()) for row in h)
        else:
            assert token_change>0
            for n in ('conv1.weight','conv2.weight','projection.weight'):assert h[0]['generator_first_batch_grad_norms'][n]>0
            if c['pool']=='masked_segment_attention':
                assert h[0]['generator_first_batch_grad_norms']['score.weight']>0 and model.generator.score.weight.norm()>0
        if c['pool']=='masked_segment_mean':assert model.generator.score.weight.count_nonzero()==0
        with np.load(paths['predictions'],allow_pickle=False) as saved:
            assert set(saved.files)=={'source','valid'}
            for role in ('source','valid'):
                batch=make_batch(data[role]);p=predict(model,batch,config['batch_size'])
                assert np.array_equal(p,saved[role]),role+' prediction mismatch'
                for metric_name,value in independent_metric(data[role]['labels'].numpy(),p).items():
                    assert abs(value-report[role][metric_name])<1e-12
                    assert abs(value-chosen[role][metric_name])<1e-12
                checked+=1
        latest=torch.load(paths['latest'],map_location='cpu',weights_only=False)
        assert latest['epoch']==100 and latest['config_sha256']==config_sha
        assert all(k in latest for k in ('optimizer','torch_rng','numpy_rng','python_rng','elapsed_seconds'))
        expected_opt=make_optimizer(model,config,name).state_dict()['param_groups']
        for wanted,actual in zip(expected_opt,latest['optimizer']['param_groups']):
            for field in ('name','lr','weight_decay','params'):assert wanted[field]==actual[field],field+' optimizer mismatch'
        assert len(expected_opt)==len(latest['optimizer']['param_groups'])
        model.load_state_dict(latest['state_dict'])
        if not c['generator_trainable']:assert state_hash(model.generator)==init_generator
        generator_check={'changed':generator_changed,'expected_trainable':c['generator_trainable'],'best_token_delta_rms':token_change,'feedback_boundary_verified':True}
        digests={str(path.relative_to(ROOT)):sha(path) for path in paths.values()}
    except Exception as error:
        errors.append(repr(error));traceback.print_exc()
    result={'condition':name,'seed':seed,'prediction_sets_checked':checked,'errors':errors,'complete':checked==2 and not errors,
            'artifact_sha256':digests,'generator':generator_check,'device':'cpu'}
    atomic_json(RUN/'artifacts'/f'verify_{key}.json',result)
    print(json.dumps(result),flush=True)
    if not result['complete']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True);parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args();verify(args.condition,args.seed)
