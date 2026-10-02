import argparse,json
from common import *

def independent_metric(y,p):
    scores=[]
    for label in range(102):
        den=((y==label).sum()+(p==label).sum())
        scores.append(float(2*((y==label)&(p==label)).sum()/den) if den else 0.)
    return {'accuracy':float((y==p).mean()),'macro_f1':float(np.mean(scores))}

def main(name,seed):
    c=config_read();key=f'{name}_{seed}';out=RUN/'artifacts'/key;report=json.loads((out/'report.json').read_text());assert report['status']=='completed' and report['steps_completed']==c['optimizer_steps'] and not report['historical_reuse'];assert report['config_sha256']==sha(RUN/'config.json')
    data=load_data(c);entries={'source':data['source150'],'valid':data['valid']};ix=torch.load(out/'index_stream.pt',weights_only=True);assert torch.equal(ix,index_stream(15300,seed,c));counts=torch.bincount(ix.flatten(),minlength=15300);assert torch.equal(counts,torch.load(out/'sample_visits.pt',weights_only=True)) and int(counts.sum())==819200 and int(counts.max()-counts.min())<=1
    torch.manual_seed(seed);model=make_model(c,name);assert state_hash(model)==report['initial_state_sha256'];initial={k:v.detach().clone() for k,v in model.generator.state_dict().items()}
    history=json.loads((out/'history.json').read_text());assert [h['block'] for h in history]==list(range(1,401)) and [h['step'] for h in history]==list(range(32,12801,32));scored=[h for h in history if 'valid' in h];assert [h['step'] for h in scored]==c['eval_steps'];chosen=max(scored,key=lambda h:h['valid']['macro_f1']);assert chosen['block']==report['best_block']
    for h in history:
        expected=group_lrs(c,name,h['step']);assert h['learning_rates']==expected
    assert all(history[0]['first_gradient_norms']['generator'][k]>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight']) and history[0]['first_gradient_norms']['classifier_norm']>0
    checked=0;paths=[]
    for kind,ref,predfile,scores in [('best',chosen,'predictions.npz',report),('latest',history[-1],'predictions_last.npz',report['last'])]:
        ck=RUN/'checkpoints'/f'{key}_{kind}.pt';state=torch.load(ck,map_location='cpu',weights_only=kind=='best');assert state['step']==ref['step'] and state['config_sha256']==sha(RUN/'config.json');model.load_state_dict(state['state_dict']);assert state_hash(model)!=report['initial_state_sha256'];delta=sum(float((v.detach()-initial[k]).double().square().sum()) for k,v in model.generator.state_dict().items())**.5;assert delta>0 and abs(delta-ref['generator_parameter_delta_l2'])<1e-5
        with np.load(out/predfile,allow_pickle=False) as saved:
            for role,e in entries.items():
                pred=predict(model,make_batch(e));assert np.array_equal(pred,saved[role]);
                for k,v in independent_metric(e['labels'].numpy(),pred).items(): assert abs(v-scores[role][k])<1e-12 and abs(v-ref[role][k])<1e-12
                checked+=1
        paths.extend([ck,out/predfile])
        if kind=='latest':
            assert state['step']==state['next_index_stream_row']==12800 and state['history']==history
            final=group_lrs(c,name,12800)
            for g in state['optimizer']['param_groups']: assert g['lr']==final[g['role']] and g['weight_decay']==c['weight_decay']
    paths += [out/x for x in ['report.json','history.json','index_stream.pt','sample_visits.pt']]
    atomic_json(out/'verification.json',{'complete':True,'prediction_sets_checked':checked,'historical_reuse':False,'lr_verified':True,'selection_verified':True,'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'errors':[]})
    print('VERIFIED',key,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();main(a.condition,a.seed)
