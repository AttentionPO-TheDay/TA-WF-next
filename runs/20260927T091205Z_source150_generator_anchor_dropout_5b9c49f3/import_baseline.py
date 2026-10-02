"""Re-score historical A; preserve source provenance and do not retrain."""
import copy
from common import *
from verify_one import independent_metric

def main():
    c=config_read();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    data=load_data(c);entries={'source':data['source150'],'common_source':data['source80'],'valid':data['valid']}
    prior=ROOT/c['prior_run'];oldconfig=json.loads((prior/'config.json').read_text())
    assert oldconfig['seeds']==c['seeds'] and oldconfig['optimizer_steps']==c['optimizer_steps']
    for key in ['batch_size','dropout','weight_decay','lr']:assert oldconfig[key]==c[key]
    for seed in c['seeds']:
        key=f'R00_{seed}';oldkey=f'B_150_l1_{seed}';out=RUN/'artifacts'/key;out.mkdir(exist_ok=False);oldout=prior/'artifacts'/oldkey
        report=json.loads((oldout/'report.json').read_text());oldverify=json.loads((oldout/'verification.json').read_text());assert oldverify['complete']
        history=json.loads((oldout/'history.json').read_text());scored=[h for h in history if h['step'] in c['eval_steps']]
        assert len(scored)==20 and all('valid' in h for h in scored)
        chosen=max(scored,key=lambda h:h['valid']['macro_f1']);assert chosen['block']==report['best_block']
        for h in history:assert h.get('learning_rate',.001)==lr_for_step(c,h['step'])
        # Preserve both source150 and common_source80 exactly.
        chosen=next(h for h in history if h['block']==report['best_block'])
        torch.manual_seed(seed);model=make_model(c,'R00');assert state_hash(model)==report['initial_state_sha256']
        checked=0;paths=[]
        for kind,ref,pfile,scores in [('best',chosen,'predictions.npz',report),('latest',history[-1],'predictions_last.npz',report['last'])]:
            sourcepath=prior/'checkpoints'/f'{oldkey}_{kind}.pt';state=torch.load(sourcepath,map_location='cpu',weights_only=kind=='best')
            assert state['config_sha256']==sha(prior/'config.json') and state['step']==ref['step']
            model.load_state_dict(state['state_dict']);preds={}
            with np.load(oldout/pfile,allow_pickle=False) as saved:
                for role,e in entries.items():
                    p=predict(model,make_batch(e));oldrole=role
                    assert np.array_equal(p,saved[oldrole]);preds[role]=p
                    for k,v in independent_metric(e['labels'].numpy(),p).items():assert abs(v-scores[oldrole][k])<1e-12 and abs(v-ref[role][k])<1e-12
                    checked+=1
            if kind=='latest':
                assert state['step']==state['next_index_stream_row']==12800 and all(int(v['step'])==12800 for v in state['optimizer']['state'].values())
                state['history']=history
            state.update(config_sha256=sha(RUN/'config.json'),historical_source_checkpoint=str(sourcepath.relative_to(ROOT)),historical_source_sha256=sha(sourcepath))
            dest=RUN/'checkpoints'/f'{key}_{kind}.pt';atomic_torch(dest,state);np.savez_compressed(out/pfile,**preds);paths.extend([dest,out/pfile])
        report.update(condition='R00',historical_reuse=True,historical_source_run=c['prior_run'],historical_source_report_sha256=sha(oldout/'report.json'),config_sha256=sha(RUN/'config.json'),shared_initial_sha256=report['initial_state_sha256'])
        report.update(generator_anchor_lambda=0.,classifier_dropout=.1)
        atomic_json(out/'report.json',report);atomic_json(out/'history.json',history)
        ix=torch.load(oldout/'index_stream.pt',weights_only=True);assert torch.equal(ix,index_stream(15300,seed,c));atomic_torch(out/'index_stream.pt',ix)
        atomic_torch(out/'sample_visits.pt',torch.bincount(ix.flatten(),minlength=15300));paths.extend(out/x for x in ['report.json','history.json','index_stream.pt','sample_visits.pt'])
        atomic_json(out/'verification.json',{'complete':True,'historical_reuse':True,'prediction_sets_checked':checked,'selection_verified':True,'lr_verified':True,'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'errors':[]})
        print('BASELINE VERIFIED',key,flush=True)
if __name__=='__main__':main()
