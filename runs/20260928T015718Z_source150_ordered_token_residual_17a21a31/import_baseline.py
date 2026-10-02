"""Reuse previously verified replay evidence; independently recalculate saved metrics."""
import copy,shutil
from common import *
from verify_one import independent_metric

def main():
 c=config_read();torch.set_num_threads(2);data=load_data(c);prior=ROOT/c['prior_run']
 entries={'source':data['source150'],'common_source':data['source80'],'valid':data['valid']}
 oc=json.loads((prior/'config.json').read_text())
 for k in ['seeds','optimizer_steps','batch_size','dropout','weight_decay','lr','lr_schedule','eval_steps']:assert c[k]==oc[k]
 for seed in c['seeds']:
  key=f'A_base_{seed}';pk=f'B_150_l1_{seed}';src=prior/'artifacts'/pk;out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
  pv=json.loads((src/'verification.json').read_text());assert pv['complete'] and pv['prediction_sets_checked']==6
  for p,h in pv['artifact_sha256'].items():assert sha(ROOT/p)==h
  report=json.loads((src/'report.json').read_text());history=json.loads((src/'history.json').read_text())
  scored=[h for h in history if 'valid' in h];assert [h['step'] for h in scored]==c['eval_steps']
  chosen=max(scored,key=lambda h:h['valid']['macro_f1']);assert chosen['block']==report['best_block']
  torch.manual_seed(seed);m=make_model(c,'A_base');assert state_hash(m)==report['initial_state_sha256']
  checked=0;paths=[]
  for kind,pfile,ref,scores in [('best','predictions.npz',chosen,report),('latest','predictions_last.npz',history[-1],report['last'])]:
   sourcepath=prior/'checkpoints'/f'{pk}_{kind}.pt';state=torch.load(sourcepath,map_location='cpu',weights_only=kind=='best')
   m.load_state_dict(state['state_dict']);assert state['step']==ref['step'] and state['config_sha256']==sha(prior/'config.json')
   with np.load(src/pfile,allow_pickle=False) as preds:
    for role,e in entries.items():
     for k,v in independent_metric(e['labels'].numpy(),preds[role]).items():assert abs(v-scores[role][k])<1e-12 and abs(v-ref[role][k])<1e-12
     checked+=1
   state.update(config_sha256=sha(RUN/'config.json'),historical_source_checkpoint=str(sourcepath.relative_to(ROOT)),historical_source_sha256=sha(sourcepath))
   dest=RUN/'checkpoints'/f'{key}_{kind}.pt';atomic_torch(dest,state);shutil.copy2(src/pfile,out/pfile);paths.extend([dest,out/pfile])
  report.update(condition='A_base',historical_reuse=True,residual_mode='none',historical_source_run=c['prior_run'],config_sha256=sha(RUN/'config.json'))
  atomic_json(out/'report.json',report);atomic_json(out/'history.json',history)
  for file in ['index_stream.pt','sample_visits.pt']:shutil.copy2(src/file,out/file)
  ix=torch.load(out/'index_stream.pt',weights_only=True);assert torch.equal(ix,index_stream(15300,seed,c))
  paths.extend(out/x for x in ['report.json','history.json','index_stream.pt','sample_visits.pt'])
  atomic_json(out/'verification.json',{'complete':True,'historical_reuse':True,'prediction_sets_checked':checked,'new_checkpoint_replays':0,'previous_replay_evidence':str((src/'verification.json').relative_to(ROOT)),'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'errors':[]})
  print('HISTORICAL METRICS VERIFIED',key,flush=True)
if __name__=='__main__':main()
