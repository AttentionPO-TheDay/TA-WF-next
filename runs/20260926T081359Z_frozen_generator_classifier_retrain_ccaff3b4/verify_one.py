"""Separate-process best-checkpoint and selection audit."""
import argparse
from common import *

def main(cond,seed):
 c=config_read();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 key=f'{cond}_{seed}';out=RUN/'artifacts'/key
 report=json.loads((out/'report.json').read_text());assert report['status']=='completed'
 assert report['config_sha256']==sha(RUN/'config.json') and sha(out/'tokens.pt')==report['cache_sha256']
 data=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False);batches=cache_load(out/'tokens.pt',data)
 model=fresh(seed+c['classifier_seed_offset']);assert state_hash(model)==report['initial_classifier_sha256']
 initial=torch.load(out/'initial_classifier.pt',map_location='cpu',weights_only=True)
 assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
 best=torch.load(RUN/'checkpoints'/f'{key}_best.pt',map_location='cpu',weights_only=True)
 model.load_state_dict(best['state_dict']);assert state_hash(model)!=report['initial_classifier_sha256']
 assert best['generator_sha256']==report['generator_sha256'] and best['cache_sha256']==report['cache_sha256']
 history=json.loads((out/'history.json').read_text());assert [r['epoch'] for r in history]==list(range(1,101))
 evaluated=[r for r in history if 'valid' in r];assert [r['epoch'] for r in evaluated]==list(range(5,101,5))
 selected=max(evaluated,key=lambda r:r['valid']['macro_f1']);assert selected['epoch']==best['epoch']==report['best_epoch']
 saved=np.load(out/'predictions.npz');checks=0
 for role in c['roles']:
  p=predict(model,batches[role]);assert np.array_equal(p,saved[role])
  # Independent direct per-class calculation rather than reuse training metric implementation.
  y=data[role]['labels'].numpy();f1=[]
  for label in range(102):
   tp=np.sum((p==label)&(y==label));den=np.sum(p==label)+np.sum(y==label);f1.append(2*tp/den if den else 0.)
  m={'accuracy':float(np.mean(p==y)),'macro_f1':float(np.mean(f1))}
  for k,v in m.items():assert abs(v-report[role][k])<1e-12 and abs(v-selected[role][k])<1e-12
  checks+=1
 saved.close()
 old=GeneratorClassifier(pool='attention',generator_trainable=cond!='C_local_attention_frozen')
 old.load_state_dict(torch.load(ROOT/c['prior_run']/'checkpoints'/f'{key}_best.pt',map_location='cpu',weights_only=True)['state_dict'])
 assert state_hash(old.generator)==report['generator_sha256']
 write_json(out/'verification.json',{'complete':True,'prediction_sets_checked':checks,'selection_verified':True,'initialization_verified':True,'frozen_generator_anchor_verified':True,'cache_verified':True,'errors':[]})
 print('VERIFIED',key,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.condition,a.seed)
