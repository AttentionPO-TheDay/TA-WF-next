"""Independent nonlinear readout fitting on fixed generator outputs; no network transfer."""
import argparse
import time
from common import *
from torch.nn import functional as F

def main(cond,seed):
 c=config_read();assert cond in c['conditions'] and seed in c['seeds']
 torch.set_num_threads(c['resource']['threads_per_worker']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();key=f'{cond}_{seed}';out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
 data=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False);assert set(data)=={'source','valid'}
 anchor=ROOT/c['prior_run']/'checkpoints'/f'{key}_best.pt'
 old=GeneratorClassifier(pool='attention',generator_trainable=cond!='C_local_attention_frozen').eval()
 old.load_state_dict(torch.load(anchor,map_location='cpu',weights_only=True)['state_dict'])
 gen_before=state_hash(old.generator)
 old_saved=np.load(ROOT/c['prior_run']/'artifacts'/f'predictions_{key}.npz')
 prior_features=np.load(ROOT/c['diagnostic_run']/'artifacts'/key/'features.npz')
 packets={};replay={}
 for role in c['roles']:
  entry=data[role];base=GeneratorTokenBatch(**entry['original']);ts=[];ps=[]
  with torch.inference_mode():
   for i in range(0,len(entry['labels']),64):
    b=subset(base,slice(i,i+64));d=entry['directions'][i:i+64]
    t=old.generator(d,d!=0,b.packet);ts.append(t.numpy().copy())
    ps.append(old.classifier(replace(b,packet=t)).argmax(1).numpy())
  packets[role]=torch.from_numpy(np.concatenate(ts));pred=np.concatenate(ps)
  assert np.array_equal(pred,old_saved[role]),'original replay mismatch'
  expected=prior_features[f'{role}_packet_tokens'][:,:5200].reshape(-1,100,52)
  assert np.array_equal(packets[role].numpy(),expected),'previous token cache mismatch'
  replay[role]=metric(entry['labels'].numpy(),pred)
 old_saved.close();prior_features.close()
 assert gen_before==state_hash(old.generator) and all(p.grad is None for p in old.generator.parameters())
 save(out/'tokens.pt',packets);cache_sha=sha(out/'tokens.pt')
 batches=cache_load(out/'tokens.pt',data)
 y=data['source']['labels'];vy=data['valid']['labels'].numpy()
 model=fresh(seed+c['classifier_seed_offset']);initial=state_hash(model)
 assert initial!=state_hash(old.classifier),'must not reuse original classifier'
 save(out/'initial_classifier.pt',model.state_dict())
 optimizer=torch.optim.AdamW(model.parameters(),lr=c['classifier_lr'],weight_decay=c['classifier_weight_decay'])
 # Same random stream and batch order in each paired B/C run, regardless of cache work.
 torch.manual_seed(seed+c['classifier_seed_offset']+1)
 history=[];best=-1.;best_epoch=None
 print(json.dumps({'key':key,'pid':os.getpid(),'cache_sha256':cache_sha,'original_replay':replay,'initial_classifier_sha256':initial,'classifier_parameters':sum(p.numel() for p in model.parameters())}),flush=True)
 for epoch in range(1,c['epochs']+1):
  if time.monotonic()-started>c['resource']['job_seconds']:raise TimeoutError('job budget')
  model.train();loss_sum=0.;correct=0
  order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+c['classifier_seed_offset']+1000+epoch))
  for start in range(0,len(y),c['batch_size']):
   idx=order[start:start+c['batch_size']];optimizer.zero_grad(set_to_none=True)
   logits=model(subset(batches['source'],idx));loss=F.cross_entropy(logits,y[idx]);assert torch.isfinite(loss)
   loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   optimizer.step();loss_sum+=float(loss.detach())*len(idx);correct+=int((logits.argmax(1)==y[idx]).sum())
  row={'epoch':epoch,'train_loss':loss_sum/len(y),'online_train_accuracy':correct/len(y)}
  if epoch%c['eval_every']==0:
   sp=predict(model,batches['source']);vp=predict(model,batches['valid'])
   row.update(source=metric(y.numpy(),sp),valid=metric(vy,vp))
   if row['valid']['macro_f1']>best:
    best=row['valid']['macro_f1'];best_epoch=epoch
    save(RUN/'checkpoints'/f'{key}_best.pt',{'state_dict':model.state_dict(),'epoch':epoch,'initial_sha256':initial,'config_sha256':sha(RUN/'config.json'),'generator_sha256':gen_before,'cache_sha256':cache_sha})
  row['elapsed_seconds']=time.monotonic()-started;history.append(row)
  save(RUN/'checkpoints'/f'{key}_latest.pt',{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'epoch':epoch,'history':history,'config_sha256':sha(RUN/'config.json')})
  write_json(out/'history.json',history)
  write_json(out/'progress.json',{'status':'running','epoch':epoch,'best_epoch':best_epoch,'best_valid_f1':best,'elapsed_seconds':row['elapsed_seconds']})
  if epoch%c['eval_every']==0:print(json.dumps(row),flush=True)
 assert gen_before==state_hash(old.generator) and all(p.grad is None for p in old.generator.parameters())
 assert sha(out/'tokens.pt')==cache_sha and state_hash(model)!=initial
 model.load_state_dict(torch.load(RUN/'checkpoints'/f'{key}_best.pt',map_location='cpu',weights_only=True)['state_dict'])
 predictions={role:predict(model,batches[role]) for role in c['roles']}
 np.savez_compressed(out/'predictions.npz',**predictions)
 report={'status':'completed','condition':cond,'seed':seed,'best_epoch':best_epoch,'epochs_completed':len(history),'initial_classifier_sha256':initial,'generator_sha256':gen_before,'cache_sha256':cache_sha,'config_sha256':sha(RUN/'config.json'),'original_replay':replay,'generator_unchanged':True,'elapsed_seconds':time.monotonic()-started}
 for role in c['roles']:report[role]=metric(data[role]['labels'].numpy(),predictions[role])
 write_json(out/'report.json',report);write_json(out/'progress.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.condition,a.seed)
