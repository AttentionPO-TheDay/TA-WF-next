from common import *
from datetime import datetime,timezone
from torch.nn import functional as F

def main():
 c=config_read(False);assert c['status']=='draft';torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 data=load_data(c);assert len(data['source150']['labels'])==15300 and len(data['source80']['labels'])==8160
 assert torch.equal(data['source150']['directions'][:8160],data['source80']['directions']) and torch.equal(data['source150']['labels'][:8160],data['source80']['labels'])
 d=torch.zeros(3,5000,dtype=torch.int8);d[0,:80]=1;d[1,:130]=-1;d[2,:200]=torch.where(torch.arange(200)%2==0,1,-1);batch=Batch(data['valid']['original'][:3] if False else make_batch(data['valid']).original,d,d!=0)
 records=[]
 for seed in c['seeds']:
  torch.manual_seed(seed);base=make_model(c,'R00');base_hash=state_hash(base)
  for name in ['R00','R10','R01','R11']:
   torch.manual_seed(seed);m=make_model(c,name);assert state_hash(m)==base_hash
   b=make_batch(data['source150']).subset(slice(0,3));m.train();opt=make_optimizer(m,c,name);initial={k:v.detach().clone() for k,v in m.generator.named_parameters()};opt.zero_grad();loss=F.cross_entropy(b.logits(m),torch.tensor([0,1,2]));pen=anchor_penalty(m,initial,condition_config(c,name)['generator_anchor_lambda']);(loss+pen).backward();
   for p in m.generator.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
   opt.step();records.append({'seed':seed,'condition':name,'anchor':condition_config(c,name)['generator_anchor_lambda'],'dropout':condition_config(c,name)['dropout']})
 c['status']='frozen';c['frozen_utc']=datetime.now(timezone.utc).isoformat();c['prepared_sha256']=sha(ROOT/c['prepared_input']);c['manifest_sha256']=sha(ROOT/c['sampling_manifest'])
 atomic_json(RUN/'config.json',c);paths=list(RUN.glob('*.py'))+list((ROOT/'src/ta_wf_next').glob('*.py'))+[ROOT/'scripts/experiment.py'];freeze={'config_sha256':sha(RUN/'config.json'),'plan_sha256':sha(RUN/'PLAN.md'),'datasets_sha256':sha(ROOT/'configs/datasets.json'),'prepared_sha256':c['prepared_sha256'],'manifest_sha256':c['manifest_sha256'],'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}};atomic_json(RUN/'artifacts/freeze.json',freeze);atomic_json(RUN/'artifacts/preflight.json',{'complete':True,'checks':records,'data_exact':True,'future_access':False});print('PREFLIGHT PASSED; FROZEN')
if __name__=='__main__':main()
