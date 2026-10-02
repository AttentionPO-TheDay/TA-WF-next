"""v2: synthetic mechanism checks and prepared-data checks; no real training."""
import copy
from datetime import datetime,timezone
from common import *
from torch.nn import functional as F
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views

def main():
 c=config_read(False);assert c['status']=='draft'
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in c['anchor_sha256'].items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/c['prepared_input'])==c['prepared_sha256'] and sha(ROOT/c['sampling_manifest'])==c['manifest_sha256']
 oldfreeze=json.loads((ROOT/c['prior_run']/'artifacts/freeze.json').read_text())
 for p,h in oldfreeze['code_sha256'].items():
  if p.startswith('src/'):assert sha(ROOT/p)==h,p
 data=load_data(c)
 for role,n in [('source150',15300),('source80',8160),('valid',510)]:
  assert len(data[role]['labels'])==n and torch.all(torch.bincount(data[role]['labels'],minlength=102)==n//102)
 assert torch.equal(data['source150']['directions'][:8160],data['source80']['directions'])
 assert torch.equal(data['source150']['labels'][:8160],data['source80']['labels'])
 manifest=json.loads((ROOT/c['sampling_manifest']).read_text())
 assert data['source150']['rows'].tolist()==manifest['large_rows'] and data['valid']['rows'].tolist()==manifest['valid_rows']
 assert not set(manifest['large_rows'])&set(manifest['excluded_full_valid_overlap'])
 d=torch.zeros(3,5000,dtype=torch.int8)
 for i,n in enumerate([84,163,237]):d[i,:n]=torch.where(torch.arange(n)%(i+3)==0,1,-1).to(torch.int8)
 original=batch_from_views([generate_views(v.numpy(),input_kind='direction',budget=5000,window_sizes=(50,250)) for v in d],packet_patch=50,max_runs=128)
 batch=Batch(original,d,d!=0);target=torch.tensor([0,1,2]);records=[]
 for seed in c['seeds']:
  oldreport=json.loads((ROOT/c['prior_run']/'artifacts'/f'B_150_l1_{seed}'/'report.json').read_text())
  initial_hash=None
  for name in ['R00','R10','R01','R11']:
   cond=condition_config(c,name);torch.manual_seed(seed);m=make_model(c,name)
   assert state_hash(m)==oldreport['initial_state_sha256']
   if initial_hash is None:initial_hash=state_hash(m)
   assert state_hash(m)==initial_hash
   explicit=attention=0
   for module in m.classifier.modules():
    if isinstance(module,torch.nn.Dropout):assert module.p==cond['dropout'];explicit+=1
    if isinstance(module,torch.nn.MultiheadAttention):assert module.dropout==cond['dropout'];attention+=1
   assert explicit>0 and attention==4 and not any(isinstance(x,torch.nn.Dropout) for x in m.generator.modules())
   initial={k:v.detach().clone() for k,v in m.generator.named_parameters()};assert all(not v.requires_grad for v in initial.values())
   assert anchor_penalty(m,initial,cond['generator_anchor_lambda']).item()==0
   # lambda=0 must reproduce CE gradient and update exactly with identical RNG.
   clone=copy.deepcopy(m);o=make_optimizer(m,c,name);oc=make_optimizer(clone,c,name);rng=torch.get_rng_state().clone()
   m.train();F.cross_entropy(batch.logits(m),target).backward()
   torch.set_rng_state(rng);clone.train();(F.cross_entropy(batch.logits(clone),target)+anchor_penalty(clone,initial,0.)).backward()
   for p,q in zip(m.parameters(),clone.parameters()):assert p.grad is not None and torch.equal(p.grad,q.grad)
   o.step();oc.step();assert state_hash(m)==state_hash(clone)
   # At a displaced point isolate anchor gradient; classifier must receive none.
   m.zero_grad(set_to_none=True);pen=anchor_penalty(m,initial,.001);assert pen.item()>0;pen.backward()
   for k,p in m.generator.named_parameters():torch.testing.assert_close(p.grad,.001*(p.detach()-initial[k]),rtol=1e-6,atol=1e-9)
   assert all(p.grad is None for p in m.classifier.parameters())
   before=float(pen.detach());torch.optim.SGD(m.generator.parameters(),lr=1.).step()
   assert float(anchor_penalty(m,initial,.001).detach())<before
   # Verify CE+anchor additive gradients in train mode, fixing RNG.
   m.zero_grad(set_to_none=True);rng=torch.get_rng_state().clone();F.cross_entropy(batch.logits(m),target).backward()
   cegrads={k:p.grad.detach().clone() for k,p in m.named_parameters()};m.zero_grad(set_to_none=True);torch.set_rng_state(rng)
   (F.cross_entropy(batch.logits(m),target)+anchor_penalty(m,initial,cond['generator_anchor_lambda'])).backward()
   for k,p in m.named_parameters():
    expected=cegrads[k]
    if k.startswith('generator.'):expected=expected+cond['generator_anchor_lambda']*(p.detach()-initial[k[len('generator.'):]])
    torch.testing.assert_close(p.grad,expected,rtol=2e-5,atol=1e-7)
   assert all(torch.isfinite(p.grad).all() for p in m.parameters())
   assert all(cegrads['generator.'+k].norm()>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight'])
   m.eval();clone.load_state_dict(m.state_dict());clone.eval()
   with torch.inference_mode():torch.testing.assert_close(batch.logits(m),batch.logits(clone),rtol=0,atol=0)
   records.append({'seed':seed,'condition':name,'initial_sha256':initial_hash,'dropout_modules':explicit,'attention_modules':attention,'zero_lambda_exact':True,'anchor_gradient_isolated':True,'combined_gradient_additive':True})
  ix=index_stream(15300,seed,c);oldix=torch.load(ROOT/c['prior_run']/'artifacts'/f'B_150_l1_{seed}'/'index_stream.pt',weights_only=True)
  assert torch.equal(ix,oldix)
 assert [lr_for_step(c,s) for s in [1,6400,6401,9600,9601,12800]]==[.001,.001,.0003,.0003,.0001,.0001]
 c['status']='frozen';c['frozen_utc']=datetime.now(timezone.utc).isoformat();atomic_json(RUN/'config.json',c)
 paths=list(RUN.glob('*.py'))+list((ROOT/'src/ta_wf_next').glob('*.py'))+[ROOT/'scripts/experiment.py']
 freeze={'config_sha256':sha(RUN/'config.json'),'plan_sha256':sha(RUN/'PLAN.md'),'datasets_sha256':sha(ROOT/'configs/datasets.json'),'prepared_sha256':c['prepared_sha256'],'manifest_sha256':c['manifest_sha256'],'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
 atomic_json(RUN/'artifacts/freeze.json',freeze)
 atomic_json(RUN/'artifacts/preflight.json',{'version':2,'complete':True,'checks':records,'data_manifest_hash_verified':True,'source_valid_unchanged':True,'sample_stream_paired':True,'synthetic_only_v2':True,'prior_smoke':'artifacts/preflight_v1_superseded/REVISION.md','future_access':False,'errors':[]})
 print('PREFLIGHT v2 PASSED; FROZEN',flush=True)
if __name__=='__main__':main()
