"""Synthetic residual intervention checks; data checks do not train or score."""
from common import *
from datetime import datetime,timezone
from torch.nn import functional as F
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views
import copy

def main():
 c=config_read(False);assert c['status']=='draft'
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in c['anchor_sha256'].items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/c['prepared_input'])==c['prepared_sha256'] and sha(ROOT/c['sampling_manifest'])==c['manifest_sha256']
 oldfreeze=json.loads((ROOT/c['prior_run']/'artifacts/freeze.json').read_text())
 for p,h in oldfreeze['code_sha256'].items():
  if p.startswith('src/'):assert sha(ROOT/p)==h,p
 data=load_data(c);manifest=json.loads((ROOT/c['sampling_manifest']).read_text())
 for role,n in [('source150',15300),('source80',8160),('valid',510)]:assert len(data[role]['labels'])==n and torch.all(torch.bincount(data[role]['labels'],minlength=102)==n//102)
 assert data['source150']['rows'].tolist()==manifest['large_rows'] and data['valid']['rows'].tolist()==manifest['valid_rows']
 assert torch.equal(data['source150']['directions'][:8160],data['source80']['directions'])
 assert torch.equal(data['source150']['labels'][:8160],data['source80']['labels'])
 assert not set(manifest['large_rows'])&set(manifest['excluded_full_valid_overlap'])
 d=torch.zeros(3,5000,dtype=torch.int8)
 for i,n in enumerate([84,163,237]):d[i,:n]=torch.where(torch.arange(n)%(i+3)==0,1,-1).to(torch.int8)
 orig=batch_from_views([generate_views(v.numpy(),input_kind='direction',budget=5000,window_sizes=(50,250)) for v in d],packet_patch=50,max_runs=128)
 batch=Batch(orig,d,d!=0);target=torch.tensor([0,1,2]);records=[]
 for seed in c['seeds']:
  torch.manual_seed(seed);base=make_model(c,'A_base');bh=state_hash(base);rng=torch.get_rng_state().clone();base.eval()
  oldrep=json.loads((ROOT/c['prior_run']/'artifacts'/f'B_150_l1_{seed}'/'report.json').read_text());assert bh==oldrep['initial_state_sha256']
  models={}
  for name in ['B_counts','C_ordered']:
   torch.manual_seed(seed);m=make_model(c,name);models[name]=m
   assert shared_state_hash(m)==bh and torch.equal(torch.get_rng_state(),rng)
   assert m.generator.order_residual.weight.count_nonzero()==0
   assert parameter_counts(m)['total']==124778 and parameter_counts(base)['total']==119578
   m.eval();base.eval()
   with torch.inference_mode():
    torch.testing.assert_close(batch.logits(m),batch.logits(base),rtol=0,atol=0)
    torch.testing.assert_close(batch.tokens(m),batch.tokens(base),rtol=0,atol=0)
   # Train-mode shared gradients are unchanged at zero residual.
   m.train();base.train();m.zero_grad(set_to_none=True);base.zero_grad(set_to_none=True)
   torch.set_rng_state(rng);F.cross_entropy(batch.logits(base),target).backward()
   torch.set_rng_state(rng);F.cross_entropy(batch.logits(m),target).backward()
   baseparams=dict(base.named_parameters())
   for k,p in m.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all()
    if k in baseparams:torch.testing.assert_close(p.grad,baseparams[k].grad,rtol=0,atol=0)
   assert m.generator.order_residual.weight.grad.norm()>0
   for k in ['conv1.weight','conv2.weight','score.weight','projection.weight']:assert dict(m.generator.named_parameters())[k].grad.norm()>0
   make_optimizer(m,c,name).step();assert m.generator.order_residual.weight.norm()>0
   m.eval();base.eval()
   with torch.inference_mode():
    output=batch.logits(m)
    torch.testing.assert_close(output,torch.cat([batch.subset(slice(i,i+1)).logits(m) for i in range(3)]),rtol=2e-5,atol=2e-5)
    changed=copy.deepcopy(batch);changed.directions[~changed.observed]=1
    torch.testing.assert_close(output,changed.logits(m),rtol=0,atol=0)
    rt=m.generator.residual_tokens(d,batch.observed,torch.float32);assert rt.square().sum()>0
    assert rt[~batch.observed.reshape(3,100,50).any(-1)].count_nonzero()==0
    clone=make_model(c,name);clone.load_state_dict(m.state_dict());clone.eval();torch.testing.assert_close(output,batch.logits(clone),rtol=0,atol=0)
   records.append({'seed':seed,'condition':name,'shared_initial_sha256':bh,'zero_residual_exact':True,'shared_initial_gradients_exact':True,'residual_and_generator_gradients_nonzero':True,'padding_and_batch_independent':True,'state_reload_exact':True})
  # B/C state before training is identical; mode only changes input computation.
  torch.manual_seed(seed);b=make_model(c,'B_counts');torch.manual_seed(seed);o=make_model(c,'C_ordered');assert state_hash(b)==state_hash(o)
  perm=d.clone();perm[:,:50]=d[:,:50].reshape(3,5,10).flip(-1).reshape(3,50)
  ib=b.generator.residual_input(d,batch.observed,torch.float32);pb=b.generator.residual_input(perm,batch.observed,torch.float32)
  io=o.generator.residual_input(d,batch.observed,torch.float32);po=o.generator.residual_input(perm,batch.observed,torch.float32)
  assert torch.equal(ib,pb) and not torch.equal(io,po)
  assert torch.equal(ib[...,50:],io[...,50:])
  torch.testing.assert_close(ib[...,:50].reshape(3,100,5,10).sum(-1),io[...,:50].reshape(3,100,5,10).sum(-1),rtol=1e-6,atol=1e-6)
  assert torch.equal(index_stream(15300,seed,c),torch.load(ROOT/c['prior_run']/'artifacts'/f'B_150_l1_{seed}'/'index_stream.pt',weights_only=True))
 c['status']='frozen';c['frozen_utc']=datetime.now(timezone.utc).isoformat();atomic_json(RUN/'config.json',c)
 paths=list(RUN.glob('*.py'))+list((ROOT/'src/ta_wf_next').glob('*.py'))+[ROOT/'scripts/experiment.py']
 atomic_json(RUN/'artifacts/freeze.json',{'config_sha256':sha(RUN/'config.json'),'plan_sha256':sha(RUN/'PLAN.md'),'datasets_sha256':sha(ROOT/'configs/datasets.json'),'prepared_sha256':c['prepared_sha256'],'manifest_sha256':c['manifest_sha256'],'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
 atomic_json(RUN/'artifacts/preflight.json',{'complete':True,'checks':records,'counts_permutation_invariant':True,'ordered_permutation_sensitive':True,'per_bin_sum_and_mask_preserved':True,'paired_sample_stream':True,'synthetic_only_training_checks':True,'new_real_training':False,'future_access':False,'errors':[]})
 print('PREFLIGHT PASSED; FROZEN',flush=True)
if __name__=='__main__':main()
