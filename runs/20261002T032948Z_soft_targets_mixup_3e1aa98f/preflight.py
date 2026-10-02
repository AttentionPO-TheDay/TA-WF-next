import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import RUN,ROOT,sha,save,deterministic
from model import MultiViewModel,span_mask,probability_kl,mix_batch,mixed_ce
from worker import predict

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0';deterministic('cuda',2)
 c=json.loads((RUN/'config.json').read_text());old=ROOT/c['historical_run'];references={}
 for p,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
 # Exact input/label pairing, independently replay Beta and permutation draws.
 tam=torch.arange(4*2*6,dtype=torch.float32).reshape(4,2,6);y=torch.tensor([0,1,2,3]);rng=np.random.default_rng(801);ref=np.random.default_rng(801)
 mixed,other,w=mix_batch(tam,y,rng,.2);expected_w=float(ref.beta(.2,.2));perm=torch.tensor(ref.permutation(4))
 assert w==expected_w and torch.equal(other,y[perm]) and torch.equal(mixed,w*tam+(1-w)*tam[perm])
 z=torch.randn(4,5,requires_grad=True);assert torch.allclose(mixed_ce(z,y,other,w),-(w*z.log_softmax(1)[torch.arange(4),y]+(1-w)*z.log_softmax(1)[torch.arange(4),other]).mean())
 assert torch.equal(mixed_ce(z,y,other,1.),F.cross_entropy(z,y))
 logits=torch.tensor([[2.,0.,-1.],[0.,1.,2.]],requires_grad=True);p=torch.tensor([.2,.3,.5],requires_grad=True)
 loss=probability_kl(logits,p,2.);manual=(p.detach()*(p.detach().log()-(logits/2).log_softmax(1))).sum()/len(logits)*4
 assert torch.allclose(loss,manual,atol=1e-6);loss.backward();assert p.grad is None and torch.isfinite(logits.grad).all()
 assert abs(float(probability_kl(torch.zeros(3,102),torch.full((102,),1/102),2.)))<1e-5
 raw=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False);assert set(raw)=={'source','valid'}
 data={role:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()}
 checks=0;marginals={}
 for seed in c['seeds']:
  targetpath=RUN/'artifacts'/f'teacher_source_{seed}.pt';target=torch.load(targetpath,weights_only=True)
  assert sha(targetpath)==sha(old/'artifacts'/targetpath.name)
  assert set(target)=={'source_logits','source_rows','source_labels'}
  assert torch.equal(target['source_rows'],raw['source']['rows']) and torch.equal(target['source_labels'],raw['source']['labels'])
  q=(target['source_logits']/2).softmax(1).mean(0);assert torch.isfinite(q).all() and (q>0).all() and abs(float(q.sum())-1)<1e-6
  marginals[str(seed)]={'min':float(q.min()),'max':float(q.max()),'sum':float(q.sum())}
  for kind in ['ce_only','shuffled_05']:
   d=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((d/'report.json').read_text());torch.manual_seed(seed);m=MultiViewModel(kind).cuda()
   state=torch.load(d/'initial_state.pt',weights_only=True);assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
   ckpath=ROOT/'runs'/rep['origin_run']/'checkpoints'/f"{rep['origin_task']}_best.pt";references[str(ckpath.relative_to(ROOT))]=sha(ckpath)
   m.load_state_dict(torch.load(ckpath,weights_only=True,map_location='cuda')['state_dict'])
   with np.load(d/'predictions_best.npz') as f:
    for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),f[role]);checks+=1
   del m
 # Three co-resident workers worth of forward/backward; all four conditions tested.
 peak=0
 for kind in c['conditions']:
  torch.manual_seed(c['seeds'][0]);m=MultiViewModel(kind).cuda()
  state=torch.load(RUN/'artifacts'/f"ce_only_{c['seeds'][0]}"/'initial_state.pt',weights_only=True)
  assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
  e=data['source'];tam=span_mask(e['tam'][:64],torch.Generator().manual_seed(c['seeds'][0]+50000));y=e['labels'][:64];other=y;w=1.
  spec=c['regularization']['conditions'][kind]
  if spec['mixup']:tam,other,w=mix_batch(tam,y,np.random.default_rng(c['seeds'][0]+80000),.2)
  z=m(e['timestamps'][:64],tam);loss=mixed_ce(z,y,other,w)
  if spec['target']:
   target=torch.load(RUN/'artifacts'/f"teacher_source_{c['seeds'][0]}.pt",weights_only=True)
   q=torch.full((102,),1/102) if spec['target']=='uniform' else (target['source_logits']/2).softmax(1).mean(0)
   loss=loss+.5*probability_kl(z,q.cuda(),2.)
  assert torch.isfinite(loss);loss.backward()
  for name,param in m.generator.named_parameters():
   if param.requires_grad:assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0,name
   else:assert param.grad is None,name
  # Verify inactive timestamp input cannot affect clean evaluation.
  m.eval()
  with torch.no_grad():assert torch.equal(m(e['timestamps'][:2],e['tam'][:2]),m(torch.zeros_like(e['timestamps'][:2]),e['tam'][:2]))
  torch.cuda.synchronize();peak=max(peak,torch.cuda.max_memory_allocated());del m,z,loss
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 data_bytes=sum(v.numel()*v.element_size() for e in data.values() for v in e.values())
 estimate=3*peak+3*1024**3+4*1024**3;total=torch.cuda.mem_get_info()[1];assert estimate<total
 save(RUN/'artifacts/preflight.json',{'passed':True,'historical_reload_checks':checks,'synthetic_mixup_kl_checks_passed':True,'candidate_count':4,'source_examples_per_candidate':64,'optimizer_updates':0,'marginals':marginals,'aggregate_gpu_bytes_with_reserve':estimate,'gpu_total_bytes':total,'historical_reference_hashes':references,'future_access':False})
 print('passed',checks,'historical replays; GPU budget GiB',estimate/1024**3)
if __name__=='__main__':main()
