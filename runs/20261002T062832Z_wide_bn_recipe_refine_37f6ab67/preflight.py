import os,json,gc,torch,numpy as np
from torch.nn import functional as F
from common import RUN,ROOT,sha,save,deterministic
from model import MultiViewModel,span_mask
from structure_base import MultiViewModel as StructureModel
from worker import predict

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0';deterministic('cuda',2);c=json.loads((RUN/'config.json').read_text());old=ROOT/c['historical_run']
 for p,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
 assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
 raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');data={k:{x:v.cuda() for x,v in e.items() if x in ['timestamps','tam','labels']} for k,e in raw.items()};checks=0
 for seed in c['seeds']:
  d=RUN/'artifacts'/f'baseline_{seed}';rep=json.loads((d/'report.json').read_text());torch.manual_seed(seed);m=StructureModel('wide_bn').cuda();m.load_state_dict(torch.load(ROOT/'runs'/rep['origin_run']/'checkpoints'/f"{rep['origin_task']}_best.pt",weights_only=True,map_location='cuda')['state_dict'])
  with np.load(d/'predictions_best.npz') as f:
   for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),f[role]);checks+=1
  del m
 params={};peak=0;seed=c['seeds'][0];e=data['source']
 for kind in c['conditions']:
  torch.manual_seed(seed);m=MultiViewModel(kind).cuda();params[kind]={'total':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad)}
  m.train();tam=span_mask(e['tam'][:64],torch.Generator().manual_seed(seed+50000));loss=F.cross_entropy(m(e['timestamps'][:64],tam),e['labels'][:64]);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
  m.eval();before={k:v.clone() for k,v in m.named_buffers()};predict(m,e,64);assert all(torch.equal(v,dict(m.named_buffers())[k]) for k,v in before.items());torch.cuda.synchronize();peak=max(peak,torch.cuda.max_memory_allocated());del m;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 total=torch.cuda.mem_get_info()[1];estimate=3*peak+7*1024**3;assert estimate<total
 save(RUN/'artifacts/preflight.json',{'passed':True,'baseline_replays':checks,'source_backward_examples_per_condition':64,'bn_eval_buffers_frozen':True,'parameter_counts':params,'aggregate_gpu_bytes_with_reserve':estimate,'gpu_total_bytes':total,'future_access':False});print('passed',checks,'budget GiB',estimate/1024**3)
if __name__=='__main__':main()
