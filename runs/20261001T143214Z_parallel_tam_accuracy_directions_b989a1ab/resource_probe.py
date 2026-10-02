"""Aggregate GPU0 memory for three candidates; no optimizer update or valid scoring."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,json
from torch.nn import functional as F
from model import MultiViewModel
from common import RUN,save,deterministic

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
 deterministic('cuda',2)
 raw=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
 datasets=[];models=[];losses=[]
 for kind in ['scale_concat','attention_readout','supcon']:
  data={role:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()}
  model=MultiViewModel(kind).cuda();e=data['source']
  logits=model(e['timestamps'][:64],e['tam'][:64]);loss=F.cross_entropy(logits,e['labels'][:64])
  datasets.append(data);models.append(model);losses.append(loss)
 for loss in losses:loss.backward()
 torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
 peak=torch.cuda.max_memory_allocated()
 # Combined graph peak + three independent CUDA-process context allowances,
 # optimizer state/caching slack and 4GiB safety reserve. Conservative estimate.
 required=peak+3*(1024**3)+3*(256*1024**2)+4*(1024**3)
 assert required<total,'three workers exceed memory budget estimate'
 out={'passed':True,'physical_gpu':0,'models':3,'full_source_valid_copies':3,'peak_combined_allocated_bytes':peak,'total_gpu_bytes':total,'remaining_bytes_at_probe':free,'required_with_context_optimizer_and_reserve_bytes':required,'max_gpu_parallel':3,'optimizer_updates':0,'future_access':False}
 save(RUN/'artifacts/resource_probe.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
