import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import RUN,ROOT,sha,save,deterministic
from model import MultiViewModel,span_mask
from worker import predict
from base_model import parameter_counts

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    c=json.loads((RUN/'config.json').read_text());deterministic('cuda',2);old=ROOT/c['historical_run']
    for path,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/path)==h,path
    assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');assert set(raw)=={'source','valid'}
    data={role:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()}
    references={};checks=0
    for seed in c['seeds']:
        d=RUN/'artifacts'/f'flat_none_{seed}';rep=json.loads((d/'report.json').read_text())
        torch.manual_seed(seed);m=MultiViewModel('flat_none').cuda()
        state=torch.load(d/'initial_state.pt',weights_only=True,map_location='cpu');assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
        ckpath=ROOT/'runs'/rep['origin_run']/'checkpoints'/f"{rep['origin_task']}_best.pt"
        m.load_state_dict(torch.load(ckpath,weights_only=True,map_location='cuda')['state_dict'])
        with np.load(d/'predictions_best.npz') as f:
            for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),f[role]);checks+=1
        references[str(ckpath.relative_to(ROOT))]=sha(ckpath);del m
    models=[];losses=[];params={}
    for kind in c['conditions']:
        torch.manual_seed(c['seeds'][0]);m=MultiViewModel(kind).cuda()
        ref=torch.load(RUN/'artifacts'/f"flat_none_{c['seeds'][0]}"/'initial_state.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(m.state_dict()[k].cpu(),v) for k,v in ref.items())
        e=data['source'];tam=span_mask(e['tam'][:64],torch.Generator().manual_seed(c['seeds'][0]+50000))
        loss=F.cross_entropy(m(e['timestamps'][:64],tam),e['labels'][:64]);assert torch.isfinite(loss)
        models.append(m);losses.append(loss);params[kind]=parameter_counts(m)
    for m,loss in zip(models,losses):
        loss.backward()
        for name,p in m.generator.named_parameters():
            if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
            else:assert p.grad is None,name
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();total=torch.cuda.mem_get_info()[1]
    data_bytes=sum(v.numel()*v.element_size() for e in data.values() for v in e.values())
    estimate=peak+2*data_bytes+3*1024**3+3*256*1024**2+4*1024**3;assert estimate<total
    params['flat_none']={'total':350006,'trainable':322726,'generator_total':3024,'generator_trainable':1472}
    references[str((old/'artifacts/prepared.pt').relative_to(ROOT))]=sha(old/'artifacts/prepared.pt')
    save(RUN/'artifacts/preflight.json',{'passed':True,'baseline_reload_checks':checks,'candidate_count':3,'source_examples_per_candidate':64,'optimizer_updates':0,'parameter_counts':params,'aggregate_gpu_bytes_with_reserve':estimate,'gpu_total_bytes':total,'historical_reference_hashes':references,'future_access':False})
    print('passed; baseline reload',checks,'GPU estimate GiB',estimate/1024**3)

if __name__=='__main__':main()
