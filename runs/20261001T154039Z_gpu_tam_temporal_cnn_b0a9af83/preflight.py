import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import RUN,ROOT,sha,save,deterministic
from model import MultiViewModel
from worker import predict
from base_model import parameter_counts

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    c=json.loads((RUN/'config.json').read_text());deterministic('cuda',2)
    old=ROOT/c['historical_run']
    for path,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/path)==h,path
    assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
    data={role:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()}
    references={};checks=0
    for seed in c['seeds']:
        p=RUN/'artifacts'/f'baseline_{seed}';rep=json.loads((p/'report.json').read_text())
        torch.manual_seed(seed);m=MultiViewModel('baseline').cuda()
        state=torch.load(p/'initial_state.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
        ckpath=ROOT/'runs'/rep['origin_run']/'checkpoints'/f"{rep['origin_task']}_best.pt"
        m.load_state_dict(torch.load(ckpath,weights_only=True,map_location='cuda')['state_dict'])
        with np.load(p/'predictions_best.npz') as f:
            for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),f[role]);checks+=1
        references[str(ckpath.relative_to(ROOT))]=sha(ckpath);del m
    models=[];losses=[]
    for seed in c['seeds']:
        torch.manual_seed(seed);m=MultiViewModel('temporal_cnn').cuda()
        ref=torch.load(RUN/'artifacts'/f'baseline_{seed}'/'initial_state.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(v.cpu(),ref[k]) for k,v in m.state_dict().items() if not k.startswith('head.'))
        e=data['source'];z=m(e['timestamps'][:64],e['tam'][:64]);loss=F.cross_entropy(z,e['labels'][:64]);assert torch.isfinite(loss)
        models.append(m);losses.append(loss)
    for m,loss in zip(models,losses):
        loss.backward()
        for name,p in m.generator.named_parameters():
            if p.requires_grad:assert p.grad is not None and p.grad.norm()>0 and torch.isfinite(p.grad).all(),name
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();total=torch.cuda.mem_get_info()[1]
    # Probe owns one dataset; production owns three. Add two exact data copies,
    # three contexts, state/cache allowances and the preceding 4GiB reserve.
    data_bytes=sum(v.numel()*v.element_size() for e in data.values() for v in e.values())
    estimate=peak+2*data_bytes+3*1024**3+3*256*1024**2+4*1024**3
    assert estimate<total
    save(RUN/'artifacts/preflight.json',{'passed':True,'baseline_reload_checks':checks,'parameters':parameter_counts(models[0]),'cnn_models':3,'source_backward_examples_per_model':64,'optimizer_updates':0,'aggregate_memory_estimate_bytes':estimate,'gpu_total_bytes':total,'frozen_historical_checkpoints':references,'future_access':False})
    print('GPU preflight passed; baseline reload checks',checks,'three-worker estimated GiB',estimate/1024**3)

if __name__=='__main__':main()
