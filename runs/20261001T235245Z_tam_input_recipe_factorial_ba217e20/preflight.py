import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import RUN,ROOT,save,sha,deterministic
from model import MultiViewModel,span_mask
from worker import predict
from base_model import parameter_counts

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    c=json.loads((RUN/'config.json').read_text());deterministic('cuda',2)
    old=ROOT/c['historical_run'];rf=ROOT/c['audit_inputs']['rf_run']
    for directory in [old,rf]:
        for path,h in json.loads((directory/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/path)==h,path
    assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
    native=torch.load(rf/'artifacts/native_prepared.pt',weights_only=False,map_location='cpu')
    assert set(raw)==set(native)=={'source','valid'}
    for role in raw:
        assert torch.equal(raw[role]['tam'],native[role]['tam'][:,0])
        assert torch.equal(raw[role]['rows'],native[role]['rows']) and torch.equal(raw[role]['labels'],native[role]['labels'])
    del native
    data={role:{k:v.cuda() for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()}
    checks=0;references={}
    for seed in c['seeds']:
        out=RUN/'artifacts'/f'log_current_{seed}';rep=json.loads((out/'report.json').read_text())
        torch.manual_seed(seed);m=MultiViewModel('log_current').cuda()
        initial=torch.load(out/'initial_state.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items())
        ckpath=old/'checkpoints'/f'span_mask_{seed}_best.pt'
        m.load_state_dict(torch.load(ckpath,weights_only=True,map_location='cuda')['state_dict'])
        with np.load(out/'predictions_best.npz') as f:
            for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),f[role]);checks+=1
        references[str(ckpath.relative_to(ROOT))]=sha(ckpath);del m
    models=[];losses=[];params={}
    for kind in c['conditions']:
        torch.manual_seed(c['seeds'][0]);m=MultiViewModel(kind).cuda()
        ref=torch.load(RUN/'artifacts'/f"log_current_{c['seeds'][0]}"/'initial_state.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(v.cpu(),ref[k]) for k,v in m.state_dict().items())
        e=data['source'];masked=span_mask(e['tam'][:64],torch.Generator().manual_seed(c['seeds'][0]+50000))
        loss=F.cross_entropy(m(e['timestamps'][:64],masked),e['labels'][:64]);assert torch.isfinite(loss)
        models.append(m);losses.append(loss);params[kind]=parameter_counts(m)
    for m,loss in zip(models,losses):
        loss.backward()
        for name,p in m.generator.named_parameters():
            if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
            else:assert p.grad is None,name
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();total=torch.cuda.mem_get_info()[1]
    data_bytes=sum(v.numel()*v.element_size() for e in data.values() for v in e.values())
    estimate=peak+2*data_bytes+3*1024**3+3*256*1024**2+4*1024**3;assert estimate<total
    params['log_current']=params['raw_current']
    references[str((rf/'artifacts/native_prepared.pt').relative_to(ROOT))]=sha(rf/'artifacts/native_prepared.pt')
    for p in [rf/'rf_native.py',rf/'worker.py',rf/'config.json',old/'artifacts/prepared.pt']:references[str(p.relative_to(ROOT))]=sha(p)
    save(RUN/'artifacts/preflight.json',{'passed':True,'baseline_reload_checks':checks,'source_samples_per_candidate':64,'candidate_count':3,'optimizer_updates':0,'source_valid_cache_parity':'TAM, labels, rows exactly equal native RF cache','parameter_counts':params,'aggregate_gpu_bytes_with_reserve':estimate,'gpu_total_bytes':total,'historical_reference_hashes':references,'future_access':False})
    print('passed; baseline reload',checks,'GPU aggregate GiB',estimate/1024**3)

if __name__=='__main__':main()
