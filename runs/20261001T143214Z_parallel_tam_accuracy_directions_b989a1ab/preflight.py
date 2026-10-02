"""Real source forward/backward and baseline reproduction, no optimizer updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,time,shutil
import numpy as np
import torch
from torch.nn import functional as F
from model import MultiViewModel,supervised_contrastive,span_mask
from common import RUN,ROOT,save,sha,deterministic
from worker import predict
from base_model import parameter_counts

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    c=json.loads((RUN/'config.json').read_text());deterministic(a.device,c['cpu_threads'] if a.device=='cpu' else c['gpu_threads'])
    raw=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
    relative=torch.load(RUN/'artifacts/relative.pt',map_location='cpu',weights_only=False)
    examples={k:v[:64].to(a.device) for k,v in raw['source'].items() if k in ['timestamps','tam','labels']}
    results={}
    for kind in c['conditions']:
        torch.manual_seed(c['seeds'][0]);m=MultiViewModel(kind).to(a.device);m.train()
        tam=relative['source']['tam'][:64].to(a.device) if kind=='relative_time' else examples['tam']
        if kind=='span_mask':tam=span_mask(tam,torch.Generator().manual_seed(7))
        start=time.monotonic();z,features=m(examples['timestamps'],tam,return_features=True)
        loss=F.cross_entropy(z,examples['labels'])
        if kind=='supcon':loss=loss+.05*supervised_contrastive(features,examples['labels'])
        loss.backward();assert torch.isfinite(loss)
        for name,param in m.generator.named_parameters():
            if param.requires_grad:assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0,name
            else:assert param.grad is None,name
        if kind=='attention_readout':assert m.readout_query.grad.norm()>0
        if a.device=='cuda':torch.cuda.synchronize()
        results[kind]={**parameter_counts(m),'forward_backward_seconds':time.monotonic()-start,'source_loss_diagnostic':float(loss.detach()),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0}
        del m
        if a.device=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    audit={'device':a.device,'source_examples':64,'optimizer_updates':0,'future_access':False,'candidates':results}
    if a.device=='cuda':
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        data={role:{k:v.to(a.device) for k,v in e.items() if k in ['timestamps','tam','labels']} for role,e in raw.items()}
        baseline_reports=[];checks=0;frozen_inputs={}
        for seed in c['seeds']:
            old=ROOT/c['historical_run']/'artifacts'/f'd2_w16_{seed}'
            report=json.loads((old/'report.json').read_text());origin=ROOT/'runs'/report['origin_run']
            ckpath=origin/'checkpoints'/f"{report['origin_task']}_best.pt"
            torch.manual_seed(seed);m=MultiViewModel('baseline').to(a.device)
            initial=torch.load(old/'initial_state.pt',weights_only=True,map_location='cpu')
            assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items())
            m.load_state_dict(torch.load(ckpath,weights_only=True,map_location=a.device)['state_dict'])
            with np.load(old/'predictions_best.npz') as pred:
                for role,e in data.items():assert np.array_equal(predict(m,e,64).argmax(1),pred[role]);checks+=1
            target=RUN/'artifacts'/f'baseline_{seed}';target.mkdir(exist_ok=False)
            for name in ['initial_state.pt','index_stream.pt','predictions_best.npz','predictions_last.npz','logits_best.npz']:
                shutil.copyfile(old/name,target/name)
                for path in [old/name,target/name]:frozen_inputs[str(path.relative_to(ROOT))]=sha(path)
            history_origin=origin/'artifacts'/report['origin_task']/'history.json'
            shutil.copyfile(history_origin,target/'history.json')
            for path in [ckpath,old/'report.json',history_origin,target/'history.json']:frozen_inputs[str(path.relative_to(ROOT))]=sha(path)
            report.update(task=f'baseline_{seed}',kind='baseline',historical_reuse=True,device='cuda',new_code_reload_verified=True)
            save(target/'report.json',report);frozen_inputs[str((target/'report.json').relative_to(ROOT))]=sha(target/'report.json')
            baseline_reports.append(report);del m
        audit.update(historical_baseline_reports=baseline_reports,baseline_prediction_reload_checks=checks,frozen_historical_inputs=frozen_inputs)
    save(RUN/'artifacts'/f'preflight_{a.device}.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['historical_baseline_reports','frozen_historical_inputs']},indent=2),flush=True)

if __name__=='__main__':main()
