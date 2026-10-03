import json, os
from pathlib import Path
import torch
from model import StatsModel, global_stats

RUN=Path(__file__).resolve().parent

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    c=json.loads((RUN/'config.json').read_text()); assert c['status']=='frozen'
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu')
    source=raw['source']; valid=raw['valid']; assert source['tam'].shape==(15300,2,1800) and valid['tam'].shape==(510,2,1800)
    stats=global_stats(source['tam']); assert stats.shape==(15300,180); mean=stats.mean(0); std=stats.std(0,unbiased=False).clamp_min(1e-6)
    assert torch.isfinite(stats).all() and torch.isfinite(mean).all() and torch.all(std>0)
    states={}; counts={}
    for kind in c['conditions']:
        torch.manual_seed(21729); m=StatsModel(kind,mean,std).cuda(); states[kind]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; counts[kind]=sum(p.numel() for p in m.parameters())
        x=source['timestamps'][:4].cuda(); t=source['tam'][:4].cuda();
        with torch.no_grad():
            z=m(x,t); assert z.shape==(4,102) and torch.isfinite(z).all()
            assert torch.equal(z,m(x,torch.zeros_like(t))) is False or kind.startswith('zero_')
        loss=torch.nn.functional.cross_entropy(m(x,t),source['labels'][:4].cuda()); loss.backward(); assert torch.isfinite(loss)
        for name,p in m.named_parameters():
            if p.requires_grad and name.startswith(('generator.', 'classifier.')):
                if p.grad is None: raise AssertionError(name)
        del m
    ref=torch.load(RUN/'artifacts/baseline_21729/initial_state.pt',weights_only=True)
    for kind,state in states.items():
        for k,v in ref.items(): assert torch.equal(v,state[k]),(kind,k)
    out={'passed':True,'source_rows':len(source['labels']),'valid_rows':len(valid['labels']),'stats_dim':180,'conditions':c['conditions'],'parameter_counts':counts,'shared_wide_bn_initialization':True,'future_access':False}
    (RUN/'artifacts/preflight.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
