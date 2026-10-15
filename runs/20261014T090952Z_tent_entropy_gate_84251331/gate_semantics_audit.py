"""Supplementary audit only; does not perform WF adaptation or change config."""
import json, hashlib
from pathlib import Path
import torch
from torch import nn
from enhanced_tent import forward_and_adapt
R=Path(__file__).resolve().parent
class LogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=nn.Parameter(torch.tensor([[8.,0.],[2.3,0.],[0.,0.]]))
    def forward(self,x): return self.logits[x]
m=LogitModel(); opt=torch.optim.Adam(m.parameters(),lr=.00025)
_,h,mask=forward_and_adapt(torch.tensor([0,1,2]),m,opt,.2,.4)
assert mask.tolist()==[False,True,False]
assert torch.equal(m.logits.grad[[0,2]],torch.zeros_like(m.logits.grad[[0,2]]))
assert bool((m.logits.grad[1]!=0).any())
before=m.logits.detach().clone(); moments={k:v.clone() if torch.is_tensor(v) else v for k,v in opt.state[m.logits].items()}
_,h,mask=forward_and_adapt(torch.tensor([0,2]),m,opt,.2,.4)
assert not mask.any() and torch.equal(m.logits,before)
assert all(torch.equal(v,opt.state[m.logits][k]) if torch.is_tensor(v) else v==opt.state[m.logits][k] for k,v in moments.items())
scores=json.loads((R/'artifacts/scores.json').read_text()); cells=[]
for row in scores:
    if row['origin']!='new':continue
    updates=json.loads((R/'artifacts'/f"update_{row['seed']}_{row['role']}_{row['condition']}.json").read_text())
    assert len(updates)==800
    for step,x in enumerate(updates):
        assert x['step']==step and x['batch']==128
        assert 0<=x['retained_samples']<=128
        assert x['skipped_samples']==128-x['retained_samples']
        assert x['retained_fraction']==x['retained_samples']/128
        assert x['executed']==(x['retained_samples']>0)
    executed=sum(x['executed'] for x in updates); exposed=sum(x['retained_samples'] for x in updates)
    assert row['executed_updates']==executed and row['skipped_updates']==800-executed
    assert row['skipped_update_count']==800-executed and row['target_backward_trace_exposures']==exposed
    assert row['retained_sample_fraction']==exposed/102400
    cells.append({'seed':row['seed'],'role':row['role'],'retained_samples':exposed,'executed_updates':executed})
for path,digest in json.loads((R/'artifacts/freeze.json').read_text()).items():
    assert hashlib.sha256((R.parents[1]/path).read_bytes()).hexdigest()==digest,path
result={'passed':True,'boundary_interval_nats':[.2,.4],'excluded_sample_gradient_zero':True,'selected_sample_gradient_nonzero':True,'empty_batch_preserves_parameters_and_adam_state':True,'gate_log_summary_cells':len(cells),'freeze_hashes_unchanged':True,'limitation':'stored logs retain counts, not every per-sample entropy/mask; no full per-sample replay claimed','cells':cells}
(R/'artifacts/gate_semantics_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('supplementary gate audit passed')
