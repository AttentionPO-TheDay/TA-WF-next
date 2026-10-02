"""Synthetic deterministic restart check; no real labels or model selection."""
import json
import random
import runpy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import train
from ta_wf_next.hierarchical_transformer import HierarchicalViewTransformer
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    checks=[]
    for file in ['tests/test_packet_patches.py','tests/test_hierarchical_transformer.py']:
        for name,func in runpy.run_path(str(ROOT/file)).items():
            if name.startswith('test_'):
                func();checks.append(file+':'+name)
    example=runpy.run_path(str(ROOT/'tests/test_hierarchical_transformer.py'))['make_batch']()
    torch.manual_seed(919);np.random.seed(919);random.seed(919)
    model=HierarchicalViewTransformer(max_lengths=(3,2,2));optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def step(m,o):
        m.train();o.zero_grad(set_to_none=True);loss=F.cross_entropy(m(example),torch.tensor([5]));loss.backward();o.step()
    step(model,optimizer)
    state_path=RUN/'artifacts/synthetic_restart.pt'
    train.atomic_torch(state_path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                       'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
    step(model,optimizer);expected=train.state_hash(model)
    expected_np=np.random.rand();expected_py=random.random()
    restored=HierarchicalViewTransformer(max_lengths=(3,2,2));restored_opt=torch.optim.AdamW(restored.parameters(),lr=.001)
    saved=torch.load(state_path,map_location='cpu',weights_only=False)
    restored.load_state_dict(saved['model']);restored_opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    step(restored,restored_opt)
    assert train.state_hash(restored)==expected
    assert np.random.rand()==expected_np and random.random()==expected_py
    checks.append('synthetic_dropout_AdamW_checkpoint_restart_exact')
    train.atomic_json(RUN/'artifacts/structural_checks.json',{'checks':checks,'passed':len(checks),'real_training':False,'gpu_used':False})
    print(json.dumps({'passed':checks},indent=2))

if __name__=='__main__':main()
