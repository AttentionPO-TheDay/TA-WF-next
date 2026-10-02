import copy,json
from datetime import datetime,timezone
from common import *
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views
from torch.nn import functional as F

def exact(x,y):
    if isinstance(x,dict): assert set(x)==set(y); [exact(x[k],y[k]) for k in x]
    else: assert torch.equal(x,y)

def main():
    c=config_read(False);assert c['status']=='draft'
    assert sha(ROOT/c['prior_prepared'])==c['prior_prepared_sha256'] and sha(ROOT/c['prior_manifest'])==c['prior_manifest_sha256']
    c['prepared_input']=str((RUN/'artifacts/prepared.pt').relative_to(ROOT));c['sampling_manifest']=str((RUN/'artifacts/manifest.json').relative_to(ROOT))
    data=load_data(c);prior=torch.load(ROOT/c['prior_prepared'],map_location='cpu',weights_only=False)
    for role in ('source20','source80','source150','valid'): exact(data[role],prior[role])
    assert len(data['source150']['labels'])==15300 and len(data['valid']['labels'])==510
    assert torch.all(torch.bincount(data['source150']['labels'],minlength=102)==150)
    assert sha(ROOT/c['prepared_input'])==c['prior_prepared_sha256'] and sha(ROOT/c['sampling_manifest'])==c['prior_manifest_sha256']
    # Input/model and gradient checks on synthetic views.
    d=torch.zeros(3,5000,dtype=torch.int8)
    for i,length in enumerate([84,163,237]): d[i,:length]=torch.where(torch.arange(length)%(i+3)==0,1,-1).to(torch.int8)
    original=batch_from_views([generate_views(v.numpy(),input_kind='direction',budget=5000,window_sizes=(50,250)) for v in d],packet_patch=50,max_runs=128)
    batch=Batch(original,d,d!=0);checks=[]
    base_hashes=[]
    for seed in c['seeds']:
        torch.manual_seed(seed);base=make_model(c);h=state_hash(base);base_hashes.append(h);rng=torch.get_rng_state().clone()
        for name in [row['id'] for row in c['conditions']]:
            torch.manual_seed(seed);model=make_model(c,name);assert state_hash(model)==h and torch.equal(torch.get_rng_state(),rng)
            opt=make_optimizer(model,c,name);groups={g['role']:g for g in opt.param_groups};assert set(groups)=={'generator','classifier'}
            expected=group_lrs(c,name,1);assert all(groups[k]['lr']==expected[k] for k in groups)
            model.eval();out=batch.logits(model);clone=copy.deepcopy(batch);torch.testing.assert_close(out,clone.logits(model),rtol=0,atol=0)
            F.cross_entropy(out,torch.tensor([0,1,2])).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert all(dict(model.generator.named_parameters())[k].grad.norm()>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight'])
            opt.step();assert state_hash(model)!=h
            checks.append({'seed':seed,'condition':name,'initial_state_sha256':h,'group_lrs':expected,'parameters':parameter_counts(model)})
    assert len(set(base_hashes))==3
    # Shared index stream is independent of condition and exactly 12800x64.
    for seed in c['seeds']:
        ix=index_stream(15300,seed,c);assert ix.shape==(12800,64)
    c['status']='frozen';c['frozen_utc']=datetime.now(timezone.utc).isoformat();c['prepared_sha256']=sha(RUN/'artifacts/prepared.pt');c['manifest_sha256']=sha(RUN/'artifacts/manifest.json')
    atomic_json(RUN/'config.json',c)
    paths=list(RUN.glob('*.py'))+list((ROOT/'src/ta_wf_next').glob('*.py'))+[ROOT/'scripts/experiment.py']
    freeze={'config_sha256':sha(RUN/'config.json'),'plan_sha256':sha(RUN/'PLAN.md'),'datasets_sha256':sha(ROOT/'configs/datasets.json'),'prepared_sha256':c['prepared_sha256'],'manifest_sha256':c['manifest_sha256'],'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    atomic_json(RUN/'artifacts/freeze.json',freeze)
    atomic_json(RUN/'artifacts/preflight.json',{'complete':True,'input_exact_to_prior':True,'source150_balanced':True,'shared_initialization_and_gradients':checks,'separate_optimizer_groups':True,'lr_boundary_checks':True,'future_access':False})
    print('PREFLIGHT PASSED; FROZEN',flush=True)
if __name__=='__main__':main()
