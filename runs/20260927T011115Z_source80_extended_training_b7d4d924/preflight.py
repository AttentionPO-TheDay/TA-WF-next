"""Frozen inputs and exact synthetic optimizer/RNG continuation checks."""
from common import *
from torch.nn import functional as F
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views

def main():
    c=config_read(False);assert c['status']=='draft';torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert sha(ROOT/c['prepared_input'])==c['prepared_sha256'];assert sha(ROOT/c['sampling_manifest'])==c['manifest_sha256']
    for p,h in c['anchor_sha256'].items():assert sha(ROOT/p)==h,p
    prior=ROOT/c['prior_run'];assert json.loads((prior/'artifacts/integrity.json').read_text())['complete']
    dc=json.loads((ROOT/'configs/datasets.json').read_text());assert (Path(dc['data_root'])/dc['datasets']['proteus_temporal']['path']).is_dir()
    d=load_data(c);assert len(d['source80']['labels'])==8160 and len(d['valid']['labels'])==510
    records=[]
    for cond in c['conditions']:
        for seed in c['seeds']:
            key=f"{cond['id']}_{seed}";old=torch.load(prior/'checkpoints'/f'{key}_latest.pt',map_location='cpu',weights_only=False)
            assert old['step']==old['next_index_stream_row']==3200
            assert all(int(v['step'])==3200 for v in old['optimizer']['state'].values())
            ix=index_stream(8160,seed,c);assert torch.equal(ix[:3200],torch.load(prior/'artifacts'/key/'index_stream.pt',weights_only=True))
            counts=torch.bincount(ix.flatten(),minlength=8160);assert int(counts.sum())==819200 and int(counts.max()-counts.min())<=1
            records.append({'key':key,'start_step':old['step'],'index_prefix_exact':True})
    for step,lr in [(3201,.001),(6400,.001),(6401,.0003),(9600,.0003),(9601,.0001),(12800,.0001)]:assert lr_for_step(c,step)==lr
    x=(torch.randint(0,2,(3,5000),generator=torch.Generator().manual_seed(19))*2-1).to(torch.int8);x[0,3501:]=0;x[1]=0
    b=batch_from_views([generate_views(row.tolist(),input_kind='direction',budget=5000,window_sizes=(50,250)) for row in x],packet_patch=50,max_runs=128)
    batch=Batch(b,x,x!=0)
    for cond in c['conditions']:
        name=cond['id'];torch.manual_seed(17);model=make_model(c,name);opt=make_optimizer(model,c,name)
        def step(m,o,idx):
            for group in o.param_groups:group['lr']=lr_for_step(c,idx)
            m.train();o.zero_grad(set_to_none=True);F.cross_entropy(batch.logits(m),torch.tensor([0,1,2])).backward();o.step()
        step(model,opt,6400)
        path=RUN/'artifacts'/f'synthetic_resume_{name}.pt'
        atomic_torch(path,{'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state()})
        step(model,opt,6401);expected=state_hash(model);expected_rng=torch.get_rng_state().clone()
        restored=make_model(c,name);ropt=make_optimizer(restored,c,name);saved=torch.load(path,map_location='cpu',weights_only=False)
        restored.load_state_dict(saved['state_dict']);ropt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);step(restored,ropt,6401)
        assert state_hash(restored)==expected and torch.equal(torch.get_rng_state(),expected_rng)
        for a,b in zip(opt.state_dict()['state'].values(),ropt.state_dict()['state'].values()):
            for k,v in a.items():assert torch.equal(v,b[k]) if torch.is_tensor(v) else v==b[k]
    atomic_json(RUN/'artifacts/preflight.json',{'complete':True,'anchors':records,'lr_boundary_checks':True,'synthetic_continuation_parameters_optimizer_rng_exact':True,'no_new_real_training':True,'errors':[]})
    print('PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
