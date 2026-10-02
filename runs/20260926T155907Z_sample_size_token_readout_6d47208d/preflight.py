"""Data nesting, pooling boundaries, paired initialization and fixed-step checks."""
from common import *
from torch.nn import functional as F
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views

def main():
    c=config_read(False);assert c['status']=='draft';torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    audit=json.loads((RUN/'artifacts/input_audit.json').read_text());assert audit['complete']
    assert sha(RUN/'artifacts/prepared.pt')==audit['prepared_sha256'] and sha(RUN/'artifacts/manifest.json')==audit['manifest_sha256']
    d=load_data(c);prior=torch.load(ROOT/c['prior_prepared'],map_location='cpu',weights_only=False)
    manifest=json.loads((RUN/'artifacts/manifest.json').read_text())
    def same(a,b):
        for k,v in a.items():
            if isinstance(v,dict):same(v,b[k])
            else:assert torch.equal(v,b[k]),k
    same(prior['source'],d['source20']);same(prior['valid'],d['valid'])
    for key,n,per in [('source20',2040,20),('source80',8160,80),('valid',510,5)]:
        assert d[key]['directions'].shape==(n,5000)
        assert torch.equal(torch.bincount(d[key]['labels'],minlength=102),torch.full((102,),per))
    assert d['source80']['rows'].tolist()==manifest['large_rows'] and d['valid']['rows'].tolist()==manifest['valid_rows']
    for k in ['directions','labels','rows']:assert torch.equal(d['source20'][k],d['source80'][k][:2040])
    for k,v in d['source20']['original'].items():
        if isinstance(v,dict):
            for n,t in v.items():assert torch.equal(t,d['source80']['original'][k][n][:2040])
        else:assert torch.equal(v,d['source80']['original'][k][:2040])
    sh=lambda e:{hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in e['directions']}
    assert len(sh(d['source80']))==8160 and not sh(d['source80'])&sh(d['valid'])
    # Mean reads contextualized ordinary outputs, excludes padding and handles empty views.
    z=torch.randn(2,5,52,requires_grad=True);mask=torch.tensor([[False,False,True,False,True],[False,True,True,True,True]])
    pooled=contextual_mean_hook(None,(),{'src_key_padding_mask':mask},z)
    assert torch.equal(pooled[0,0],(z[0,1]+z[0,3])/2) and torch.count_nonzero(pooled[1,0])==0
    pooled[:,0].sum().backward();assert torch.count_nonzero(z.grad[:,0])==0 and torch.count_nonzero(z.grad[mask])==0
    assert z.grad[0,1].abs().sum()>0 and z.grad[0,3].abs().sum()>0
    # Full-model synthetic equivalence and end-to-end feedback.
    x=(torch.randint(0,2,(3,5000),generator=torch.Generator().manual_seed(881))*2-1).to(torch.int8);x[0,3201:]=0;x[1]=0
    base=batch_from_views([generate_views(row.tolist(),input_kind='direction',budget=5000,window_sizes=(50,250)) for row in x],packet_patch=50,max_runs=128)
    batch=Batch(base,x,x!=0);paired={}
    for seed in c['seeds']:
        hashes=[]
        for cond in c['conditions']:
            torch.manual_seed(seed);m=make_model(c,cond['id']);hashes.append(state_hash(m))
            assert parameter_counts(m)['total']==119578
        assert len(set(hashes))==1;paired[str(seed)]=hashes[0]
    torch.manual_seed(123);original=GeneratorClassifier(pool='attention').eval()
    torch.manual_seed(123);cls=make_model(c,'A_20_cls').eval()
    with torch.inference_mode():assert torch.equal(batch.logits(original),batch.logits(cls))
    for name in ['A_20_cls','B_20_mean']:
        torch.manual_seed(8);m=make_model(c,name);initial=state_hash(m.generator);opt=make_optimizer(m,c,name)
        F.cross_entropy(batch.logits(m),torch.tensor([0,1,2])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.generator.parameters())
        opt.step();assert state_hash(m.generator)!=initial
        atomic_torch(RUN/'artifacts'/f'synthetic_{name}.pt',m.state_dict())
        restored=make_model(c,name);restored.load_state_dict(torch.load(RUN/'artifacts'/f'synthetic_{name}.pt',weights_only=True));m.eval();restored.eval()
        with torch.inference_mode():assert torch.equal(batch.logits(m),batch.logits(restored))
    sampler={}
    for n in [2040,8160]:
        ix=index_stream(n,21729,c);assert ix.shape==(3200,64) and torch.equal(ix,index_stream(n,21729,c))
        counts=torch.bincount(ix.flatten(),minlength=n);assert int(counts.sum())==204800 and int(counts.max()-counts.min())<=1
        sampler[str(n)]={'min':int(counts.min()),'max':int(counts.max()),'mean':float(counts.float().mean())}
    atomic_json(RUN/'artifacts/preflight.json',{'complete':True,'paired_initial_states':paired,'old_cls_forward_exact':True,'contextual_mean_mask_and_gradient_verified':True,'checkpoint_reconstruction_exact':True,'generator_feedback_both_readouts':True,'data_nesting_verified':True,'fixed_validation_exact':True,'sampler':sampler,'no_real_training':True,'errors':[]})
    print('PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
