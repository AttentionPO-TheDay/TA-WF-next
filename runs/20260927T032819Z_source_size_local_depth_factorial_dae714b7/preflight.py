"""Bounded data and architecture checks, then freeze and import verified A."""
import copy,time
from datetime import datetime,timezone
from common import *
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views
from torch.nn import functional as F

def main():
    c=config_read(False);assert c['status']=='draft'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for p,h in c['anchor_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/c['prior_prepared'])==c['prior_prepared_sha256'] and sha(ROOT/c['prior_manifest'])==c['prior_manifest_sha256']
    audit=json.loads((RUN/'artifacts/input_audit.json').read_text());assert audit['complete']
    c.update(prepared_input=str((RUN/'artifacts/prepared.pt').relative_to(ROOT)),sampling_manifest=str((RUN/'artifacts/manifest.json').relative_to(ROOT)))
    data=load_data(c);old=torch.load(ROOT/c['prior_prepared'],map_location='cpu',weights_only=False)
    def exact(x,y):
        if isinstance(x,dict):assert set(x)==set(y);[exact(x[k],y[k]) for k in x]
        else:assert torch.equal(x,y)
    for role in ('source20','source80','valid'):exact(data[role],old[role])
    for role,n in [('source80',8160),('source150',15300),('valid',510)]:
        e=data[role];assert len(e['labels'])==n and torch.all(torch.bincount(e['labels'],minlength=102)==n//102)
    def prefix(x,y):
        if isinstance(x,dict):[prefix(x[k],y[k]) for k in x]
        else:assert torch.equal(x,y[:len(x)])
    prefix(data['source80'],data['source150'])
    manifest=json.loads((RUN/'artifacts/manifest.json').read_text())
    assert manifest['small_rows']==data['source80']['rows'].tolist() and manifest['large_rows']==data['source150']['rows'].tolist()
    assert not set(manifest['large_rows'])&set(manifest['excluded_full_valid_overlap'])
    assert not set(manifest['large_rows'])&set(manifest['excluded_label_conflict'])
    assert len(set(manifest['large_rows']))==15300
    rawdirs=data['source150']['directions'].numpy();hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in rawdirs]
    assert len(set(hashes))==15300
    for p,h in manifest['raw_sha256'].items():assert sha(p)==h
    # Synthetic only: padding, batch, save/load, generator and both local layers.
    d=torch.zeros(3,5000,dtype=torch.int8)
    for i,length in enumerate([84,163,237]):
        d[i,:length]=torch.where(torch.arange(length)%(i+3)==0,1,-1).to(torch.int8)
    original=batch_from_views([generate_views(v.numpy(),input_kind='direction',budget=5000,window_sizes=(50,250)) for v in d],packet_patch=50,max_runs=128)
    batch=Batch(original,d,d!=0);model_checks=[]
    for seed in c['seeds']:
        torch.manual_seed(seed);base=make_model(c,'A_80_l1');base_hash=state_hash(base);base_rng=torch.get_rng_state().clone()
        oldreport=json.loads((ROOT/c['prior_run']/'artifacts'/f'C_80_cls_{seed}'/'report.json').read_text())
        assert base_hash==oldreport['initial_state_sha256']
        for name in ['B_150_l1','C_80_l2','D_150_l2']:
            torch.manual_seed(seed);model=make_model(c,name)
            assert shared_state_hash(model)==base_hash and torch.equal(torch.get_rng_state(),base_rng)
            model.eval()
            with torch.inference_mode():
                output=batch.logits(model);single=torch.cat([batch.subset(slice(i,i+1)).logits(model) for i in range(3)])
                torch.testing.assert_close(output,single,rtol=2e-5,atol=2e-5)
                changed=copy.deepcopy(batch)
                for view in ('packet','runs','windows'):
                    tensor=getattr(changed.original,view);mask=getattr(changed.original,view+'_mask');tensor[~mask]=777
                changed.directions[~changed.observed]=1
                torch.testing.assert_close(output,changed.logits(model),rtol=0,atol=0)
                clone=make_model(c,name);clone.load_state_dict(model.state_dict());clone.eval()
                torch.testing.assert_close(output,batch.logits(clone),rtol=0,atol=0)
            model.train();opt=make_optimizer(model,c,name);before={k:v.detach().clone() for k,v in model.state_dict().items()}
            F.cross_entropy(batch.logits(model),torch.tensor([0,1,2])).backward()
            for p in model.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
            for view in model.classifier.local_encoders.values():
                for layer in view.layers:assert layer.self_attn.in_proj_weight.grad.norm()>0 and layer.linear1.weight.grad.norm()>0
            for k in ['conv1.weight','conv2.weight','score.weight','projection.weight']:assert dict(model.generator.named_parameters())[k].grad.norm()>0
            opt.step();assert state_hash(model.generator)!=oldreport['generator_initial_sha256']
            model_checks.append({'seed':seed,'condition':name,'shared_initial_sha256':base_hash,'parameters':parameter_counts(model),'synthetic_checks':True})
        ix=index_stream(8160,seed,c);ref=torch.load(ROOT/c['prior_run']/'artifacts'/f'C_80_cls_{seed}'/'index_stream.pt',weights_only=True)
        assert torch.equal(ix,ref)
    assert [lr_for_step(c,s) for s in [1,3200,3201,6400,6401,9600,9601,12800]]==[.001,.001,.001,.001,.0003,.0003,.0001,.0001]
    c['status']='frozen';c['frozen_utc']=datetime.now(timezone.utc).isoformat();c['prepared_sha256']=audit['prepared_sha256'];c['manifest_sha256']=audit['manifest_sha256']
    atomic_json(RUN/'config.json',c)
    paths=list(RUN.glob('*.py'))+list((ROOT/'src/ta_wf_next').glob('*.py'))+[ROOT/'scripts/experiment.py']
    freeze={'config_sha256':sha(RUN/'config.json'),'plan_sha256':sha(RUN/'PLAN.md'),'datasets_sha256':sha(ROOT/'configs/datasets.json'),'prepared_sha256':sha(RUN/'artifacts/prepared.pt'),'manifest_sha256':sha(RUN/'artifacts/manifest.json'),'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    atomic_json(RUN/'artifacts/freeze.json',freeze)
    atomic_json(RUN/'artifacts/preflight.json',{'complete':True,'synthetic_checks':model_checks,'source80_and_valid_exact':True,'nested_data_exact':True,'lr_checks':True,'baseline_index_stream_exact':True,'no_new_real_training':True,'errors':[]})
    print('PREFLIGHT PASSED; FROZEN',flush=True)

if __name__=='__main__':main()
