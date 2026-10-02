"""Synthetic checks for the new regularizer and paired factorial design."""
import copy
import time
from common import *
from torch.nn import functional as F
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views
from input_audit import audit

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    c=config_read(False);assert c['status']=='draft'
    audit()
    g=torch.Generator().manual_seed(919)
    x=(torch.randint(0,2,(3,5000),generator=g)*2-1).to(torch.int8);x[0,3701:]=0;x[1]=0
    views=[generate_views(row.tolist(),input_kind='direction',budget=5000,window_sizes=(50,250)) for row in x]
    batch=Batch(batch_from_views(views,packet_patch=50,max_runs=128),x,x!=0)
    paired={}
    for seed in c['seeds']:
        hashes=[];outputs=[]
        for cond in c['conditions']:
            torch.manual_seed(seed);m=make_model(c,cond['id']).eval();hashes.append(state_hash(m))
            assert parameter_counts(m)=={'total':119578,'trainable':119578,'generator_total':6064,'generator_trainable':6064}
            assert all(d.p==cond['classifier_dropout'] for d in m.classifier.modules() if isinstance(d,torch.nn.Dropout))
            assert all(d.dropout==cond['classifier_dropout'] for d in m.classifier.modules() if isinstance(d,torch.nn.MultiheadAttention))
            opt=make_optimizer(m,c,cond['id'])
            assert opt.param_groups[0]['weight_decay']==cond['classifier_weight_decay'] and opt.param_groups[1]['weight_decay']==.0001
            assert {id(p) for p in opt.param_groups[0]['params']}=={id(p) for p in m.classifier.parameters()}
            assert {id(p) for p in opt.param_groups[1]['params']}=={id(p) for p in m.generator.parameters()}
            with torch.inference_mode():outputs.append(batch.logits(m))
        assert len(set(hashes))==1 and all(torch.equal(outputs[0],v) for v in outputs)
        paired[str(seed)]=hashes[0]
    m=make_model(c,'R11_both');initial={n:p.detach().clone() for n,p in m.generator.named_parameters()}
    assert anchor_penalty(m,initial,.01).item()==0
    first=next(m.generator.parameters())
    with torch.no_grad():first.flatten()[0].add_(.2)
    loss=anchor_penalty(m,initial,.01);loss.backward()
    assert all(p.grad is None for p in m.classifier.parameters())
    for n,p in m.generator.named_parameters():assert torch.allclose(p.grad,.01*(p.detach()-initial[n]),atol=1e-8,rtol=1e-5)
    old_loss=loss.item()
    with torch.no_grad():
        for p in m.generator.parameters():p.sub_(.1*p.grad)
    assert anchor_penalty(m,initial,.01).item()<old_loss
    m.zero_grad(set_to_none=True);m.eval();logits=batch.logits(m);ce=F.cross_entropy(logits,torch.tensor([0,1,2]))
    a=torch.autograd.grad(ce,tuple(m.parameters()),retain_graph=True)
    b=torch.autograd.grad(ce+anchor_penalty(m,initial,0),tuple(m.parameters()))
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    # Exact resume includes nonzero anchor and dropout; reference is the ORIGINAL init.
    torch.manual_seed(888);m=make_model(c,'R11_both');initial={n:p.detach().clone() for n,p in m.generator.named_parameters()};opt=make_optimizer(m,c,'R11_both')
    def step(model,optimizer):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(batch.logits(model),torch.tensor([0,1,2]))+anchor_penalty(model,initial,.01)
        loss.backward();optimizer.step()
    start=time.monotonic();step(m,opt)
    atomic_torch(RUN/'artifacts/synthetic_resume.pt',{'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state()})
    step(m,opt);expected=state_hash(m)
    restored=make_model(c,'R11_both');ropt=make_optimizer(restored,c,'R11_both')
    saved=torch.load(RUN/'artifacts/synthetic_resume.pt',map_location='cpu',weights_only=False)
    restored.load_state_dict(saved['state_dict']);ropt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);step(restored,ropt)
    assert state_hash(restored)==expected
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in restored.generator.parameters())
    atomic_json(RUN/'artifacts/preflight.json',{'complete':True,'paired_initial_states':paired,'same_eval_logits':True,'anchor_gradient_analytic_match':True,'anchor_classifier_isolation':True,'anchor_toward_initialization':True,'lambda_zero_exact_gradient_parity':True,'optimizer_groups_and_dropout_verified':True,'resume_exact':True,'synthetic_three_steps_seconds':time.monotonic()-start,'no_real_data_training':True,'errors':[]})
    print('PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
