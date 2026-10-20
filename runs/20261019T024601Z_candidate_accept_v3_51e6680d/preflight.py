"""Source/selection-valid only; never reads target caches."""
import json,os,time
import numpy as np
import torch
from runtime import RUN,ROOT,C,setup,load_model,save,sha,entropy
from methods import tokenize,evaluate,encode,attach,standardize,assert_base_locked
from candidate import gpu_dict,banks,neighbors,nearest_distance,target_distribution,acceptance_inputs,policy_logits

def main():
    setup();seed=int(os.environ['CANDIDATE_SEED']);start=time.monotonic()
    old=ROOT/C['previous_run'];prev=json.loads((old/'artifacts'/f'source_preflight_{seed}.json').read_text());assert prev['passed']
    path=old/'artifacts'/f'source_gmm_{seed}.pt';assert sha(path)==prev['artifact_sha256']
    art=torch.load(path,map_location='cpu',weights_only=True);p=gpu_dict(art['fits']['1']);bank_data=banks(art,p)
    data=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True);valid=data['valid']
    assert len(data['source']['rows'])==18553 and len(valid['rows'])==2160
    m=load_model(seed);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    t=tokenize(m,valid);e,z=evaluate(m,t)
    # Bandwidth from source validation nearest source neighbors, without labels.
    _,nearest=neighbors(e,p,bank_data,torch.tensor(1.,device='cuda'))
    temp=nearest.flatten().median().clamp_min(.001)
    votes,_=neighbors(e,p,bank_data,temp)
    d=nearest_distance(e,p);bp=z.argmax(-1);cp=votes.mean(0).argmax(-1)
    gain=(votes.gather(2,cp[None,:,None].expand(2,-1,1)).squeeze(-1).log()-votes.gather(2,bp[None,:,None].expand(2,-1,1)).squeeze(-1).log()).min(0).values
    distcut=torch.quantile(d,.99);hcut=torch.quantile(entropy(z),.95);top=z.topk(5,dim=-1).indices
    y=valid['labels'].cuda();correct=bp==y;wrong=~correct;trials=[];chosen=None
    for threshold in C['adaptation']['acceptance_grid']:
        keep=(cp!=bp)&(gain>threshold)&(top==cp[:,None]).any(-1)&(d<=distcut)
        harmful=int((keep&correct&(cp!=y)).sum());recovery=int((keep&wrong&(cp==y)).sum())
        feasible=int(keep.sum())>=10 and recovery>=2*max(harmful,1) and harmful/max(int(correct.sum()),1)<=.01
        trials.append({'threshold':threshold,'changes':int(keep.sum()),'harmful':harmful,'recovered':recovery,'neutral_wrong_changes':int((keep&wrong&(cp!=y)).sum()),'nar':harmful/max(int(correct.sum()),1),'feasible':bool(feasible)})
        if feasible and chosen is None:chosen=threshold
    calibration={'neighbor_temperature':temp.cpu(),'entropy_cutoff':hcut.cpu(),'distance_cutoff':distcut.cpu(),'acceptance_enabled':chosen is not None,'acceptance_threshold':chosen if chosen is not None else 0.,'seed':seed}
    info={'votes':votes,'distance':d,'entropy':entropy(z),'top5':top}
    # Verify probability constraints and allow changing class without target labels.
    checks={}
    for target in ['hard','soft_teacher','soft_neighbor']:
        q=target_distribution(z,votes,target)
        checks[target]={'probability_sum_max_error':float((q.sum(-1)-1).abs().max()),'argmax_teacher_disagreement':float((q.argmax(-1)!=bp).float().mean()),'mean_entropy':float(-(q*q.clamp_min(1e-12).log()).sum(-1).mean())}
    for kind in ['residual','lora']:
        test=load_model(seed);adapter=attach(test,kind)
        with torch.no_grad():_,zz=encode(test,t[:16],adapter);_,base=encode(m,t[:16])
        assert torch.equal(zz,base),kind
        ee,zz=encode(test,t[:16],adapter);q=target_distribution(z[:16],votes[:,:16],'soft_neighbor')
        loss=torch.nn.functional.kl_div((zz/2).log_softmax(-1),q,reduction='batchmean')*4
        loss.backward();grad=sum(float(v.grad.square().sum()) for v in adapter.parameters() if v.grad is not None)
        assert np.isfinite(grad) and grad>0
        assert_base_locked(test,initial)
        checks[kind]={'zero_init_bitwise':True,'gradient_square_sum':grad,'base_locked':True,'parameters':sum(v.numel() for v in adapter.parameters())}
    # An artificial candidate formed from source-neighbor votes tests the actual reject path.
    pseudo=votes.mean(0).clamp_min(1e-12).log()
    for policy in ['valid_entropy','evidence_accept']:
        out,keep=policy_logits(pseudo,z,info,gpu_dict(calibration),policy)
        assert torch.equal(out[~keep],z[~keep]);checks[policy]={'accepted':int(keep.sum()),'rejected_bitwise':True}
    anchors=art['anchor_indices'];source=data['source']
    anchor_t=tokenize(m,{k:source[k][anchors] for k in ('timestamps','tam','rows')})
    torch.save({'calibration':calibration,'valid_embeddings':e.cpu(),'valid_logits':z.cpu(),'valid_rows':valid['rows'],'source_anchor_tokens':anchor_t.cpu(),'source_anchor_embeddings':art['source_embeddings'][anchors],'seed':seed},RUN/'artifacts'/f'source_calibration_{seed}.pt')
    save(f'preflight_{seed}.json',{'passed':True,'source_only':True,'seed':seed,'source_rows':18553,'valid_rows':2160,'valid_baseline_accuracy':float(correct.float().mean()),'neighbor_accuracy':float((cp==y).float().mean()),'trials':trials,'acceptance_enabled':chosen is not None,'threshold':chosen,'entropy_cutoff':float(hcut),'distance_cutoff':float(distcut),'neighbor_temperature':float(temp),'checks':checks,'seconds':time.monotonic()-start,'calibration_sha256':sha(RUN/'artifacts'/f'source_calibration_{seed}.pt')})
    print(json.dumps({'seed':seed,'passed':True,'acceptance_enabled':chosen is not None,'threshold':chosen,'trials':trials}),flush=True)

if __name__=='__main__':main()
