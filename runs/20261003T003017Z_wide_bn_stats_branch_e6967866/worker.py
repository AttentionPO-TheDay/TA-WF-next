import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, json, time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from model import StatsModel, global_stats, span_mask, learning_rate, make_optimizer
from common import RUN, ROOT, save, savet, sha, config, deterministic


def metric(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'macro_f1': float(f1_score(y, p, labels=np.arange(102), average='macro', zero_division=0))}


def predict(model, e, bs):
    model.eval(); out = []
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    with torch.inference_mode():
        for i in range(0, len(e['labels']), bs):
            out.append(model(e['timestamps'][i:i+bs], e['tam'][i:i+bs]).cpu().numpy())
    assert all(torch.equal(v, dict(model.named_buffers())[k]) for k, v in buffers.items())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--kind', required=True); ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args(); c = config(); assert a.kind in c['conditions'] and a.seed in c['seeds']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    deterministic('cuda', 2); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    raw = torch.load(RUN/'artifacts/prepared.pt', weights_only=False, map_location='cpu')
    data = {role: {k: v.cuda() for k, v in e.items() if k in ('timestamps', 'tam', 'labels')} for role, e in raw.items()}
    labels = {role: e['labels'].cpu().numpy() for role, e in data.items()}
    source_stats = global_stats(data['source']['tam']).detach().float().cpu()
    stats_mean, stats_std = source_stats.mean(0), source_stats.std(0, unbiased=False).clamp_min(1e-6)
    out = RUN/'artifacts'/f'{a.kind}_{a.seed}'; out.mkdir(exist_ok=False)
    model = StatsModel(a.kind, stats_mean, stats_std).cuda()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; savet(out/'initial_state.pt', initial)
    # Shared initialization check against the historical wide+BN model.
    ref = torch.load(RUN/'artifacts'/f'baseline_{a.seed}'/'initial_state.pt', weights_only=True, map_location='cpu')
    for k, v in ref.items(): assert k in initial and torch.equal(v, initial[k]), k
    n, bs, steps = len(data['source']['labels']), c['batch_size'], c['optimizer_steps']
    pieces=[]; count=0; cycle=0
    while count < steps*bs:
        pieces.append(torch.randperm(n, generator=torch.Generator().manual_seed(a.seed+9000+cycle))); count += n; cycle += 1
    indices=torch.cat(pieces)[:steps*bs].reshape(steps,bs); savet(out/'index_stream.pt', indices)
    opt=make_optimizer(model); aug_rng=torch.Generator().manual_seed(a.seed+50000)
    history=[]; best=-1.; best_step=0; start=time.monotonic(); loss_sum=0.; prev=0
    for step in range(1, steps+1):
        for group in opt.param_groups: group['lr']=learning_rate(step, steps)
        model.train(); idx=indices[step-1].cuda(); tam=span_mask(data['source']['tam'][idx], aug_rng); y=data['source']['labels'][idx]
        opt.zero_grad(set_to_none=True); logits, features=model(data['source']['timestamps'][idx], tam, return_features=True)
        loss=torch.nn.functional.cross_entropy(logits,y); assert torch.isfinite(loss); loss.backward(); opt.step(); loss_sum += float(loss.detach())
        if step == 1:
            grads={k:(float(p.grad.norm()) if p.grad is not None else None) for k,p in model.named_parameters()}; save(out/'first_gradients.json', grads)
        if step in c['eval_steps']:
            pred={role:predict(model,e,c['eval_batch_size']) for role,e in data.items()}; scores={r:metric(labels[r], p.argmax(1)) for r,p in pred.items()}
            row={'step':step,'lr':learning_rate(step,steps),'train_loss':loss_sum/(step-prev),'elapsed_seconds':time.monotonic()-start,**scores}; loss_sum=0.;prev=step; history.append(row)
            if scores['valid']['accuracy'] > best: best=scores['valid']['accuracy'];best_step=step;savet(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt', {'state_dict':model.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            save(out/'history.json',history)
    last={role:predict(model,e,c['eval_batch_size']) for role,e in data.items()}; np.savez_compressed(out/'predictions_last.npz', **{r:p.argmax(1) for r,p in last.items()})
    ck=torch.load(RUN/'checkpoints'/f'{a.kind}_{a.seed}_best.pt',weights_only=True,map_location='cuda'); model.load_state_dict(ck['state_dict'])
    pred={role:predict(model,e,c['eval_batch_size']) for role,e in data.items()}; np.savez_compressed(out/'predictions_best.npz', **{r:p.argmax(1) for r,p in pred.items()}); np.savez_compressed(out/'logits_best.npz', **pred)
    report={'kind':a.kind,'seed':a.seed,'device':'cuda','steps_completed':steps,'validation_opportunities':len(history),'best_step':best_step,'parameters_total':sum(p.numel() for p in model.parameters()),'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'best':{r:metric(labels[r],p.argmax(1)) for r,p in pred.items()},'last':{r:metric(labels[r],p.argmax(1)) for r,p in last.items()},'verified_roles':['source','valid'],'inactive_parameters_unchanged':True,'stats_input':'same span-masked TAM; source-only mean/std; no labels','future_access':False,'elapsed_seconds':time.monotonic()-start}
    save(out/'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
