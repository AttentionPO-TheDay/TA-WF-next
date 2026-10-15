"""Controlled loss ablation, no query selection and no support-label access."""
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import port_base as B

R, ROOT, C, DEVICE = B.R, B.ROOT, B.C, B.DEVICE
CONDITIONS = list(C['loss_weights'])
HIST = ROOT / C['comparison_run']


def setup():
    assert C['status'] == 'frozen' and C['id'] == R.name
    torch.set_num_threads(C['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def loader(x, t, y, rows, seed):
    generator = torch.Generator().manual_seed(seed)
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(t), torch.as_tensor(y), torch.as_tensor(rows))
    return DataLoader(data, batch_size=128, shuffle=True, drop_last=True, num_workers=0, generator=generator)


def next_batch(iterator, data_loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator


def parameter_changes(model, initial):
    delta, original, changed = 0., 0., 0
    for k, value in model.named_parameters():
        before = initial[k].to(value.device)
        delta += float((value.detach().double()-before.double()).square().sum())
        original += float(before.double().square().sum())
        changed += int(not torch.equal(value.detach(), before))
    bn = [k for k in model.state_dict() if k.endswith('running_mean') or k.endswith('running_var')]
    return {'relative_parameter_change': (delta/max(original, 1e-12))**.5,
            'changed_parameter_tensors': changed,
            'changed_bn_buffers': sum(not torch.equal(model.state_dict()[k].cpu(), initial[k]) for k in bn)}


def train(model, source, support, condition, seed, role):
    assert set(support) == {'timestamps', 'tam', 'rows'}, 'support labels must not reach adaptation'
    assert condition in C['loss_weights']
    random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
    source_loader = loader(source['timestamps'], source['tam'], source['labels'], source['rows'], 2025)
    target_loader = loader(support['timestamps'], support['tam'], np.zeros(1024, dtype=np.int64), support['rows'], 2026)
    source_iterator = iter(source_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=C['adaptation']['learning_rate'])
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    streams = {name:hashlib.sha256() for name in ('source', 'target', 'pseudo')}
    weights = C['loss_weights'][condition]
    histories, gmms = [], []
    updates, max_gradient, start = 0, 0., time.monotonic()
    for epoch in range(100):
        if time.monotonic()-start > C['budget']['max_task_seconds']:
            raise TimeoutError('single task budget exceeded')
        if epoch % 5 == 0:
            report = B.gmm_pseudo_labels(model, support['timestamps'], support['tam'])
            selected = report['selected']
            refresh = epoch//5
            gmms.append({'epoch':epoch, **{k:v for k,v in report.items() if k not in ('selected','predictions','probabilities')}})
            np.savez_compressed(R/'artifacts'/f'gmm_{seed}_{role}_{condition}_{epoch}.npz',
                rows=support['rows'], selected=selected, predictions=report['predictions'], probabilities=report['probabilities'])
            pseudo_loader = loader(support['timestamps'][selected], support['tam'][selected],
                report['predictions'][selected], support['rows'][selected], 2030+refresh)
            pseudo_iterator = iter(pseudo_loader)
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        for batch_index, target_batch in enumerate(target_loader):
            source_batch, source_iterator = next_batch(source_iterator, source_loader)
            pseudo_batch, pseudo_iterator = next_batch(pseudo_iterator, pseudo_loader)
            for key, batch in zip(('source','target','pseudo'), (source_batch,target_batch,pseudo_batch)):
                streams[key].update(batch[3].numpy().astype('<i8').tobytes())
            torch.manual_seed(2024+100000+epoch*8+batch_index)
            sx,st,sy = [v.to(DEVICE) for v in source_batch[:3]]
            tx,tt = [v.to(DEVICE) for v in target_batch[:2]]
            px,pt,py = [v.to(DEVICE) for v in pseudo_batch[:3]]
            optimizer.zero_grad(set_to_none=True)
            so,sf = B.forward_with_features(model,sx,st)
            to,tf = B.forward_with_features(model,tx,tt)
            po,_ = B.forward_with_features(model,px,pt)
            marginal = to.softmax(1).mean(0)
            terms = [torch.nn.functional.cross_entropy(so,sy), torch.nn.functional.cross_entropy(po,py),
                B.entropy(to).mean()+torch.sum(marginal*torch.log(marginal+1e-5)), B.calculate_mmd_loss(sf,tf)]
            # Keep every forward and graph; only loss coefficients change.
            loss = sum(w*term for w,term in zip(weights,terms))
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('nonfinite gradient')
                    max_gradient = max(max_gradient, float(p.grad.detach().abs().max()))
            optimizer.step()
            totals += np.array([float(term.detach()) for term in terms])
            updates += 1
        histories.append({'epoch':epoch, **dict(zip(C['loss_order'],(totals/8).tolist())),
            'weighted_total':float(np.dot(totals/8,weights)), 'pseudo_coverage':gmms[-1]['coverage']})
        if epoch % 10 == 0 or epoch == 99:
            B.event('epoch',seed=seed,role=role,condition=condition,epoch=epoch,updates=updates)
    assert updates == 800 and max_gradient > 0
    changes = parameter_changes(model, initial)
    assert changes['changed_parameter_tensors'] > 0
    model.eval()
    return {'updates':updates,'loss_history':histories,'gmm_history':gmms,
        'streams':{k:v.hexdigest() for k,v in streams.items()},'task_seconds':time.monotonic()-start,
        'max_gradient':max_gradient, 'peak_memory_bytes':torch.cuda.max_memory_allocated(), **changes}


def freeze_inputs():
    prior = json.loads((HIST/'artifacts/freeze.json').read_text())
    paths = list(R.glob('*.py'))+[R/'PLAN.md',R/'SOURCES.md',R/'config.json',ROOT/'configs/datasets.json']
    paths += [ROOT/p for p in C['prepared'].values()]
    paths += [ROOT/C['base_checkpoint_run']/'checkpoints'/f'wide_bn_{s}_best.pt' for s in C['seeds']]
    paths += [HIST/'artifacts/scores.json']
    paths += list((HIST/'artifacts').glob('pred_*.npz'))+list((HIST/'artifacts').glob('split_*.npz'))
    frozen = {}
    for p in paths:
        key = str(p.relative_to(ROOT)); value = B.sha(p)
        if key in prior: assert value == prior[key], f'prior input hash changed: {key}'
        frozen[key] = value
    B.save_json('freeze.json',frozen)


def report_results(scores):
    def avg(cond,key,role=None):
        return float(np.mean([r[key] for r in scores if r['condition']==cond and (role is None or r['role']==role)]))
    controls = {(r['seed'],r['role']):r for r in scores if r['condition']=='full_control'}
    contrasts = {}
    for cond in CONDITIONS[1:]:
        rows = [r for r in scores if r['condition']==cond]
        f1 = (avg(cond,'macro_f1')-avg('full_control','macro_f1'))*100
        neg = (avg(cond,'negative_adaptation_rate')-avg('full_control','negative_adaptation_rate'))*100
        positive = sum(r['macro_f1']>controls[(r['seed'],r['role'])]['macro_f1'] for r in rows)
        contrasts[cond] = {'macro_f1_delta_pp':f1,'negative_delta_pp':neg,'positive_cells':positive,
            'candidate_signal':f1>=.5 and neg<=-1 and positive>=6}
    B.save_json('contrasts.json',contrasts)
    lines = ['# Proteus 损失组件消融结果','', '所有54个新任务完成并通过审计；历史36组预测注明复用，主归因相对本轮full_control。', '',
        '| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |','|---|---:|---:|---:|---:|']
    for cond in ['frozen','proteus_port']+CONDITIONS:
        lines.append(f"| {cond} | {avg(cond,'macro_f1')*100:.3f}% | {avg(cond,'accuracy')*100:.3f}% | {avg(cond,'negative_adaptation_rate')*100:.3f}% | {avg(cond,'recovery_rate')*100:.3f}% |")
    for role in C['roles']:
        lines += ['',f'## {role}','', '| 条件 | Macro-F1 | 负适应率 |','|---|---:|---:|']
        for cond in ['frozen','proteus_port']+CONDITIONS:
            lines.append(f"| {cond} | {avg(cond,'macro_f1',role)*100:.3f}% | {avg(cond,'negative_adaptation_rate',role)*100:.3f}% |")
    lines += ['','```json',json.dumps(contrasts,ensure_ascii=False,indent=2),'```','',
        'no_pseudo仅删除伪标签监督损失；source_ce_only仍保留目标三路前向和BN更新。结果是当前wide+BN及开发日期的条件性损失贡献，不能直接归因于GMM本身或外推为论文RF复现。',
        '新配对调度与上一轮端口不同。全部条件100 epoch/800步；历史10步adapter不是同预算控制。外部未来与其他因素未开放。']
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return contrasts


def main():
    setup()
    start = time.time()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('batch wall budget exceeded')))
    signal.alarm(C['budget']['max_wall_seconds'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available()
    assert not (R/'artifacts/events.jsonl').exists(), 'refuse duplicate execution'
    freeze_inputs()
    source = B.load_prepared(C['prepared']['source_valid'],'source')
    valid = B.load_prepared(C['prepared']['source_valid'],'valid')
    source_all = {k:np.concatenate([source[k],valid[k]]) for k in source}
    targets = {role:B.load_prepared(C['prepared'][role]) for role in C['roles']}
    splits = {}
    for role,t in targets.items():
        s,q,detail = B.split_targets(source_all,role,t)
        with np.load(HIST/'artifacts'/f'split_{role}.npz') as old:
            assert np.array_equal(s,old['support_indices']) and np.array_equal(q,old['query_indices'])
        np.savez_compressed(R/'artifacts'/f'split_{role}.npz',support=t['rows'][s],query=t['rows'][q],support_indices=s,query_indices=q)
        splits[role]=(s,q)
    scores=json.loads((HIST/'artifacts/scores.json').read_text())
    for row in scores:
        row['origin']='historical_reference'
        p=f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz"
        shutil.copy2(HIST/'artifacts'/p,R/'artifacts'/p)
    B.event('preflight',gpu=C['physical_gpu'],tasks=54,split_match=True)
    B.registry('running','GPU0单进程执行54项损失消融；每项100epoch/800步，support标签未传入适应，query只最终评分。')
    for seed in C['seeds']:
        for role in C['roles']:
            t=targets[role]; s,q=splits[role]
            support={k:t[k][s] for k in ('timestamps','tam','rows')}
            paired=None
            for cond in CONDITIONS:
                model,ck=B.load_model(seed)
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    x=torch.from_numpy(source['timestamps'][:4]).to(DEVICE)
                    tam=torch.from_numpy(source['tam'][:4]).to(DEVICE)
                    assert torch.allclose(B.forward_with_features(model,x,tam)[0],model(x,tam),atol=1e-5,rtol=1e-5)
                result=train(model,source,support,cond,seed,role)
                if paired is None: paired=result['streams']
                else:
                    for key in ('source','target'): assert paired[key]==result['streams'][key], 'unmatched input stream'
                # Adaptation complete; only now allow query prediction and labels.
                z=B.predict(model,t['timestamps'][q],t['tam'][q],128)
                assert np.isfinite(z).all()
                pred=z.argmax(1)
                with np.load(R/'artifacts'/f'pred_{seed}_{role}_frozen.npz') as old:
                    assert np.array_equal(old['rows'],t['rows'][q]) and np.array_equal(old['labels'],t['labels'][q])
                    base=old['predictions']
                row={'seed':seed,'role':role,'condition':cond,'origin':'new','rows':len(q),**B.cell_metrics(t['labels'][q],base,pred),**result}
                np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_{cond}.npz',rows=t['rows'][q],labels=t['labels'][q],predictions=pred,logits=z)
                torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'seed':seed,'role':role,'condition':cond},R/'checkpoints'/f'{seed}_{role}_{cond}.pt')
                scores.append(row); B.save_json('scores.json',scores)
                completed=sum(r['origin']=='new' for r in scores)
                B.save_json('progress.json',{'completed_tasks':completed,'total_tasks':54,'seed':seed,'role':role,'condition':cond,'elapsed_seconds':time.time()-start})
                B.event('task_completed',seed=seed,role=role,condition=cond,completed=completed)
                del model
                torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(R/'postrun_audit.py')],check=True)
    contrasts=report_results(scores)
    summary='54项完成且审计通过；损失消融对比见RESULTS.md/contrasts.json，历史对照与配对控制明确区分。'
    B.registry('completed',summary)
    status=ROOT/'STATUS.md'; contents=status.read_text()
    marker=f'当前 Proteus 损失组件消融：`{R.name}`。'
    lines=contents.splitlines()
    for i,line in enumerate(lines):
        if line.startswith(marker): lines[i]=marker+summary
    status.write_text('\n'.join(lines)+'\n')
    B.event('completed',elapsed_seconds=time.time()-start,contrasts=contrasts)


if __name__=='__main__':
    try: main()
    except BaseException as e:
        B.save_json('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        B.registry('failed',f'损失消融执行失败：{type(e).__name__}: {e}；产物保留。')
        raise
