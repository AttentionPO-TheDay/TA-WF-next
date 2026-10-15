"""MVI paired factorial; labeled query is used only after each fixed adaptation."""
import hashlib,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import port_base as B
from mechanism import control,keep_mask,set_bn_momentum
R,ROOT,C,DEVICE=B.R,B.ROOT,B.C,B.DEVICE
CONDS=C['conditions']; A=C['adaptation']

def setup():
    assert C['status']=='frozen' and C['id']==R.name
    torch.set_num_threads(C['threads']); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def snapshot_gpu():
    return subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()

def status_update(summary):
    import fcntl
    with (ROOT/'.status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=ROOT/'STATUS.md';lines=p.read_text().splitlines();prefix=f'当前 MVI 无伪标签闭环适应：`{R.name}`。'
        for i,line in enumerate(lines):
            if line.startswith(prefix):lines[i]=prefix+summary;break
        else:lines.insert(1,prefix+summary)
        p.write_text('\n'.join(lines)+'\n')

def loader(data,target=False):
    tensors=[torch.from_numpy(data[k]) for k in ('timestamps','tam')]
    if not target:tensors.append(torch.from_numpy(data['labels']))
    tensors.append(torch.from_numpy(data['rows']))
    return DataLoader(TensorDataset(*tensors),batch_size=128,shuffle=True,drop_last=True,num_workers=0,
                      generator=torch.Generator().manual_seed(2026 if target else 2025))

def take(it,dl):
    try:return next(it),it
    except StopIteration:it=iter(dl);return next(it),it

def clock_start():torch.cuda.synchronize();return time.perf_counter()
def clock_end(start):torch.cuda.synchronize();return time.perf_counter()-start

@torch.no_grad()
def probe(model,x,t):
    model.eval();z=B.forward_with_features(model,x,t)[0]
    h=B.entropy(z).detach();assert torch.isfinite(h).all()
    return h

@torch.no_grad()
def inference_profile(model,inputs):
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.eval();out={}
    for batch_size in (1,128):
        x=torch.from_numpy(inputs['timestamps'][:batch_size]).to(DEVICE)
        t=torch.from_numpy(inputs['tam'][:batch_size]).to(DEVICE)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):B.forward_with_features(model,x,t)
        values=[]
        for _ in range(30):
            start=clock_start();B.forward_with_features(model,x,t);values.append(clock_end(start)*1000)
        out[str(batch_size)]={'device_resident_forward_ms':values,'median_ms':float(np.median(values)),
            'p95_ms':float(np.percentile(values,95)), 'amortized_median_ms_per_trace':float(np.median(values)/batch_size),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    assert all(torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items())
    return out

def train(model,source,support,condition,seed,role,max_steps=None):
    assert set(support)=={'timestamps','tam','rows'} and condition in CONDS
    random.seed(2024);np.random.seed(2024);torch.manual_seed(2024)
    dynamic_lr=condition in ('dynamic_lr_only','mvi_joint','mvi_joint_skip')
    dynamic_bn=condition in ('bn_interp_only','mvi_joint','mvi_joint_skip')
    skip=condition=='mvi_joint_skip'
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    sl=loader(source);tl=loader(support,True);si=iter(sl);ti=iter(tl)
    optimizer=torch.optim.Adam(model.parameters(),lr=A['base_learning_rate'])
    start=clock_start();all_h=[]
    for i in range(0,len(support['rows']),128):
        x=torch.from_numpy(support['timestamps'][i:i+128]).to(DEVICE);t=torch.from_numpy(support['tam'][i:i+128]).to(DEVICE)
        all_h.extend(probe(model,x,t).cpu().tolist())
    initial_probe_seconds=clock_end(start);h0=float(np.mean(all_h));assert h0>0
    rows=[];entropies=[];selected_rows=[];streams={k:hashlib.sha256() for k in ('source','target')}
    nsteps=A['candidate_updates'] if max_steps is None else max_steps
    executed=0;exposed=0;maxgrad=0.;begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(nsteps):
        if time.monotonic()-begin>C['budget']['max_task_seconds']:raise TimeoutError('task budget exceeded')
        sb,si=take(si,sl);tb,ti=take(ti,tl)
        streams['source'].update(sb[-1].numpy().astype('<i8').tobytes());streams['target'].update(tb[-1].numpy().astype('<i8').tobytes())
        sx,st,sy=[v.to(DEVICE) for v in sb[:3]];tx,tt=[v.to(DEVICE) for v in tb[:2]]
        probe_start=clock_start();h=probe(model,tx,tt);hm=float(h.double().mean().cpu());hn=h/math.log(102)
        rule=control(hm,h0,A['base_learning_rate']);lr=rule['learning_rate'] if dynamic_lr else A['base_learning_rate']
        momentum=rule['target_bn_momentum'] if dynamic_bn else .10
        chosen=keep_mask(hn,A['skip_normalized_entropy_min'],A['skip_normalized_entropy_max']) if skip else torch.ones(128,dtype=torch.bool,device=DEVICE)
        retained=int(chosen.sum().cpu());probe_sec=clock_end(probe_start)
        entropies.append(hn.cpu().numpy());selected_rows.append(chosen.cpu().numpy())
        row={'step':step,'entropy_nats':hm,'initial_support_entropy_nats':h0,**rule,'applied_lr':lr,
             'applied_target_bn_momentum':momentum,'retained_target_traces':retained,'executed':retained>0,
             'probe_seconds':probe_sec,'forward_seconds':0.,'backward_seconds':0.,'optimizer_seconds':0.}
        if retained:
            torch.manual_seed(2024+100000+step);optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr']=lr;model.train()
            start=clock_start();set_bn_momentum(model,.10);so,sf=B.forward_with_features(model,sx,st)
            set_bn_momentum(model,momentum);to,tf=B.forward_with_features(model,tx[chosen],tt[chosen]);set_bn_momentum(model,.10)
            marg=to.softmax(1).mean(0)
            terms=[torch.nn.functional.cross_entropy(so,sy),B.entropy(to).mean()+(marg*torch.log(marg+1e-5)).sum(),B.calculate_mmd_loss(sf,tf)]
            loss=sum(terms);assert torch.isfinite(loss);row['forward_seconds']=clock_end(start)
            start=clock_start();loss.backward();row['backward_seconds']=clock_end(start)
            for p in model.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all();maxgrad=max(maxgrad,float(p.grad.abs().max()))
            start=clock_start();optimizer.step();row['optimizer_seconds']=clock_end(start)
            row.update(dict(zip(('source_ce','entropy_diversity','mmd'),[float(v.detach()) for v in terms])))
            row['weighted_total']=float(loss.detach());executed+=1;exposed+=retained
        rows.append(row)
        if max_steps is None and (step%80==79 or step==nsteps-1):B.event('adaptation',seed=seed,role=role,condition=condition,candidates=step+1,executed=executed)
    model.eval();delta=orig=0.;changed=0
    for k,v in model.named_parameters():
        b=initial[k].to(v.device);delta+=float((v.detach().double()-b.double()).square().sum());orig+=float(b.double().square().sum());changed+=int(not torch.equal(v.detach(),b))
    key=f'{seed}_{role}_{condition}'
    if max_steps is None:
        B.save_json(f'update_{key}.json',rows)
        np.savez_compressed(R/'artifacts'/f'entropy_gate_{key}.npz',normalized_entropies=np.stack(entropies),selected=np.stack(selected_rows))
    totals={f'{k}_seconds':sum(row[f'{k}_seconds'] for row in rows) for k in ('probe','forward','backward','optimizer')}
    result={'candidate_updates':nsteps,'executed_updates':executed,'skipped_updates':nsteps-executed,'target_backward_trace_exposures':exposed,
        'source_backward_trace_exposures':executed*128,'initial_support_entropy':h0,'initial_probe_seconds':initial_probe_seconds,
        **totals,'task_seconds':time.monotonic()-begin+initial_probe_seconds,
        'backward_amortized_ms_per_target_trace_exposure':1000*totals['backward_seconds']/max(1,exposed),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'streams':{k:v.hexdigest() for k,v in streams.items()},'relative_parameter_change':math.sqrt(delta/max(orig,1e-12)),
        'changed_parameter_tensors':changed,'max_gradient':maxgrad,'gmm_calls':0,'pseudo_label_calls':0}
    assert executed>0 and changed>0 and maxgrad>0
    return result

def freeze_inputs():
    hist_files=[]
    for alias,relative in C['historical_references'].items():
        p=ROOT/relative;hist_files.append(p)
        old_condition={'frozen':'frozen','proteus_strength025':'strength_025','rollback10':'rollback_10'}[alias]
        hist_files.extend(p.parent/f'pred_{s}_{r}_{old_condition}.npz' for s in C['seeds'] for r in C['roles'])
    files=list(R.glob('*.py'))+[R/'PLAN.md',R/'SOURCES.md',R/'config.json',ROOT/C['dataset_config']]
    files+=[ROOT/p for p in C['prepared'].values()]
    files+=[ROOT/C['base_checkpoint_run']/'checkpoints'/f'wide_bn_{s}_best.pt' for s in C['seeds']]
    files+=[ROOT/C['comparison_run']/'artifacts'/f'split_{r}.npz' for r in C['roles']]
    files+=hist_files
    B.save_json('freeze.json',{str(p.relative_to(ROOT)):B.sha(p) for p in files})

def historical_rows():
    out=[]
    for alias,relative in C['historical_references'].items():
        p=ROOT/relative;old=json.loads(p.read_text());old_condition={'frozen':'frozen','proteus_strength025':'strength_025','rollback10':'rollback_10'}[alias]
        picked=[r for r in old if r['condition']==old_condition];assert len(picked)==9
        for row in picked:
            x={k:row[k] for k in ('seed','role','rows','accuracy','macro_f1','negative_adaptation_rate','recovery_rate','changed_rate')}
            x.update(condition=alias,origin='historical_reference',source_scores=relative,source_condition=old_condition)
            out.append(x);shutil.copy2(p.parent/f"pred_{row['seed']}_{row['role']}_{old_condition}.npz",R/'artifacts'/f"pred_{row['seed']}_{row['role']}_{alias}.npz")
    assert len(out)==27
    return out

def report(scores):
    def avg(c,k,r=None):return float(np.mean([x[k] for x in scores if x['condition']==c and (r is None or x['role']==r)]))
    verdict={};base={(x['seed'],x['role']):x for x in scores if x['condition']=='proteus_strength025'}
    for c in CONDS:
        f=avg(c,'macro_f1');neg=avg(c,'negative_adaptation_rate')
        verdict[c]={'macro_f1':f,'negative_adaptation_rate':neg,'f1_delta_vs_proteus_pp':100*(f-avg('proteus_strength025','macro_f1')),
            'negative_delta_vs_proteus_pp':100*(neg-avg('proteus_strength025','negative_adaptation_rate')),
            'positive_paired_cells':sum(x['macro_f1']>base[(x['seed'],x['role'])]['macro_f1'] for x in scores if x['condition']==c),
            'target_pass':f>.745 and neg<.02,'pre_registered_primary':c=='mvi_joint'}
    B.save_json('verdict.json',verdict)
    lines=['# TemporalDrift MVI 结果','','45个新任务和27条历史参考完成；独立审计通过。主候选为mvi_joint；不根据最好结果重定义主候选。','','| 条件 | Macro-F1 | Accuracy | 负适应率 |','|---|---:|---:|---:|']
    for c in ['frozen','proteus_strength025','rollback10']+CONDS:lines.append(f"| {c} | {avg(c,'macro_f1')*100:.3f}% | {avg(c,'accuracy')*100:.3f}% | {avg(c,'negative_adaptation_rate')*100:.3f}% |")
    for role in C['roles']:
        lines+=['',f'## {role}','','| 条件 | Macro-F1 | 负适应率 |','|---|---:|---:|']
        for c in ['frozen','proteus_strength025','rollback10']+CONDS:lines.append(f"| {c} | {avg(c,'macro_f1',role)*100:.3f}% | {avg(c,'negative_adaptation_rate',role)*100:.3f}% |")
    lines+=['','## 计算成本','','| 条件 | 适应秒/任务 | 反向秒/任务 | target暴露数 | 跳过更新 | 峰值显存GiB |','|---|---:|---:|---:|---:|---:|']
    for c in CONDS:lines.append(f"| {c} | {avg(c,'task_seconds'):.3f} | {avg(c,'backward_seconds'):.3f} | {avg(c,'target_backward_trace_exposures'):.1f} | {avg(c,'skipped_updates'):.1f} | {avg(c,'peak_allocated_bytes')/1024**3:.3f} |")
    lines+=['','逐条控制记录、entropy_gate、support-only batch1/128推理p50/p95见artifacts。GPU共享；反向耗时是batch摊销，不是单条在线时延。静态/历史对照成本不能跨硬件直接作速度倍数比较。','',
        '当前输入依赖时间戳；WTT/AWF及非时间因素尚未评分。完全去除伪标签同时移除第三路前向，不等同上一轮no_pseudo损失消融。','',json.dumps(verdict,ensure_ascii=False,indent=2)]
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
    setup();started=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('batch budget exceeded')));signal.alarm(C['budget']['max_wall_seconds'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available()
    assert not (R/'artifacts/events.jsonl').exists(),'refuse duplicate execution'
    freeze_inputs();B.save_json('hardware_start.json',{'snapshot':snapshot_gpu(),'gpu':torch.cuda.get_device_name(0)})
    B.registry('running','45项MVI适应已启动；3seed×3日期×5条件，每项800候选更新；主候选mvi_joint，外部评分未启动。')
    status_update('45项MVI已启动；完全移除伪标签/GMM，动态lr×BN动量2×2及跳过条件；目标F1>74.5%且负适应<2%。')
    source=B.load_prepared(C['prepared']['source_valid'],'source');valid=B.load_prepared(C['prepared']['source_valid'],'valid');source_all={k:np.concatenate([source[k],valid[k]]) for k in source}
    splits={}
    for role in C['roles']:
        target=B.load_prepared(C['prepared'][role]);s,q,_=B.split_targets(source_all,role,target)
        with np.load(ROOT/C['comparison_run']/'artifacts'/f'split_{role}.npz') as old:
            assert np.array_equal(s,old['support_indices']) and np.array_equal(q,old['query_indices'])
        np.savez_compressed(R/'artifacts'/f'split_{role}.npz',support=target['rows'][s],query=target['rows'][q],support_indices=s,query_indices=q);splits[role]=(s,q)
    scores=historical_rows();B.save_json('scores.json',scores);B.event('preflight_verified',splits_match=True,new_tasks=45,historical_rows=27)
    for seed in C['seeds']:
        for role in C['roles']:
            target=B.load_prepared(C['prepared'][role]);s,q=splits[role];support={k:target[k][s] for k in ('timestamps','tam','rows')};paired=None
            # Query untouched during adaptation/profiling.
            for condition in CONDS:
                model,_=B.load_model(seed);cost=train(model,source,support,condition,seed,role)
                if paired is None:paired=cost['streams']
                else:assert paired==cost['streams']
                cost['inference_profile']=inference_profile(model,support)
                start=clock_start();z=B.predict(model,target['timestamps'][q],target['tam'][q],128);cost['query_inference_seconds']=clock_end(start)
                pred=z.argmax(1);assert np.isfinite(z).all()
                with np.load(R/'artifacts'/f'pred_{seed}_{role}_frozen.npz') as base:
                    assert np.array_equal(base['rows'],target['rows'][q]) and np.array_equal(base['labels'],target['labels'][q]);frozen=base['predictions']
                cost['query_latency_ms_per_trace']=1000*cost['query_inference_seconds']/len(q)
                row={'seed':seed,'role':role,'condition':condition,'origin':'new','rows':len(q),**B.cell_metrics(target['labels'][q],frozen,pred),**cost}
                np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_{condition}.npz',rows=target['rows'][q],labels=target['labels'][q],predictions=pred,logits=z)
                torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'seed':seed,'role':role,'condition':condition,'config_sha':B.sha(R/'config.json')},R/'checkpoints'/f'{seed}_{role}_{condition}.pt')
                scores.append(row);B.save_json('scores.json',scores);completed=sum(x['origin']=='new' for x in scores)
                B.save_json('progress.json',{'completed_tasks':completed,'total_tasks':45,'seed':seed,'role':role,'condition':condition,'elapsed_seconds':time.time()-started})
                B.event('task_completed',seed=seed,role=role,condition=condition,completed=completed);del model;torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(R/'postrun_audit.py')],check=True)
    report(scores);B.save_json('hardware_end.json',{'snapshot':snapshot_gpu()});B.registry('completed','45项MVI新任务、27条历史参考完成并通过审计；主候选及成本报告见RESULTS.md/verdict.json。')
    status_update('45项完成且审计通过；MVI主候选与目标裁决见RESULTS.md，外部/跨因素待方法冻结和模态审计。');B.event('completed',elapsed_seconds=time.time()-started)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        B.save_json('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        B.registry('failed',f'MVI失败：{type(e).__name__}: {e}；产物保留。');status_update(f'执行失败：{type(e).__name__}，已保留产物。');raise
