"""MVI paired factorial; labeled query is used only after each fixed adaptation."""
import copy,hashlib,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import port_base as B
from mechanism import update_teacher,StageCapture,topology_loss,WindowController,set_bn_momentum
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
        p=ROOT/'STATUS.md';lines=p.read_text().splitlines();prefix=f'当前图片实验 {CONDS[0]}：`{R.name}`。'
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
    assert set(support)=={'timestamps','tam','rows'} and condition in CONDS+['no_pseudo_fixed']
    random.seed(2024);np.random.seed(2024);torch.manual_seed(2024)
    teacher=copy.deepcopy(model).eval() if condition=='ema_anchor' else None
    if teacher is not None:teacher.requires_grad_(False)
    capture=StageCapture(model) if condition=='scale_topology' else None
    for name,param in model.named_parameters():
        if condition=='wide_bn_only': ok=name.startswith('generator.')
        elif condition=='attention_bn_only': ok=('.self_attn.' in name) or (name.startswith('generator.') and '.norm' in name)
        else: ok=True
        param.requires_grad_(ok)
    trainable_names=[n for n,p in model.named_parameters() if p.requires_grad]; assert trainable_names
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    sl=loader(source);tl=loader(support,True);si=iter(sl);ti=iter(tl)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=A['base_learning_rate'])
    start=clock_start();all_h=[]
    for i in range(0,len(support['rows']),128):
        x=torch.from_numpy(support['timestamps'][i:i+128]).to(DEVICE);t=torch.from_numpy(support['tam'][i:i+128]).to(DEVICE)
        all_h.extend(probe(model,x,t).cpu().tolist())
    initial_probe_seconds=clock_end(start);h0=float(np.mean(all_h));assert h0>0
    window=WindowController(h0,C['mechanism'].get('length',16),C['mechanism'].get('stride',4),C['mechanism'].get('gain',4.))
    rows=[];entropies=[];selected_rows=[];streams={k:hashlib.sha256() for k in ('source','target')}
    nsteps=A['candidate_updates'] if max_steps is None else max_steps
    executed=0;exposed=0;maxgrad=0.;begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(nsteps):
        if time.monotonic()-begin>C['budget']['max_task_seconds']:raise TimeoutError('task budget exceeded')
        sb,si=take(si,sl);tb,ti=take(ti,tl)
        streams['source'].update(sb[-1].numpy().astype('<i8').tobytes());streams['target'].update(tb[-1].numpy().astype('<i8').tobytes())
        sx,st,sy=[v.to(DEVICE) for v in sb[:3]];tx,tt=[v.to(DEVICE) for v in tb[:2]]
        probe_start=clock_start();h=probe(model,tx,tt);hm=float(h.double().mean().cpu());hn=h/math.log(102)
        rule=window.step(hm) if condition=='window_drift' else {'scale':1.,'window_growth':0.,'window_refreshed':False}
        lr=A['base_learning_rate']*rule['scale'];momentum=.10
        chosen=torch.ones(128,dtype=torch.bool,device=DEVICE)
        teacher_logits=None
        if teacher is not None:
            with torch.no_grad():teacher_logits=B.forward_with_features(teacher,tx,tt)[0]
        retained=int(chosen.sum().cpu());probe_sec=clock_end(probe_start)
        entropies.append(hn.cpu().numpy());selected_rows.append(chosen.cpu().numpy())
        row={'step':step,'entropy_nats':hm,'initial_support_entropy_nats':h0,**rule,'applied_lr':lr,
             'applied_target_bn_momentum':momentum,'retained_target_traces':retained,'executed':retained>0,
             'probe_seconds':probe_sec,'forward_seconds':0.,'backward_seconds':0.,'optimizer_seconds':0.}
        if retained:
            torch.manual_seed(2024+100000+step);optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]['lr']=lr;model.train()
            start=clock_start();set_bn_momentum(model,.10);so,sf=B.forward_with_features(model,sx,st)
            if capture is not None:capture.clear()
            set_bn_momentum(model,momentum);to,tf=B.forward_with_features(model,tx[chosen],tt[chosen]);set_bn_momentum(model,.10)
            marg=to.softmax(1).mean(0)
            terms=[torch.nn.functional.cross_entropy(so,sy),B.entropy(to).mean()+(marg*torch.log(marg+1e-5)).sum(),B.calculate_mmd_loss(sf,tf)]
            extra=to.new_zeros(())
            if teacher is not None:extra=torch.nn.functional.kl_div(to.log_softmax(1),teacher_logits.softmax(1),reduction='batchmean')
            elif capture is not None:extra=topology_loss(capture.values,C['mechanism']['bins'])
            weight=C['mechanism'].get('weight',0.) if condition!='no_pseudo_fixed' else 0.
            loss=sum(terms)+weight*extra;assert torch.isfinite(loss);row['extra_loss']=float(extra.detach());row['extra_weight']=weight;row['forward_seconds']=clock_end(start)
            start=clock_start();loss.backward();row['backward_seconds']=clock_end(start)
            for p in model.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all();maxgrad=max(maxgrad,float(p.grad.abs().max()))
            start=clock_start();optimizer.step()
            if teacher is not None:update_teacher(teacher,model,C['mechanism']['decay'])
            row['optimizer_seconds']=clock_end(start)
            row.update(dict(zip(('source_ce','entropy_diversity','mmd'),[float(v.detach()) for v in terms])))
            row['weighted_total']=float(loss.detach());executed+=1;exposed+=retained
        rows.append(row)
        if max_steps is None and (step%80==79 or step==nsteps-1):B.event('adaptation',seed=seed,role=role,condition=condition,candidates=step+1,executed=executed)
    if capture is not None:capture.close()
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
        'changed_parameter_tensors':changed,'trainable_parameter_tensors':len(trainable_names),'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'max_gradient':maxgrad,'gmm_calls':0,'pseudo_label_calls':0}
    assert executed>0 and changed>0 and maxgrad>0
    return result

def freeze_inputs():
    hist_files=[]
    for alias,relative in C['historical_references'].items():
        p=ROOT/relative;hist_files.append(p)
        old_condition=alias
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
        p=ROOT/relative;old=json.loads(p.read_text());old_condition=alias
        picked=[r for r in old if r['condition']==old_condition];assert len(picked)==9
        for row in picked:
            x={k:row[k] for k in ('seed','role','rows','accuracy','macro_f1','negative_adaptation_rate','recovery_rate','changed_rate')}
            x.update(condition=alias,origin='historical_reference',source_scores=relative,source_condition=old_condition)
            out.append(x);shutil.copy2(p.parent/f"pred_{row['seed']}_{row['role']}_{old_condition}.npz",R/'artifacts'/f"pred_{row['seed']}_{row['role']}_{alias}.npz")
    assert len(out)==27
    return out

def report(scores):
    def avg(c,k,role=None):return float(np.mean([x[k] for x in scores if x['condition']==c and (role is None or x['role']==role)]))
    base={(x['seed'],x['role']):x for x in scores if x['condition']=='no_pseudo_fixed'}
    verdict={}
    for c in CONDS:
        f=avg(c,'macro_f1');n=avg(c,'negative_adaptation_rate')
        verdict[c]={'macro_f1':f,'negative_adaptation_rate':n,'f1_delta_vs_fixed_pp':100*(f-avg('no_pseudo_fixed','macro_f1')),'negative_delta_vs_fixed_pp':100*(n-avg('no_pseudo_fixed','negative_adaptation_rate')),'positive_paired_cells':sum(x['macro_f1']>base[x['seed'],x['role']]['macro_f1'] for x in scores if x['condition']==c),'screen_pass':f>avg('no_pseudo_fixed','macro_f1') and n<.02,'aspirational_goal_pass':f>.745 and n<.02}
    B.save_json('verdict.json',verdict)
    lines=['# 图片实验结果','','9个新适应任务与27条历史参考完成，postrun审计通过。历史参考不计独立重复。','','| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |','|---|---:|---:|---:|---:|']
    aliases=list(C['historical_references'])+CONDS
    for c in aliases:lines.append(f"| {c} | {avg(c,'macro_f1')*100:.3f}% | {avg(c,'accuracy')*100:.3f}% | {avg(c,'negative_adaptation_rate')*100:.3f}% | {avg(c,'recovery_rate')*100:.3f}% |")
    for role in C['roles']:
        lines+=['',f'## {role}','','| 条件 | Macro-F1 | 负适应率 |','|---|---:|---:|']
        for c in aliases:lines.append(f"| {c} | {avg(c,'macro_f1',role)*100:.3f}% | {avg(c,'negative_adaptation_rate',role)*100:.3f}% |")
    for c in CONDS:
        lines+=['',f"适应秒/任务 {avg(c,'task_seconds'):.3f}；反向秒/任务 {avg(c,'backward_seconds'):.3f}；query摊销毫秒/trace {avg(c,'query_latency_ms_per_trace'):.3f}；峰值分配GiB {avg(c,'peak_allocated_bytes')/1024**3:.3f}；更新数 {avg(c,'executed_updates'):.0f}。"]
    lines+=['','负适应率=静态模型原本正确、适应后错误的样本数 / 静态正确样本数。纠错率=适应后纠正的样本数 / 静态错误样本数。','support-only batch1/128计时、逐更新控制/损失记录见artifacts；共享GPU上仅报告描述性时间，不作跨批次速度倍率结论。','仅TemporalDrift已观察开发证据。目标数据不是按时间排序流，Exp-4仅检验适应优化过程的窗口反馈，不能声称检测了真实在线漂移速率。','EMA与一致性为已有通用思想；当前实验只检验本模型收益，不证明论文创新。','初版错误与保留产物见repair_history.md。',json.dumps(verdict,ensure_ascii=False,indent=2)]
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
    setup();started=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('batch budget exceeded')));signal.alarm(C['budget']['max_wall_seconds'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available()
    assert not (R/'artifacts/events.jsonl').exists(),'refuse duplicate execution'
    freeze_inputs();B.save_json('hardware_start.json',{'snapshot':snapshot_gpu(),'gpu':torch.cuda.get_device_name(0)})
    B.registry('running','9项图片实验已启动；3seed×3日期，每项800候选更新；外部评分未启动。')
    status_update('9项实现v2已启动；无伪标签、固定配对数据和800次更新，目标见PLAN。')
    source=B.load_prepared(C['prepared']['source_valid'],'source');valid=B.load_prepared(C['prepared']['source_valid'],'valid');source_all={k:np.concatenate([source[k],valid[k]]) for k in source}
    splits={}
    for role in C['roles']:
        target=B.load_prepared(C['prepared'][role]);s,q,_=B.split_targets(source_all,role,target)
        with np.load(ROOT/C['comparison_run']/'artifacts'/f'split_{role}.npz') as old:
            assert np.array_equal(s,old['support_indices']) and np.array_equal(q,old['query_indices'])
        np.savez_compressed(R/'artifacts'/f'split_{role}.npz',support=target['rows'][s],query=target['rows'][q],support_indices=s,query_indices=q);splits[role]=(s,q)
    scores=historical_rows();B.save_json('scores.json',scores);B.event('preflight_verified',splits_match=True,new_tasks=9,historical_rows=27)
    for seed in C['seeds']:
        for role in C['roles']:
            target=B.load_prepared(C['prepared'][role]);s,q=splits[role];support={k:target[k][s] for k in ('timestamps','tam','rows')};paired=None
            # Query untouched during adaptation/profiling.
            for condition in CONDS:
                model,_=B.load_model(seed);cost=train(model,source,support,condition,seed,role)
                historical=next(x for x in json.loads((ROOT/C['historical_references']['no_pseudo_fixed']).read_text()) if x['seed']==seed and x['role']==role and x['condition']=='no_pseudo_fixed')
                assert historical['streams']==cost['streams']
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
                B.save_json('progress.json',{'completed_tasks':completed,'total_tasks':9,'seed':seed,'role':role,'condition':condition,'elapsed_seconds':time.time()-started})
                B.event('task_completed',seed=seed,role=role,condition=condition,completed=completed);del model;torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(R/'postrun_audit.py')],check=True)
    report(scores);B.save_json('hardware_end.json',{'snapshot':snapshot_gpu()});B.registry('completed','9项新适应与27条历史参考完成，指标与checkpoint重载审计通过；结果见RESULTS.md/verdict.json。')
    status_update('9项完成且审计通过；裁决与成本见RESULTS.md，仍仅开发证据。');B.event('completed',elapsed_seconds=time.time()-started)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        B.save_json('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        B.registry('failed',f'图片实验失败：{type(e).__name__}: {e}；产物保留。');status_update(f'执行失败：{type(e).__name__}，已保留产物。');raise
