"""Matched-budget Proteus adaptation with unlabeled interval rollback."""
import copy, hashlib, json, os, random, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import port_base as B
R, ROOT, C, DEVICE = B.R, B.ROOT, B.C, B.DEVICE
CONDITIONS = list(C['conditions'])
HIST = ROOT / C['comparison_run']

def setup():
    assert C['status']=='frozen' and C['id']==R.name
    torch.set_num_threads(C['threads']); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False

def save_json(name, value):
    p=R/'artifacts'/name; q=p.with_suffix('.tmp'); q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); q.replace(p)

def event(stage, **payload):
    with (R/'artifacts/events.jsonl').open('a') as f: f.write(json.dumps({'utc':time.time(),'stage':stage,**payload},ensure_ascii=False)+'\n')
    print(stage,json.dumps(payload,ensure_ascii=False),flush=True)

def registry(state, summary):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',R.name,'--state',state,'--summary',summary],check=True)

def metric(labels,pred):
    cm=np.bincount(labels*102+pred,minlength=102*102).reshape(102,102); den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0)
    return {'accuracy':float((labels==pred).mean()),'macro_f1':float(f.mean())}

def fp_rows(values):
    return [hashlib.sha256(np.ascontiguousarray(row,dtype=np.float32).tobytes()).hexdigest() for row in values]

def reconstruct(values):
    out=np.zeros((len(values),2,1800),np.float32)
    for i,row in enumerate(values):
        nz=row[row!=0]; bins=np.minimum(np.floor(np.abs(nz).astype(np.float64)*1799/80),1799).astype(np.int64)
        out[i]=np.bincount(bins+1800*(nz<0),minlength=3600).reshape(2,1800)
    return out

def load_prepared(path, role=None):
    d=torch.load(ROOT/path,map_location='cpu',weights_only=False); d=d if 'timestamps' in d else d[role or 'source']
    x=d['timestamps'].numpy().astype(np.float32,copy=False); t=d['tam'].numpy().astype(np.float32,copy=False); y=d['labels'].numpy().astype(np.int64,copy=False); rows=d['rows'].numpy().astype(np.int64,copy=False)
    assert x.shape==(len(x),5000) and t.shape==(len(x),2,1800) and y.shape==rows.shape and np.isfinite(x).all() and np.isfinite(t).all() and np.array_equal(reconstruct(x),t)
    return {'timestamps':x,'tam':t,'labels':y,'rows':rows}

def loader(x,t,y,rows,seed):
    g=torch.Generator().manual_seed(seed); ds=TensorDataset(torch.from_numpy(x),torch.from_numpy(t),torch.as_tensor(y),torch.as_tensor(rows))
    return DataLoader(ds,batch_size=128,shuffle=True,drop_last=True,num_workers=0,generator=g)

def next_batch(it, dl):
    try:return next(it),it
    except StopIteration:it=iter(dl);return next(it),it

def clone_cpu(module): return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def cpu_obj(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu_obj(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_obj(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_obj(v) for v in x)
    return copy.deepcopy(x)
def device_obj(x):
    if torch.is_tensor(x): return x.to(DEVICE)
    if isinstance(x,dict): return {k:device_obj(v) for k,v in x.items()}
    if isinstance(x,list): return [device_obj(v) for v in x]
    if isinstance(x,tuple): return tuple(device_obj(v) for v in x)
    return x

def support_monitor(model,support):
    was=model.training; model.eval(); total=0.; high=0; n=len(support['timestamps'])
    with torch.no_grad():
        for i in range(0,n,128):
            x=torch.from_numpy(support['timestamps'][i:i+128]).to(DEVICE); t=torch.from_numpy(support['tam'][i:i+128]).to(DEVICE)
            z=B.forward_with_features(model,x,t)[0]; p=z.softmax(1); total+=float(B.entropy(z).double().sum().cpu()); high+=int((p.amax(1)>=C['adaptation']['confidence_threshold']).sum().cpu())
    if was:model.train()
    return {'entropy':total/n,'high_confidence_rate':high/n}

def parameter_changes(model,initial):
    ds=orig=0.; changed=0; bn=[]
    for k,v in model.named_parameters():
        b=initial[k].to(v.device); ds+=float((v.detach().double()-b.double()).square().sum()); orig+=float(b.double().square().sum()); changed+=int(not torch.equal(v.detach(),b))
    for k in model.state_dict():
        if k.endswith('running_mean') or k.endswith('running_var'): bn.append(not torch.equal(model.state_dict()[k].cpu(),initial[k]))
    return {'relative_parameter_change':(ds/max(orig,1e-12))**.5,'changed_parameter_tensors':changed,'changed_bn_buffers':sum(bn)}

def train(model,source,support,condition,seed,role):
    assert set(support)=={'timestamps','tam','rows'}
    random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
    sl=loader(source['timestamps'],source['tam'],source['labels'],source['rows'],2025)
    tl=loader(support['timestamps'],support['tam'],np.zeros(len(support['rows']),np.int64),support['rows'],2026)
    si=iter(sl); ti=iter(tl); optimizer=torch.optim.Adam(model.parameters(),lr=C['adaptation']['learning_rate'])
    initial=clone_cpu(model); streams={k:hashlib.sha256() for k in ('source','target','pseudo')}; weights=np.ones(4); interval=C['rollback_intervals'][condition]
    histories=[]; gmms=[]; acceptance=[]; attempted_monitor=[]; accepted_monitor=[]; accepted_state=support_monitor(model,support); accepted_monitor.append(accepted_state); updates=0; maxgrad=0.; start=time.monotonic(); pseudo_loader=None; pseudo_it=None
    for epoch in range(100):
        if time.monotonic()-start>C['budget']['max_task_seconds']: raise TimeoutError('task budget exceeded')
        if epoch%5==0:
            report=B.gmm_pseudo_labels(model,support['timestamps'],support['tam']); selected=report['selected']; gmms.append({'epoch':epoch,**{k:v for k,v in report.items() if k not in ('selected','predictions','probabilities')}})
            np.savez_compressed(R/'artifacts'/f"gmm_{seed}_{role}_{condition}_{epoch}.npz",rows=support['rows'],selected=selected,predictions=report['predictions'],probabilities=report['probabilities'])
            pseudo_loader=loader(support['timestamps'][selected],support['tam'][selected],report['predictions'][selected],support['rows'][selected],2030+epoch//5); pseudo_it=iter(pseudo_loader)
        model.train(); totals=np.zeros(4,np.float64); interval_model=None; interval_opt=None
        for bi in range(8):
            if interval and updates%interval==0:
                interval_model=clone_cpu(model); interval_opt=cpu_obj(optimizer.state_dict())
            sb,si=next_batch(si,sl); tb,ti=next_batch(ti,tl); pb,pseudo_it=next_batch(pseudo_it,pseudo_loader)
            for key,batch in zip(('source','target','pseudo'),(sb,tb,pb)): streams[key].update(batch[3].numpy().astype('<i8').tobytes())
            torch.manual_seed(2024+100000+updates)
            sx,st,sy=[v.to(DEVICE) for v in sb[:3]]; tx,tt=[v.to(DEVICE) for v in tb[:2]]; px,pt,py=[v.to(DEVICE) for v in pb[:3]]
            optimizer.zero_grad(set_to_none=True); so,sf=B.forward_with_features(model,sx,st); to,tf=B.forward_with_features(model,tx,tt); po,_=B.forward_with_features(model,px,pt)
            marg=to.softmax(1).mean(0); terms=[torch.nn.functional.cross_entropy(so,sy),torch.nn.functional.cross_entropy(po,py),B.entropy(to).mean()+torch.sum(marg*torch.log(marg+1e-5)),B.calculate_mmd_loss(sf,tf)]
            loss=sum(w*x for w,x in zip(weights,terms)); assert torch.isfinite(loss); loss.backward()
            for p in model.parameters():
                if p.grad is not None: assert torch.isfinite(p.grad).all(); maxgrad=max(maxgrad,float(p.grad.detach().abs().max()))
            optimizer.step(); totals+=np.array([float(x.detach()) for x in terms]); updates+=1
            if interval and updates%interval==0:
                cand=support_monitor(model,support); attempted_monitor.append(cand); ok=cand['entropy']<=accepted_state['entropy']-C['adaptation']['entropy_minimum_improvement'] and cand['high_confidence_rate']+C['adaptation']['confidence_tolerance']>=accepted_state['high_confidence_rate']
                if ok: accepted_state=dict(cand)
                else:
                    model.load_state_dict(interval_model,strict=True); optimizer.load_state_dict(device_obj(interval_opt))
                acceptance.append(bool(ok)); accepted_monitor.append(dict(accepted_state)); model.train()
        histories.append({'epoch':epoch,**dict(zip(C['loss_order'],(totals/8).tolist())),'weighted_total':float((totals/8).sum()),'pseudo_coverage':gmms[-1]['coverage']})
        if epoch%10==0 or epoch==99:event('epoch',seed=seed,role=role,condition=condition,epoch=epoch,updates=updates,accepted=sum(acceptance))
    assert updates==800 and maxgrad>0
    model.eval(); final=support_monitor(model,support); changes=parameter_changes(model,initial)
    return {'updates':updates,'attempted_intervals':len(acceptance),'accepted_steps':int(sum(acceptance)*interval) if interval else 800,'rejected_intervals':int(len(acceptance)-sum(acceptance)),'acceptance_history':acceptance,'support_monitor_attempted':attempted_monitor,'support_monitor_accepted':accepted_monitor,'support_entropy_before':accepted_monitor[0]['entropy'],'support_entropy_after':final['entropy'],'support_high_confidence_before':accepted_monitor[0]['high_confidence_rate'],'support_high_confidence_after':final['high_confidence_rate'],'loss_history':histories,'gmm_history':gmms,'streams':{k:v.hexdigest() for k,v in streams.items()},'task_seconds':time.monotonic()-start,'max_gradient':maxgrad,'peak_memory_bytes':torch.cuda.max_memory_allocated(),**changes}

def split_targets(source,role,target):
    s=set(fp_rows(source['timestamps'])); fps=fp_rows(target['timestamps']); first={};
    for i,f in enumerate(fps):
        if f not in s:first.setdefault(f,i)
    keep=np.array(sorted(first.values()),np.int64); keys=[fps[i] for i in keep]; order=np.argsort([hashlib.sha256(f'{role}|{f}|20261009'.encode()).hexdigest() for f in keys]); keep=keep[order]
    sup=keep[:1024]; q=keep[1024:]; assert len(set(fps[i] for i in sup))==len(sup) and set(fps[i] for i in sup).isdisjoint(set(fps[i] for i in q)); return sup,q

def cell_metrics(y,base,pred):
    b=metric(y,base); a=metric(y,pred); bc=base==y; bw=~bc
    return {**a,'base_accuracy':b['accuracy'],'base_macro_f1':b['macro_f1'],'negative_adaptation_rate':float((bc&(pred!=y)).sum()/max(1,bc.sum())),'recovery_rate':float((bw&(pred==y)).sum()/max(1,bw.sum())),'changed_rate':float((base!=pred).mean())}

def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def freeze_inputs():
    files=list(R.glob('*.py'))+[R/'PLAN.md',R/'SOURCES.md',R/'config.json',ROOT/'configs/datasets.json']+[ROOT/p for p in C['prepared'].values()]
    files += [ROOT/C['base_checkpoint_run']/'checkpoints'/f'wide_bn_{s}_best.pt' for s in C['seeds']]
    files += [HIST/'artifacts'/f'split_{r}.npz' for r in C['roles']]
    save_json('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in files})

def main():
    setup(); started=time.time(); signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('batch budget exceeded'))); signal.alarm(C['budget']['max_wall_seconds']); assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available(); assert not (R/'artifacts/events.jsonl').exists()
    freeze_inputs(); source=B.load_prepared(C['prepared']['source_valid'],'source'); valid=B.load_prepared(C['prepared']['source_valid'],'valid'); source_all={k:np.concatenate([source[k],valid[k]]) for k in source}; targets={r:B.load_prepared(C['prepared'][r]) for r in C['roles']}; splits={}
    for role,t in targets.items():
        s,q=split_targets(source_all,role,t); old=np.load(HIST/'artifacts'/f'split_{role}.npz'); assert np.array_equal(s,old['support_indices']) and np.array_equal(q,old['query_indices']); old.close(); np.savez_compressed(R/'artifacts'/f'split_{role}.npz',support=t['rows'][s],query=t['rows'][q],support_indices=s,query_indices=q); splits[role]=(s,q)
    event('preflight',gpu=C['physical_gpu'],tasks=36,split_match=True); registry('running','GPU0单进程执行36项；4条件×3seed×3日期，每项800次候选更新，support无标签接受/回退。')
    scores=[]
    for seed in C['seeds']:
        for role in C['roles']:
            t=targets[role]; s,q=splits[role]; support={k:t[k][s] for k in ('timestamps','tam','rows')}; qx=t['timestamps'][q]; qt=t['tam'][q]; y=t['labels'][q]; frozen_model,_=B.load_model(seed); frozen_logits=B.predict(frozen_model,qx,qt,128); base=frozen_logits.argmax(1); np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_frozen.npz',rows=t['rows'][q],labels=y,predictions=base,logits=frozen_logits); scores.append({'seed':seed,'role':role,'condition':'frozen','origin':'historical_reference','updates':0,'rows':len(y),**metric(y,base),'negative_adaptation_rate':0.,'recovery_rate':0.,'changed_rate':0.}); del frozen_model
            paired=None
            for condition in CONDITIONS:
                model,_=B.load_model(seed); torch.cuda.reset_peak_memory_stats(); result=train(model,source,support,condition,seed,role); z=B.predict(model,qx,qt,128); pred=z.argmax(1); assert np.isfinite(z).all();
                if paired is None: paired=result['streams']
                else: assert paired['source']==result['streams']['source'] and paired['target']==result['streams']['target']
                row={'seed':seed,'role':role,'condition':condition,'origin':'new','rows':len(y),**cell_metrics(y,base,pred),**result}; scores.append(row); np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_{condition}.npz',rows=t['rows'][q],labels=y,predictions=pred,logits=z); torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'seed':seed,'role':role,'condition':condition},R/'checkpoints'/f'{seed}_{role}_{condition}.pt'); save_json('scores.json',scores); event('task_completed',seed=seed,role=role,condition=condition,completed=sum(x['origin']=='new' for x in scores)); del model; torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(R/'postrun_audit.py')],check=True); report_results(scores); registry('completed','36项新任务完成且审计通过；比较固定候选预算下的无标签回退。'); status=ROOT/'STATUS.md'; lines=status.read_text().splitlines(); marker=f'当前 Proteus 无标签接受与回退（800候选更新）：`{R.name}`。';
    for i,line in enumerate(lines):
        if line.startswith(marker): lines[i]=marker+'36项完成且审计通过；结果见RESULTS.md。'
    status.write_text('\n'.join(lines)+'\n'); event('completed',elapsed_seconds=time.time()-started)

def report_results(scores):
    def avg(c,k,r=None):return float(np.mean([x[k] for x in scores if x['condition']==c and (r is None or x['role']==r)]))
    control={(x['seed'],x['role']):x for x in scores if x['condition']=='full_control'}; contrasts={}
    for c in CONDITIONS[1:]:
        rows=[x for x in scores if x['condition']==c]; f=(avg(c,'macro_f1')-avg('full_control','macro_f1'))*100; n=(avg(c,'negative_adaptation_rate')-avg('full_control','negative_adaptation_rate'))*100; pos=sum(x['macro_f1']>control[(x['seed'],x['role'])]['macro_f1'] for x in rows); contrasts[c]={'macro_f1_delta_pp':f,'negative_delta_pp':n,'positive_cells':pos,'candidate_signal':f>=.5 and n<=-1 and pos>=6}
    save_json('contrasts.json',contrasts); lines=['# Proteus 800 候选更新无标签接受与回退结果','','所有36项新适应任务完成；每项均尝试800次候选更新，回退只使用support无标签熵与高置信度比例。','', '| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 | 接受更新均值 |','|---|---:|---:|---:|---:|---:|']
    for c in ['frozen']+CONDITIONS: lines.append(f"| {c} | {avg(c,'macro_f1')*100:.3f}% | {avg(c,'accuracy')*100:.3f}% | {avg(c,'negative_adaptation_rate')*100:.3f}% | {avg(c,'recovery_rate')*100:.3f}% | {0 if c=='frozen' else avg(c,'accepted_steps'):.1f} |")
    for role in C['roles']:
        lines += ['',f'## {role}','','| 条件 | Macro-F1 | 负适应率 | 接受更新 |','|---|---:|---:|---:|']
        for c in ['frozen']+CONDITIONS: lines.append(f"| {c} | {avg(c,'macro_f1',role)*100:.3f}% | {avg(c,'negative_adaptation_rate',role)*100:.3f}% | {0 if c=='frozen' else avg(c,'accepted_steps',role):.1f} |")
    lines += ['','```json',json.dumps(contrasts,ensure_ascii=False,indent=2),'```','', '当前结论仅限 TemporalDrift 已观察开发数据和 wide+BN 上的 Proteus 移植；不等同官方 RF 复现。']
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    try: main()
    except BaseException as e:
        save_json('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}); registry('failed',f'回退实验失败：{type(e).__name__}: {e}；产物保留。'); raise
