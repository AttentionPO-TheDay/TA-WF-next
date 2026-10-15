"""Unlabeled generator-token selective adaptation screening."""
import hashlib, json, math, os, subprocess, sys, time, traceback, signal
from pathlib import Path
import numpy as np
import torch
from torch import nn

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
C = json.loads((R/'config.json').read_text())
MODEL = R/'model.py'
sys.path.insert(0, str(R))
from model import MultiViewModel

SEEDS = C['seeds']
ROLES = C['roles']
CONDITIONS = C['conditions']
DEVICE = torch.device('cuda:0')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save_json(name, value):
    p=R/'artifacts'/name; tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)

def event(stage, **payload):
    with (R/'artifacts/events.jsonl').open('a') as f:
        f.write(json.dumps({'utc':time.time(),'stage':stage,**payload},ensure_ascii=False)+'\n')
    print(stage, json.dumps(payload,ensure_ascii=False), flush=True)

def metric(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102)
    den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0)
    return {'accuracy':float((y==p).mean()),'macro_f1':float(f.mean())}

def fp_rows(x):
    return [hashlib.sha256(np.ascontiguousarray(row,dtype=np.float32).tobytes()).hexdigest() for row in x]

def reconstruct(x):
    out=np.zeros((len(x),2,1800),np.float32)
    for i,row in enumerate(x):
        a=row[row!=0]; b=np.minimum(np.floor(np.abs(a).astype(np.float64)*1799/80),1799).astype(np.int64)
        c=(a<0).astype(np.int64); out[i]=np.bincount(b+1800*c,minlength=3600).reshape(2,1800)
    return out

def load_prepared(path, role=None):
    d=torch.load(ROOT/path,map_location='cpu',weights_only=False)
    if 'timestamps' not in d:
        d=d[role or 'source']
    x=d['timestamps'].numpy().astype(np.float32,copy=False)
    tam=d['tam'].numpy().astype(np.float32,copy=False)
    y=d['labels'].numpy().astype(np.int64,copy=False)
    rows=d['rows'].numpy().astype(np.int64,copy=False)
    assert x.ndim==2 and x.shape[1]==5000 and tam.shape==(len(x),2,1800)
    assert y.shape==(len(x),) and rows.shape==(len(x),)
    assert np.isfinite(x).all() and np.isfinite(tam).all() and (x!=0).any(1).all()
    assert np.array_equal(reconstruct(x),tam)
    return {'timestamps':x,'tam':tam,'labels':y,'rows':rows}

class Adapter(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.dim=dim; self.preserve_fixed=False; self.norm=nn.LayerNorm(dim); self.down=nn.Linear(dim,16); self.act=nn.GELU(); self.up=nn.Linear(16,dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self,x): return self.up(self.act(self.down(self.norm(x))))

def residual(adapter,temporal,gate):
    if adapter.dim==110:
        delta=adapter(temporal)
        if adapter.preserve_fixed:
            delta=torch.cat([torch.zeros_like(delta[:,:,:30]),delta[:,:,30:]],dim=-1)
    else:
        delta=torch.cat([torch.zeros_like(temporal[:,:,:30]),adapter(temporal[:,:,30:])],dim=-1)
    return gate[...,None]*delta

def load_model(seed):
    ck=ROOT/'runs/20261005T233151Z_wide_bn_day14_day90_eval_94f6ef85/checkpoints'/f'wide_bn_{seed}_best.pt'
    m=MultiViewModel('wide_bn').to(DEVICE)
    state=torch.load(ck,map_location='cpu',weights_only=True)['state_dict']; m.load_state_dict(state,strict=True); m.eval(); m.requires_grad_(False)
    return m,ck

def forward_logits(m,x,tam,adapter=None,gate=None):
    g=m.generator(x,tam); packet,temporal=g['packet'],g['time']; pv,tv=g['packet_mask'],g['time_mask']
    if adapter is not None:
        temporal=temporal+residual(adapter,temporal,gate)
    p=m.packet_projection(packet)+m.packet_positions+m.view_types[0]
    t=m.time_projection(temporal)+m.time_positions+m.view_types[1]
    valid=torch.cat([pv,tv],1); padding=~valid; z=torch.cat([p,t],1).masked_fill(padding[...,None],0)
    for layer in m.head:
        z=layer(z,src_key_padding_mask=padding); z=z.masked_fill(padding[...,None],0)
    z=m.final_norm(z).masked_fill(padding[...,None],0)
    pp=z[:,:100]; tt=z[:,100:]
    pm=(pp*pv[...,None]).sum(1)/pv.sum(1).clamp_min(1)[...,None]
    tm=(tt*tv[...,None]).sum(1)/tv.sum(1).clamp_min(1)[...,None]
    return m.classifier(torch.cat([pm,tm],-1))

@torch.no_grad()
def source_stats(m, source):
    chosen=np.argsort(fp_rows(source['timestamps']))[:C['source_reference_rows']]
    total=0; s=None; ss=None
    for i in range(0,len(chosen),C['eval_batch_size']):
        ix=chosen[i:i+C['eval_batch_size']]
        x=torch.from_numpy(source['timestamps'][ix]).to(DEVICE); t=torch.from_numpy(source['tam'][ix]).to(DEVICE)
        z=m.generator(x,t)['time'].float(); s=z.sum(0) if s is None else s+z.sum(0); ss=(z*z).sum(0) if ss is None else ss+(z*z).sum(0); total+=len(ix)
    mean=s/total; std=(ss/total-mean.square()).clamp_min(1e-4).sqrt()
    return mean,std,chosen.tolist()

@torch.no_grad()
def drift_gate(m,x,tam,mean,std,mode,rowids,seed,role):
    gates=[]
    for i in range(0,len(x),C['eval_batch_size']):
        xb=torch.from_numpy(x[i:i+C['eval_batch_size']]).to(DEVICE); tb=torch.from_numpy(tam[i:i+C['eval_batch_size']]).to(DEVICE)
        n=len(xb)
        if mode in ('full_global','full_masked_learned_global','learned_global'): g=torch.ones(n,120,device=DEVICE)
        else:
            raw=m.generator(xb,tb)['time']; score=((raw[:,:,30:]-mean[:,30:])/std[:,30:]).abs().mean(-1)
            if mode=='learned_random_selective':
                # Deterministic label-free pseudo-random scores tied to row id and token index.
                rid=torch.as_tensor(rowids[i:i+n],device=DEVICE,dtype=torch.float64)[:,None]
                tok=torch.arange(120,device=DEVICE,dtype=torch.float64)[None,:]
                score=torch.frac(torch.sin((rid+1)*(tok+3)*(seed+17))*43758.5453).float()
            idx=score.topk(C['adapter']['selective_tokens'],dim=1).indices; g=torch.zeros_like(score); g.scatter_(1,idx,1.)
        assert torch.isfinite(g).all()
        expected=120 if mode in ('full_global','full_masked_learned_global','learned_global') else 60
        assert torch.equal(g.sum(1),g.new_full((len(g),),expected))
        gates.append(g.cpu())
    return torch.cat(gates,0)

@torch.no_grad()
def predict(m,x,tam,adapter=None,gates=None):
    out=[]
    for i in range(0,len(x),C['eval_batch_size']):
        xb=torch.from_numpy(x[i:i+C['eval_batch_size']]).to(DEVICE); tb=torch.from_numpy(tam[i:i+C['eval_batch_size']]).to(DEVICE)
        g=None if gates is None else gates[i:i+C['eval_batch_size']].to(DEVICE)
        out.append(forward_logits(m,xb,tb,adapter,g).cpu())
    return torch.cat(out).numpy()

def entropy(z):
    p=torch.softmax(z,1); return -(p*p.clamp_min(1e-8).log()).sum(1).mean()

def adapt(m,init_state,x,tam,gates,seed,role,condition):
    a=Adapter(110 if condition.startswith('full_') else 80).to(DEVICE); a.preserve_fixed=condition=='full_masked_learned_global'; a.load_state_dict(init_state); a.train()
    opt=torch.optim.AdamW(a.parameters(),lr=C['adaptation']['learning_rate'],weight_decay=C['adaptation']['weight_decay'])
    before=[]; after=[]; n=len(x); bs=C['adaptation']['batch_size']; history=[]
    for step in range(C['adaptation']['steps']):
        ix=(np.arange(bs)+step*bs)%n
        xb=torch.from_numpy(x[ix]).to(DEVICE); tb=torch.from_numpy(tam[ix]).to(DEVICE); gb=gates[ix].to(DEVICE)
        with torch.enable_grad():
            a.train(); z=forward_logits(m,xb,tb,a,gb); loss=entropy(z); assert torch.isfinite(loss); opt.zero_grad(set_to_none=True); loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())
            opt.step()
        history.append(float(loss.detach().cpu()))
    a.eval();
    with torch.no_grad():
        for i in range(0,n,bs):
            ix=np.arange(i,min(i+bs,n)); xb=torch.from_numpy(x[ix]).to(DEVICE);tb=torch.from_numpy(tam[ix]).to(DEVICE);gb=gates[ix].to(DEVICE)
            before.append(float(entropy(forward_logits(m,xb,tb)).cpu())); after.append(float(entropy(forward_logits(m,xb,tb,a,gb)).cpu()))
    state={k:v.detach().cpu() for k,v in a.state_dict().items()}
    delta_ss=0.; raw_ss=0.; fixed_ss=0.; learned_ss=0.; raw_fixed_ss=0.; raw_learned_ss=0.; count=0
    with torch.no_grad():
        for i in range(0,n,bs):
            ix=np.arange(i,min(i+bs,n)); xb=torch.from_numpy(x[ix]).to(DEVICE); tb=torch.from_numpy(tam[ix]).to(DEVICE); gb=gates[ix].to(DEVICE)
            raw=m.generator(xb,tb)['time']; delta=residual(a,raw,gb)
            delta_ss+=float(delta.double().square().sum()); raw_ss+=float(raw.double().square().sum())
            fixed_ss+=float(delta[:,:,:30].double().square().sum()); learned_ss+=float(delta[:,:,30:].double().square().sum())
            raw_fixed_ss+=float(raw[:,:,:30].double().square().sum()); raw_learned_ss+=float(raw[:,:,30:].double().square().sum()); count+=len(raw)*120
            if a.dim==80 or a.preserve_fixed: assert torch.count_nonzero(delta[:,:,:30])==0
            assert torch.count_nonzero(delta[gb==0])==0
    return a,state,{'support_entropy_before':float(np.mean(before)),'support_entropy_after':float(np.mean(after)),'relative_feature_change':math.sqrt(delta_ss/max(raw_ss,1e-12)),'fixed_delta_rms':math.sqrt(fixed_ss/(count*30)),'learned_delta_rms':math.sqrt(learned_ss/(count*80)),'fixed_relative_change':math.sqrt(fixed_ss/max(raw_fixed_ss,1e-12)),'learned_relative_change':math.sqrt(learned_ss/max(raw_learned_ss,1e-12)),'adapter_parameters':sum(p.numel() for p in a.parameters()),'loss_history':history}

def split_targets(source,role,target):
    source_set=set(fp_rows(source['timestamps']))
    fps=fp_rows(target['timestamps']); first={}
    source_overlap=0
    for i,h in enumerate(fps):
        if h in source_set:
            source_overlap+=1
            continue
        first.setdefault(h,i)
    keep=np.array(sorted(first.values()),dtype=np.int64)
    keys=[fps[i] for i in keep]; order=np.argsort([hashlib.sha256(f'{role}|{h}|20261009'.encode()).hexdigest() for h in keys])
    keep=keep[order]; support=keep[:1024]; query=keep[1024:]
    assert len(set(fps[i] for i in support))==len(support); assert set(fps[i] for i in support).isdisjoint(set(fps[i] for i in query))
    assert len(query)>0
    return support,query,{'raw_rows':len(target['timestamps']),'source_overlap_excluded':source_overlap,'unique_after_source_exclusion':len(keep),'support_rows':len(support),'query_rows':len(query),'query_classes':int(len(np.unique(target['labels'][query]))),'query_fp_unique':len(set(fps[i] for i in query))}

def cell_metrics(y,base_p,new_p):
    base=metric(y,base_p); new=metric(y,new_p); bc=base_p==y; bw=~bc
    return {**new,'base_accuracy':base['accuracy'],'base_macro_f1':base['macro_f1'],'negative_adaptation_rate':float(((bc)&(new_p!=y)).sum()/max(1,bc.sum())),'recovery_rate':float(((bw)&(new_p==y)).sum()/max(1,bw.sum())),'changed_rate':float((new_p!=base_p).mean())}

def registry(state,summary):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',R.name,'--state',state,'--summary',summary],check=True)

def main():
    started=time.time(); torch.set_num_threads(C['threads']); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False
    assert C['status']=='frozen' and C['id']==R.name
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('pipeline time budget exhausted')))
    signal.alarm(C['budget']['max_wall_seconds'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    refs=json.loads((ROOT/'configs/datasets.json').read_text())
    src_path=Path(C['prepared']['source_valid']); source_train=load_prepared(src_path,'source'); source_valid=load_prepared(src_path,'valid')
    source_all={k:np.concatenate([source_train[k],source_valid[k]],axis=0) for k in source_train}
    targets={role:load_prepared(Path(C['prepared'][role])) for role in ROLES}
    splits={}; split_report={}
    for role,t in targets.items():
        s,q,report=split_targets(source_all,role,t); splits[role]=(s,q); split_report[role]=report
        np.savez_compressed(R/'artifacts'/f'split_{role}.npz',support=t['rows'][s],query=t['rows'][q],support_indices=s,query_indices=q)
        with np.load(ROOT/'runs/20261009T014641Z_generator_token_selective_tta_b7b531fa/artifacts'/f'pred_21729_{role}_frozen.npz') as previous:
            assert np.array_equal(t['rows'][q],previous['rows'])
    save_json('split_manifest.json',split_report)
    files=list(R.glob('*.py'))+[R/'config.json',R/'PLAN.md',R/'SOURCES.md',ROOT/'configs/datasets.json']+[ROOT/p for p in C['prepared'].values()]+[ROOT/C['checkpoint_run']/'checkpoints'/f'wide_bn_{seed}_best.pt' for seed in SEEDS]
    freeze={str(p.relative_to(ROOT)):sha(p) for p in files}; save_json('freeze.json',freeze); event('preflight',gpu=torch.cuda.get_device_name(0),split=split_report)
    registry('running','GPU1执行中；固定统计与学习特征分离适应，3日期×3seed×5自适应条件；query隔离与10步预算已冻结。')
    all_rows=[]
    for seed in SEEDS:
        m,ck=load_model(seed); mean,std,source_indices=source_stats(m,source_train)
        original_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        torch.save({'mean':mean.cpu(),'std':std.cpu(),'source_indices':source_indices},R/'artifacts'/f'source_stats_{seed}.pt')
        init_states={}
        for dim in (110,80):
            torch.manual_seed(seed+91000); init=Adapter(dim); init_states[dim]={k:v.detach().cpu().clone() for k,v in init.state_dict().items()}
        with torch.no_grad():
            xx=torch.from_numpy(source_train['timestamps'][:8]).to(DEVICE);tt=torch.from_numpy(source_train['tam'][:8]).to(DEVICE); assert torch.allclose(forward_logits(m,xx,tt),m(xx,tt),atol=1e-5,rtol=1e-5)
        for role in ROLES:
            t=targets[role]; support,query=splits[role]; sx,st=t['timestamps'][support],t['tam'][support]; qx,qt=t['timestamps'][query],t['tam'][query]; qy=t['labels'][query]
            gates={c:drift_gate(m,sx,st,mean,std,c,support,seed,role) for c in CONDITIONS if c!='frozen'}
            qg={c:drift_gate(m,qx,qt,mean,std,c,query,seed,role) for c in CONDITIONS if c!='frozen'}
            base_z=predict(m,qx,qt); base_p=base_z.argmax(1)
            with np.load(ROOT/'runs/20261009T014641Z_generator_token_selective_tta_b7b531fa/artifacts'/f'pred_{seed}_{role}_frozen.npz') as previous: assert np.array_equal(base_p,previous['predictions'])
            np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_frozen.npz',rows=t['rows'][query],labels=qy,predictions=base_p,logits=base_z)
            all_rows.append({'seed':seed,'role':role,'condition':'frozen','steps':0,'rows':len(qy),**metric(qy,base_p),'negative_adaptation_rate':0.,'recovery_rate':0.,'changed_rate':0.})
            for condition in [c for c in CONDITIONS if c!='frozen']:
                dim=110 if condition.startswith('full_') else 80
                a,state,adapt_report=adapt(m,init_states[dim],sx,st,gates[condition],seed,role,condition); z=predict(m,qx,qt,a,qg[condition]); p=z.argmax(1)
                for k,v in m.state_dict().items(): assert torch.equal(v.cpu(),original_state[k]),k
                assert all(p.grad is None for p in m.parameters())
                assert np.isfinite(z).all()
                row={'seed':seed,'role':role,'condition':condition,'steps':C['adaptation']['steps'],'rows':len(qy),'gate_coverage_support':float(gates[condition].mean()),'gate_coverage_query':float(qg[condition].mean()),**cell_metrics(qy,base_p,p),**adapt_report}
                all_rows.append(row)
                torch.save({'state_dict':state,'dim':dim,'preserve_fixed':a.preserve_fixed,'seed':seed,'role':role,'condition':condition},R/'checkpoints'/f'adapter_{seed}_{role}_{condition}.pt')
                torch.save({'support':gates[condition],'query':qg[condition]},R/'artifacts'/f'gates_{seed}_{role}_{condition}.pt')
                np.savez_compressed(R/'artifacts'/f'pred_{seed}_{role}_{condition}.npz',rows=t['rows'][query],labels=qy,predictions=p,logits=z)
                del a
            save_json('scores.json',all_rows); event('cell_completed',seed=seed,role=role)
        del m; torch.cuda.empty_cache()
    save_json('scores.json',all_rows)
    registry('running','全部54组预测和45个适配器任务完成；正在独立复核指标、门控、冻结hash及checkpoint重载。')
    subprocess.run([sys.executable,str(R/'postrun_audit.py')],check=True)
    # Aggregate only after all predictions and audits are present.
    def mean(cond,key): return float(np.mean([r[key] for r in all_rows if r['condition']==cond]))
    sel=[r for r in all_rows if r['condition']=='learned_drift_selective']; glob=[r for r in all_rows if r['condition']=='learned_global']; ran=[r for r in all_rows if r['condition']=='learned_random_selective']
    verdict={'learned_global_minus_full_global_macro_f1_mean_pp':(mean('learned_global','macro_f1')-mean('full_global','macro_f1'))*100,'selective_minus_global_macro_f1_mean_pp':(mean('learned_drift_selective','macro_f1')-mean('learned_global','macro_f1'))*100,'selective_minus_random_macro_f1_mean_pp':(mean('learned_drift_selective','macro_f1')-mean('learned_random_selective','macro_f1'))*100,'selective_minus_global_negative_rate_pp':(mean('learned_drift_selective','negative_adaptation_rate')-mean('learned_global','negative_adaptation_rate'))*100,'positive_cells_vs_global':sum(a['macro_f1']>b['macro_f1'] for a,b in zip(sel,glob)),'positive_cells_vs_random':sum(a['macro_f1']>b['macro_f1'] for a,b in zip(sel,ran))}
    verdict['pass']=verdict['learned_global_minus_full_global_macro_f1_mean_pp']>0 and verdict['selective_minus_global_macro_f1_mean_pp']>0 and verdict['positive_cells_vs_global']>=6 and verdict['selective_minus_global_negative_rate_pp']<0 and verdict['selective_minus_random_macro_f1_mean_pp']>0
    save_json('verdict.json',verdict)
    lines=['# 生成器固定统计与学习特征分离适应结果','','全部 3 个训练 seed × 3 个开发日期完成；query 与无标签 adaptation reference 按精确输入指纹隔离，目标标签仅用于预测后评分。','','| 条件 | Macro-F1均值 | Accuracy均值 | 负适应率均值 | 特征相对改变量 |','|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        change=0 if c=='frozen' else mean(c,'relative_feature_change')
        lines.append(f"| {c} | {mean(c,'macro_f1')*100:.3f}% | {mean(c,'accuracy')*100:.3f}% | {mean(c,'negative_adaptation_rate')*100:.3f}% | {change:.6f} |")
    lines += ['', '裁决 JSON：', '```json', json.dumps(verdict,ensure_ascii=False,indent=2), '```', '', '逐单元结果见 artifacts/scores.json；完整logits、门控、适配器checkpoint、逐行split、源统计与冻结hash均保留。', '', 'full_masked_learned_global 是与完整110维适配器同结构、同初始化的输出屏蔽对照，用于区分固定统计保护与适配器维度变化。学习分支的三个条件使用相同80维结构和初始化。', '', '相对改变量在support上用完整110维输入与残差的L2比计算；只记录幅度，尚未做幅度匹配，因此安全性差异不能单独归因于选区质量。', '', '本轮仅为TemporalDrift开发筛查，未来日期已观察；固定统计不是已证明的稳定因素，本轮未重新训练生成器，未访问WTT-Time、AWF、Network/Behavior/Version目标。']
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n'); registry('completed',f"完成且审计通过；drift_selective相对global Macro-F1 {verdict['selective_minus_global_macro_f1_mean_pp']:.3f}pp，负适应率差 {verdict['selective_minus_global_negative_rate_pp']:.3f}pp，预注册裁决={verdict['pass']}。")
    event('completed',elapsed_seconds=time.time()-started,verdict=verdict)

if __name__=='__main__':
    try: main()
    except BaseException as e:
        save_json('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}); registry('failed',f'执行失败：{type(e).__name__}: {e}；产物保留。'); raise
