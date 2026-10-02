from pathlib import Path
import os,sys,json,hashlib,time,subprocess,traceback
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from ta_wf_next.learnable_generator import GeneratorClassifier
from ta_wf_next.transformer_proto import GeneratorTokenBatch
from df_reference import DFReference

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def savej(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def savet(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def registry(state,summary):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
def status(summary):
    p=ROOT/'STATUS.md';s=p.read_text();key='当前GPU源期诊断：'
    lines=s.splitlines();lines=[x for x in lines if not x.startswith(key)];lines.insert(1,f'{key}`{RUN.name}`。{summary}');p.write_text('\n'.join(lines)+'\n')
def metrics(y,p):
    cm=np.bincount(y*102+p,minlength=10404).reshape(102,102);den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float(np.mean(y==p)),'macro_f1':float(np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0).mean())}
def subset(e,idx):
    return {'directions':e['directions'][idx],'labels':e['labels'][idx],'original':{k:({a:b[idx] for a,b in v.items()} if isinstance(v,dict) else v[idx]) for k,v in e['original'].items()}}
def device_entry(e):
    return {'directions':e['directions'].float().cuda(),'labels':e['labels'].long().cuda(),'original':{k:({a:b.cuda() for a,b in v.items()} if isinstance(v,dict) else v.cuda()) for k,v in e['original'].items()}}
def logits(m,e,idx,kind):
    d=e['directions'][idx]
    if kind=='df':return m(d[:,None])
    o=e['original'];b=GeneratorTokenBatch(*(o[k][idx] for k in ['packet','packet_mask','runs','runs_mask','windows','windows_mask']),{k:v[idx] for k,v in o['spans'].items()})
    return m(b,d,d!=0)
def predict(m,e,kind):
    m.eval();out=[]
    with torch.inference_mode():
        for i in range(0,len(e['labels']),128):out.append(logits(m,e,slice(i,i+128),kind).argmax(1).cpu().numpy())
    return np.concatenate(out)
def new_model(kind,tiny=False):return (DFReference() if kind=='df' else GeneratorClassifier(dropout=0. if tiny else .1)).cuda()

def train(c,data,kind,seed,tiny=False):
    key=('tiny' if tiny else kind)+'_'+str(seed);out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    e=data['source150'];roles={'source':e} if tiny else {'source':e,'valid':data['valid']}
    if tiny:
        idx=torch.tensor([int(torch.where(e['labels']==k)[0][j]) for k in range(102) for j in range(2)],device='cuda')
        roles={'source':subset(e,idx)};np.save(out/'source_indices.npy',idx.cpu().numpy())
    e=roles['source'];n=len(e['labels']);m=new_model(kind,tiny)
    opt=torch.optim.Adamax(m.parameters(),lr=.002,eps=1e-8) if kind=='df' else torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0. if tiny else .0001)
    bs=128 if kind=='df' else 64
    steps=c['tiny_steps'] if tiny else (30*((n+bs-1)//bs) if kind=='df' else 12800)
    checks=list(range(100,steps+1,100)) if tiny else (c['df_eval_steps'] if kind=='df' else c['transformer_eval_steps'])
    g=torch.Generator().manual_seed(seed+9000);order=[]
    if kind=='df':
        for epoch in range(30):order.extend(torch.randperm(n,generator=g).split(bs))
    else:
        parts=[];count=0;cycle=0
        while count<steps*bs:
            parts.append(torch.randperm(n,generator=torch.Generator().manual_seed(seed+9000+cycle)));count+=n;cycle+=1
        order=list(torch.cat(parts)[:steps*bs].reshape(steps,bs))
    savet(out/'index_stream.pt',order)
    best=-1.;history=[];start=time.monotonic();loss_sum=0.;best_step=0
    print(json.dumps({'task':key,'steps':steps,'samples':n,'parameters':sum(p.numel() for p in m.parameters())}),flush=True)
    for step,idx in enumerate(order[:steps],1):
        if time.monotonic()-start>c['job_seconds']:raise TimeoutError(key+' job budget')
        if time.monotonic()-PIPE_START>c['pipeline_seconds']:raise TimeoutError('pipeline budget')
        if kind!='df' and not tiny:
            lr=.001 if step<=6400 else (.0003 if step<=9600 else .0001)
            for group in opt.param_groups:group['lr']=lr
        m.train();idx=idx.cuda();opt.zero_grad(set_to_none=True);z=logits(m,e,idx,kind);loss=F.cross_entropy(z,e['labels'][idx]);assert torch.isfinite(loss);loss.backward()
        if step==1:
            grad={k:float(p.grad.norm()) for k,p in m.named_parameters() if p.grad is not None};savej(out/'first_gradients.json',grad)
            assert all(np.isfinite(list(grad.values()))) and sum(grad.values())>0
            if kind!='df':assert all(grad['generator.'+k]>0 for k in ['conv1.weight','conv2.weight','score.weight','projection.weight'])
        opt.step();loss_sum+=float(loss.detach())
        if step in checks:
            preds={r:predict(m,v,kind) for r,v in roles.items()};scores={r:metrics(v['labels'].cpu().numpy(),preds[r]) for r,v in roles.items()}
            row={'step':step,'elapsed_seconds':time.monotonic()-start,'ce_since_last':loss_sum/(step-(history[-1]['step'] if history else 0)),**scores};loss_sum=0.;history.append(row);savej(out/'history.json',history)
            score=scores['source' if tiny else 'valid']['accuracy' if tiny else 'macro_f1']
            if score>best:
                best=score;best_step=step;savet(RUN/'checkpoints'/f'{key}_best.pt',{'state_dict':m.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            savet(RUN/'checkpoints'/f'{key}_latest.pt',{'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'config_sha256':sha(RUN/'config.json')})
            savej(RUN/'artifacts/progress.json',{'task':key,'step':step,'total_steps':steps,'best_step':best_step,'elapsed_seconds':time.monotonic()-PIPE_START})
            print(json.dumps({'task':key,**row}),flush=True)
            if tiny and scores['source']['accuracy']>=.99:break
    last={r:predict(m,v,kind) for r,v in roles.items()};np.savez_compressed(out/'predictions_last.npz',**last)
    ck=torch.load(RUN/'checkpoints'/f'{key}_best.pt',map_location='cuda',weights_only=True);m.load_state_dict(ck['state_dict'])
    pred={r:predict(m,v,kind) for r,v in roles.items()};np.savez_compressed(out/'predictions_best.npz',**pred)
    # Fresh model reload and independently recomputed sklearn metrics.
    from sklearn.metrics import accuracy_score,f1_score
    reload=new_model(kind,tiny);reload.load_state_dict(ck['state_dict']);verified=[]
    for r,v in roles.items():
        p=predict(reload,v,kind);assert np.array_equal(p,pred[r]);y=v['labels'].cpu().numpy();s=metrics(y,p)
        assert abs(s['accuracy']-accuracy_score(y,p))<1e-12
        assert abs(s['macro_f1']-f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0))<1e-12
        verified.append(r)
    report={'task':key,'kind':kind,'tiny':tiny,'seed':seed,'best_step':best_step,'steps_completed':step,'elapsed_seconds':time.monotonic()-start,'verified_roles':verified,'parameters':sum(p.numel() for p in m.parameters()),'best':{r:metrics(v['labels'].cpu().numpy(),pred[r]) for r,v in roles.items()},'last':{r:metrics(v['labels'].cpu().numpy(),last[r]) for r,v in roles.items()}}
    savej(out/'report.json',report);print(json.dumps(report),flush=True);del m,reload,opt;torch.cuda.empty_cache();return report

def main():
    c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen' and c['roles']==['source','valid'] and not c['future_access']
    freeze=json.loads((RUN/'artifacts/freeze.json').read_text())
    for p,h in freeze.items():assert sha(ROOT/p)==h,p
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False)
    for role,count in [('source150',150),('valid',5)]:
        e=raw[role];assert torch.equal(torch.bincount(e['labels']),torch.full((102,),count));assert set(e['directions'].unique().tolist())<={-1.,0.,1.}
    hashes=lambda x:[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in x]
    hs=hashes(raw['source150']['directions']);hv=hashes(raw['valid']['directions']);assert not set(hs)&set(hv)
    savej(RUN/'artifacts/input_audit.json',{'source_count':15300,'valid_count':510,'source_unique':len(set(hs)),'valid_unique':len(set(hv)),'source_valid_overlap':0,'source_rows':raw['source150']['rows'].tolist(),'valid_rows':raw['valid']['rows'].tolist()})
    data={r:device_entry(raw[r]) for r in ['source150','valid']};del raw
    reports=[]
    for seed in c['seeds']:reports.append(train(c,data,'transformer',seed,True))
    for seed in c['seeds']:
        for kind in ['df','transformer']:reports.append(train(c,data,kind,seed))
    rows=['# GPU源期拟合与DF参照结果','', '|任务|seed|best step|source accuracy|valid accuracy|valid F1|','|---|---:|---:|---:|---:|---:|']
    for r in reports:
        v=r['best'].get('valid');rows.append(f"|{r['task']}|{r['seed']}|{r['best_step']}|{100*r['best']['source']['accuracy']:.3f}%|"+(f"{100*v['accuracy']:.3f}%|{100*v['macro_f1']:.3f}%|" if v else '—|—|'))
    means={k:{m:float(np.mean([r['best']['valid'][m] for r in reports if r['kind']==k and not r['tiny']])) for m in ['accuracy','macro_f1']} for k in ['df','transformer']}
    tiny_pass=all(r['best']['source']['accuracy']>=.99 for r in reports if r['tiny'])
    delta=[next(r for r in reports if r['task']==f'df_{s}')['best']['valid']['macro_f1']-next(r for r in reports if r['task']==f'transformer_{s}')['best']['valid']['macro_f1'] for s in c['seeds']]
    gate=np.mean(delta)>=.01 and all(x>0 for x in delta)
    summary=f"小集99%拟合门槛={tiny_pass}；DF/Transformer valid F1={means['df']['macro_f1']*100:.3f}/{means['transformer']['macro_f1']*100:.3f}%；DF相对差距门槛={gate}；9任务完成、重载预测及独立指标核验通过。"
    rows.extend(['',summary,'',json.dumps(means,ensure_ascii=False,indent=2),'','限制：固定valid为已观察开发集。DF是官方结构与配方的PyTorch移植，非原Keras逐数值复现；初始化、BN统计实现存在差异。两模型样本、方向信息和20次选模机会匹配，但步数、样本呈现量、参数量及计算不同，不能将差距唯一归因于架构。小集关闭正则，仅诊断可拟合性。无未来评分、无外部评价。GPU结果与历史CPU结果分列，不替换旧结论。'])
    (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');savej(RUN/'artifacts/summary.json',{'reports':reports,'means':means,'tiny_pass':tiny_pass,'df_gap_gate':bool(gate),'delta_f1':delta});registry('completed',summary);status(summary)

PIPE_START=time.monotonic()
if __name__=='__main__':
    try:main()
    except BaseException as e:
        error=traceback.format_exc();(RUN/'logs/error.txt').write_text(error)
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n执行中止，保留所有已完成任务，详见 logs/error.txt。\n'+error)
        registry('failed',str(e));status('执行中止：'+str(e)+'；已完成产物保留，未访问未来数据。');raise
