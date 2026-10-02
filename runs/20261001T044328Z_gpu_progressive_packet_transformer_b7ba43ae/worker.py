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
from model import PacketModel

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def savej(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def savet(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def metrics(y,p):
    cm=np.bincount(y*102+p,minlength=10404).reshape(102,102);den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float(np.mean(y==p)),'macro_f1':float(np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0).mean())}
def subset(e,idx):
    return {'directions':e['directions'][idx],'labels':e['labels'][idx],'original':{k:({a:b[idx] for a,b in v.items()} if isinstance(v,dict) else v[idx]) for k,v in e['original'].items()}}
def device_entry(e):
    return {'directions':e['directions'].float().cuda(),'labels':e['labels'].long().cuda(),'original':{k:({a:b.cuda() for a,b in v.items()} if isinstance(v,dict) else v.cuda()) for k,v in e['original'].items()}}
def logits(m,e,idx,kind):
    return m(e['directions'][idx])
def predict(m,e,kind):
    m.eval();out=[]
    with torch.inference_mode():
        for i in range(0,len(e['labels']),128):out.append(logits(m,e,slice(i,i+128),kind).argmax(1).cpu().numpy())
    return np.concatenate(out)
def new_model(kind,tiny=False):
    assert kind in ['transformer','mlp'] and not tiny
    return PacketModel(head=kind).cuda()

def train(c,data,kind,seed,tiny=False):
    key=kind+'_'+str(seed);out=RUN/'artifacts'/key;out.mkdir(exist_ok=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    assert not tiny
    e=data['source150'];roles={'source':e,'valid':data['valid']}
    e=roles['source'];n=len(e['labels']);m=new_model(kind,tiny)
    opt=torch.optim.AdamW(m.parameters(),lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])
    bs=c['batch_size']
    steps=c['optimizer_steps']
    checks=c['eval_steps']
    initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    savet(out/'initial_state.pt',initial)
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
        if True:
            lr=next(seg[2] for seg in c['lr_schedule'] if seg[0]<=step<=seg[1])
            for group in opt.param_groups:group['lr']=lr
        m.train();idx=idx.cuda();opt.zero_grad(set_to_none=True);z=logits(m,e,idx,kind);loss=F.cross_entropy(z,e['labels'][idx]);assert torch.isfinite(loss);loss.backward()
        if step==1:
            grad={k:float(p.grad.norm()) for k,p in m.named_parameters() if p.grad is not None};savej(out/'first_gradients.json',grad)
            assert all(np.isfinite(list(grad.values()))) and sum(grad.values())>0
            assert any(v>0 for k,v in grad.items() if k.startswith('generator.'))
        opt.step();loss_sum+=float(loss.detach())
        if step % 100 == 0:
            savej(out/'progress.json',{'task':key,'step':step,'total_steps':steps,'best_step':best_step,'elapsed_seconds':time.monotonic()-start})
        if step in checks:
            preds={r:predict(m,v,kind) for r,v in roles.items()};scores={r:metrics(v['labels'].cpu().numpy(),preds[r]) for r,v in roles.items()}
            row={'step':step,'elapsed_seconds':time.monotonic()-start,'ce_since_last':loss_sum/(step-(history[-1]['step'] if history else 0)),**scores};loss_sum=0.;history.append(row);savej(out/'history.json',history)
            score=scores['valid']['macro_f1']
            if score>best:
                best=score;best_step=step;savet(RUN/'checkpoints'/f'{key}_best.pt',{'state_dict':m.state_dict(),'step':step,'config_sha256':sha(RUN/'config.json')})
            savet(RUN/'checkpoints'/f'{key}_latest.pt',{'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'config_sha256':sha(RUN/'config.json')})
            savej(out/'progress.json',{'task':key,'step':step,'total_steps':steps,'best_step':best_step,'elapsed_seconds':time.monotonic()-PIPE_START})
            print(json.dumps({'task':key,**row}),flush=True)
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
    report={'task':key,'kind':kind,'historical_reuse':False,'seed':seed,'best_step':best_step,'steps_completed':step,'elapsed_seconds':time.monotonic()-start,'verified_roles':verified,'parameters':sum(p.numel() for p in m.parameters()),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'index_sha256':sha(out/'index_stream.pt'),'best':{r:metrics(v['labels'].cpu().numpy(),pred[r]) for r,v in roles.items()},'last':{r:metrics(v['labels'].cpu().numpy(),last[r]) for r,v in roles.items()}}
    savej(out/'report.json',report);print(json.dumps(report),flush=True);del m,reload,opt;torch.cuda.empty_cache();return report

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['transformer','mlp'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    c=json.loads((RUN/'config.json').read_text());assert c['status']=='frozen' and c['roles']==['source','valid'] and not c['future_access'] and a.seed in c['seeds']
    freeze=json.loads((RUN/'artifacts/freeze.json').read_text())
    for p,h in freeze.items():assert sha(ROOT/p)==h,p
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False)
    data={r:{'directions':raw[r]['directions'].float().cuda(),'labels':raw[r]['labels'].long().cuda()} for r in ['source150','valid']};del raw
    train(c,data,a.kind,a.seed)

PIPE_START=time.monotonic()
if __name__=='__main__':main()
