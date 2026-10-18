"""Frozen inference-only TemporalDrift development evaluation; no optimizer/TTA."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, hashlib, json, signal, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from model import MultiViewModel
R=Path(__file__).resolve().parent
ROOT=R.parents[1]
C=json.loads((R/'config.json').read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()

def save(path,data):
    p=R/path;tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def event(stage,**data):
    with (R/'artifacts/exposure.jsonl').open('a') as f:
        f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**data},ensure_ascii=False)+'\n')
    print(stage,json.dumps(data,ensure_ascii=False),flush=True)

def verify():
    assert C['status']=='frozen' and C['roles']==['valid','day30','day150','day270']
    assert not any(C['permissions'][k] for k in ['training','tta','bn_update','target_checkpoint_selection','target_tuning','wtt_time','awf'])
    fr=json.loads((R/'artifacts/freeze.json').read_text())
    for p,h in fr.items(): assert sha(ROOT/p)==h,p
    return len(fr)

def setup():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert torch.cuda.is_available()
    torch.set_num_threads(C['threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False

def reconstruct(x):
    t=np.zeros((len(x),2,1800),np.float32)
    for i,row in enumerate(x):
        a=row[row!=0]; times=np.abs(a).astype(np.float64)
        b=np.minimum(np.floor(times*1799/80),1799).astype(np.int64)
        ch=(a<0).astype(np.int64)
        t[i]=np.bincount(b+1800*ch,minlength=3600).reshape(2,1800)
        independent=np.zeros((2,1800),np.float32);np.add.at(independent,(ch,b),1)
        assert np.array_equal(t[i],independent)
    assert np.array_equal(t[:,0].sum(1),(x>0).sum(1))
    assert np.array_equal(t[:,1].sum(1),(x<0).sum(1))
    return t

def fingerprints(x,direction=False):
    return [hashlib.sha256((np.sign(row) if direction else row).tobytes()).hexdigest() for row in x]

def load_model(seed):
    ck=torch.load(R/f'checkpoints/wide_bn_{seed}_best.pt',map_location='cpu',weights_only=True)
    assert ck['config_sha256']==sha(R/'artifacts/reference/training_config.json')
    history=json.loads((R/f'artifacts/reference/{seed}_history.json').read_text())
    best=max(history,key=lambda h:h['valid']['accuracy'])
    assert ck['step']==best['step']
    m=MultiViewModel('wide_bn').cuda();m.load_state_dict(ck['state_dict'],strict=True)
    m.requires_grad_(False);m.eval();return m,ck['step']

def predict(m,e):
    initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};out=[]
    with torch.inference_mode():
        for i in range(0,len(e['timestamps']),C['eval_batch_size']):
            x=e['timestamps'][i:i+64].cuda();t=e['tam'][i:i+64].cuda()
            z=m(x,t);assert torch.isfinite(z).all();out.append(z.cpu().numpy())
    assert all(torch.equal(v,initial[k]) for k,v in {k:v.cpu() for k,v in m.state_dict().items()}.items())
    return np.concatenate(out)

def metrics(y,p):
    cm=np.bincount(y*102+p,minlength=102*102).reshape(102,102)
    den=cm.sum(0)+cm.sum(1)
    return {'accuracy':float((y==p).mean()),'macro_f1':float(np.divide(2*cm.diagonal(),den,out=np.zeros(102),where=den>0).mean())}

def preflight():
    setup();n=verify();data=torch.load(R/'artifacts/source_valid_prepared.pt',map_location='cpu',weights_only=False)
    valid=data['valid'];x=valid['timestamps'].numpy()
    assert np.array_equal(reconstruct(x),valid['tam'].numpy())
    records=[]
    for s in C['seeds']:
        m,step=load_model(s);z=predict(m,valid)
        with np.load(R/f'artifacts/reference/{s}_logits_best.npz') as f: previous=f['valid'].copy()
        assert np.array_equal(z.argmax(1),previous.argmax(1))
        assert np.allclose(z,previous,rtol=1e-5,atol=1e-5)
        with torch.inference_mode():
            t=valid['tam'][:8].cuda();a=valid['timestamps'][:8].cuda()
            assert torch.equal(m(a,t),m(torch.zeros_like(a),t))
        record={'seed':s,'best_step':step,**metrics(valid['labels'].numpy(),z.argmax(1))}
        records.append(record);del m
    result={'passed':True,'freeze_files':n,'gpu':torch.cuda.get_device_name(0),'valid_rows':len(x),'valid_tam_exact_parity':True,'checkpoint_selection_verified':True,'valid_predictions_exact_parity':True,'packet_view_inactive':True,'records':records}
    save('artifacts/preflight.json',result);event('source_valid_preflight',**result)

def prepare_targets():
    raw=torch.load(R/'artifacts/source_valid_prepared.pt',map_location='cpu',weights_only=False)
    refs={mode:{role:set(fingerprints(raw[role]['timestamps'].numpy(),mode=='direction')) for role in ['source','valid']} for mode in ['timestamp','direction']}
    datasets=json.loads((ROOT/C['dataset_config']).read_text());base=Path(datasets['data_root'])/datasets['datasets'][C['dataset_key']]['path']
    manifest={};target_fp={}
    for role in ['day30','day150','day270']:
        path=base/f'{role}.npz';event('structure_access_started',role=role,file=str(path),label_permission='coverage audit and later fixed scoring only')
        raw_sha=sha(path)
        with np.load(path,allow_pickle=False) as f:
            x=np.ascontiguousarray(f['X'][:,:5000],dtype=np.float32)
            y0=f['y'];y=np.asarray(y0,dtype=np.int64)
            assert np.array_equal(y0,y)
        assert x.ndim==2 and x.shape[1]==5000 and y.shape==(len(x),)
        assert np.isfinite(x).all() and ((x!=0).sum(1)>0).all()
        assert not (((x==0).cumsum(1)>0)&(x!=0)).any(),'internal zero'
        assert set(np.unique(y))==set(range(102)),'class coverage/mapping changed'
        t=reconstruct(x);rows=np.arange(len(x),dtype=np.int64)
        torch.save({'timestamps':torch.from_numpy(x),'tam':torch.from_numpy(t),'labels':torch.from_numpy(y),'rows':torch.from_numpy(rows)},R/f'artifacts/{role}_prepared.pt')
        overlaps={}
        for mode in refs:
            fp=fingerprints(x,mode=='direction');target_fp.setdefault(mode,{})[role]=set(fp)
            overlaps[mode]={ref:int(sum(h in hs for h in fp)) for ref,hs in refs[mode].items()}
            overlaps[mode]['within_date_duplicate_rows']=len(fp)-len(set(fp))
        manifest[role]={'file':str(path),'raw_sha256':raw_sha,'rows':len(x),'raw_row_indices':'arange stored in prepared.pt','class_counts':np.bincount(y,minlength=102).tolist(),'prepared_sha256':sha(R/f'artifacts/{role}_prepared.pt'),'tam_crosscheck_rows':len(x),'overlap_rows':overlaps,'packets_ge80':int((np.abs(x)>=80).sum()),'input_nonfinite':0,'empty_rows':0,'internal_zero_rows':0}
        event('structure_completed',role=role,rows=len(x),overlap_rows=overlaps)
    manifest['cross_date_unique_overlap']={mode:{a: {b: len(d[a] & d[b]) for b in d if b != a} for a in d} for mode,d in target_fp.items()}
    save('artifacts/data_manifest.json',manifest)
    save('artifacts/score_freeze.json',{'manifest_sha256':sha(R/'artifacts/data_manifest.json'),'prepared':{role:manifest[role]['prepared_sha256'] for role in C['roles'] if role != 'valid'}})
    return raw['valid']

def registry(state,summary):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',R.name,'--state',state,'--summary',summary],check=True)
    p=ROOT/'STATUS.md';lines=p.read_text().splitlines();prefix=f'当前Day30/Day150/Day270漂移诊断：`{R.name}`。'
    replacement=prefix+summary
    if any(line.startswith(prefix) for line in lines):lines=[replacement if line.startswith(prefix) else line for line in lines]
    else:lines.insert(1,replacement)
    p.write_text('\n'.join(lines)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    if a.preflight:preflight();return
    assert json.loads((R/'artifacts/preflight.json').read_text())['passed']
    started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('3600s pipeline budget')));signal.alarm(C['budget']['max_wall_seconds'])
    try:
        setup();verify();registry('running','GPU0已启动；冻结wide+BN CE三seed，仅valid/Day30/Day150/Day270评分，无训练/TTA。')
        valid=prepare_targets();reports=[]
        for seed in C['seeds']:
            m,step=load_model(seed)
            for role in C['roles']:
                e=valid if role=='valid' else torch.load(R/f'artifacts/{role}_prepared.pt',map_location='cpu',weights_only=True)
                event('performance_scoring_started',seed=seed,role=role,rows=len(e['labels']))
                z=predict(m,e);y=e['labels'].numpy();p=z.argmax(1)
                np.savez_compressed(R/f'artifacts/predictions_{seed}_{role}.npz',rows=e['rows'].numpy(),labels=y,predictions=p,logits=z)
                report={'seed':seed,'role':role,'best_step':step,'rows':len(y),**metrics(y,p)};reports.append(report)
                save('artifacts/scores.json',reports);event('performance_scoring_completed',**report)
            del m
        subprocess.run([sys.executable,str(R/'postrun_audit.py')],check=True,timeout=max(1,C['budget']['max_wall_seconds']-int(time.monotonic()-started)))
        summary={}
        for role in C['roles']:
            summary[role]={}
            for metric in ['accuracy','macro_f1']:
                vals=np.array([q[metric] for q in reports if q['role']==role]);base=np.array([q[metric] for q in reports if q['role']=='valid'])
                summary[role][metric]={'mean':float(vals.mean()),'sample_std':float(vals.std(ddof=1)),'valid_minus_role_per_seed_pp':((base-vals)*100).tolist(),'mean_drop_pp':float((base-vals).mean()*100)}
        save('artifacts/summary.json',{'seeds':C['seeds'],'metrics':summary,'elapsed_seconds':time.monotonic()-started,'audit_passed':True})
        text='# Day30 / Day150 / Day270 冻结基线评分结果\n\n完成；独立指标与checkpoint重载审计通过。无训练、TTA、BN更新或目标选模。\n\n| 数据 | Accuracy均值±样本SD | Macro-F1均值±样本SD | Accuracy下降pp | F1下降pp |\n|---|---:|---:|---:|---:|\n'
        for role,d in summary.items():
            a=d['accuracy'];f=d['macro_f1'];text+=f"| {role} | {a['mean']*100:.3f} ± {a['sample_std']*100:.3f}% | {f['mean']*100:.3f} ± {f['sample_std']*100:.3f}% | {a['mean_drop_pp']:.3f} | {f['mean_drop_pp']:.3f} |\n"
        text+='\n逐seed及每日期样本数见 artifacts/scores.json；完整预测及logits、原始行号均在artifacts。数据hash、类别计数、训练/valid/跨日期精确指纹重叠见data_manifest.json；重叠未按成绩过滤。输入和权重来源见SOURCES.md及provenance.json。\n\n该差值以反复选模的源期510行valid作参照，不是独立Day0测试，也不是时间因素的因果效应。三seed方差不代表跨日期或数据抽样的不确定性。Day30/Day150/Day270现为本方法开发暴露；结果不能作为未经反馈的独立确认。未评价其他日期、WTT-Time或AWF。\n'
        (R/'RESULTS.md').write_text(text)
        s='完成且审计通过；'+ '/'.join(f"{role} accuracy {d['accuracy']['mean']*100:.3f}% F1 {d['macro_f1']['mean']*100:.3f}%" for role,d in summary.items())+'；仅开发评分，无训练/TTA。'
        registry('completed',s);event('completed',elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};save('artifacts/failure.json',failure)
        with (R/'RESULTS.md').open('a') as f:f.write('\n执行失败；保留产物。详见 artifacts/failure.json。\n')
        registry('failed',f'评分/审计失败：{type(exc).__name__}: {exc}；保留全部暴露和产物。');raise

if __name__=='__main__':main()
